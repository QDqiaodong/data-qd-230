"""Disk-backed stable sort and budgeted full-join operators for keyed rows."""
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
import bisect
import heapq
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class Row:
    key: int
    payload: str


def _entries(rows, max_payload, row_limit=None):
    for seq, row in enumerate(rows):
        if row_limit is not None and seq >= row_limit:
            raise ValueError(f"row {seq}: exceeds the {row_limit}-row limit")
        if (not isinstance(row, Row) or type(row.key) is not int
                or not -(1 << 63) <= row.key < (1 << 63)
                or not isinstance(row.payload, str)):
            raise ValueError(f"row {seq}: expected signed 64-bit key and string payload")
        if len(row.payload) > max_payload or len(row.payload.encode("utf-8")) > max_payload:
            raise ValueError(f"row {seq}: payload exceeds {max_payload} UTF-8 bytes")
        yield (row.key, seq, row.payload)


def _write(path, entries):
    with path.open("w", encoding="ascii", newline="\n") as out:
        for entry in entries:
            out.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n")


def _read(stream):
    for line in stream:
        yield tuple(json.loads(line))


def _merge(paths):
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open(encoding="ascii")) for path in paths]
        yield from heapq.merge(*(_read(stream) for stream in streams))


def _output(entries):
    for key, _, payload in entries:
        yield Row(key, payload)


@contextmanager
def sorted_rows(rows, *, run_rows=1024, fan_in=8, max_payload=65536, scratch=None):
    """Yield a sorted iterator valid only within its with-block."""
    if (any(type(n) is not int for n in (run_rows, fan_in, max_payload))
            or run_rows < 1 or fan_in < 2 or max_payload < 1):
        raise ValueError("run_rows >= 1, fan_in >= 2 and max_payload >= 1 required")
    with TemporaryDirectory(dir=scratch, prefix="row-sort-") as base:
        root = Path(base)

        def path(level, index):
            return root / f"{level}-{index}.jsonl"

        entries = _entries(rows, max_payload)
        count = 0
        while True:
            chunk = list(islice(entries, run_rows))
            if not chunk:
                break
            chunk.sort()
            _write(path(0, count), chunk)
            count += 1
        level = 0
        while count > fan_in:
            written = 0
            for start in range(0, count, fan_in):
                group = [path(level, i) for i in range(start, min(start + fan_in, count))]
                with closing(_merge(group)) as merged:
                    _write(path(level + 1, written), merged)
                for old in group:
                    old.unlink()
                written += 1
            count, level = written, level + 1
        with closing(_merge([path(level, i) for i in range(count)])) as merged:
            with closing(_output(merged)) as output:
                yield output


# ---------------------------------------------------------------------------
# Budgeted full join
#
# The join consumes each side exactly once and spills key-sorted copies of
# both inputs.  Admission is decided eagerly, in ascending key order, before
# any cursor is handed out; the cursors themselves only replay the spilled
# input rows and apply that decision.  Joined output is generated pair by
# pair, so a huge same-key group never has to be materialised.

_JOIN_ROW_LIMIT = 200_000
_MAX_PAIR_BUDGET = 1_000_000_000


def _check_join_params(pair_budget, run_rows, fan_in, max_payload):
    if (any(type(n) is not int for n in (pair_budget, run_rows, fan_in, max_payload))
            or not 0 <= pair_budget <= _MAX_PAIR_BUDGET
            or not 1 <= run_rows <= 256
            or not 2 <= fan_in <= 16
            or not 1 <= max_payload <= 4096):
        raise ValueError(
            "pair_budget in [0, 1000000000], run_rows in [1, 256], "
            "fan_in in [2, 16] and max_payload in [1, 4096] required")


def _spill_side(rows, root, name, *, run_rows, fan_in, max_payload):
    """Sort one input side into ``root/name.sorted.jsonl``; return its path."""

    def part(level, index):
        return root / f"{name}-{level}-{index}.jsonl"

    entries = _entries(rows, max_payload, row_limit=_JOIN_ROW_LIMIT)
    count = 0
    while True:
        chunk = list(islice(entries, run_rows))
        if not chunk:
            break
        chunk.sort()
        _write(part(0, count), chunk)
        count += 1
    level = 0
    while count > fan_in:
        written = 0
        for start in range(0, count, fan_in):
            group = [part(level, i) for i in range(start, min(start + fan_in, count))]
            with closing(_merge(group)) as merged:
                _write(part(level + 1, written), merged)
            for old in group:
                old.unlink()
            written += 1
        count, level = written, level + 1
    final = root / f"{name}.sorted.jsonl"
    if count == 0:
        final.write_bytes(b"")
    else:
        with closing(_merge([part(level, i) for i in range(count)])) as merged:
            _write(final, merged)
        for i in range(count):
            part(level, i).unlink()
    return final


def _line_key(line):
    """Extract the key from a spilled ``[key,seq,payload]`` line (as bytes)."""
    return int(line[1:line.index(b",", 1)])


def _row_from_line(line):
    entry = json.loads(line)
    return Row(entry[0], entry[2])


class _LineReader:
    """Buffered one-line-ahead reader over a spilled side, in binary mode."""

    __slots__ = ("_file", "_pending")

    def __init__(self, path):
        self._file = open(path, "rb")
        self._pending = None

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def peek(self):
        if self._pending is None:
            self._pending = self._file.readline() or None
        return self._pending

    def pop(self):
        line = self._pending
        if line is None:
            return self._file.readline() or None
        self._pending = None
        return line

    def position(self):
        if self._pending is None:
            return self._file.tell()
        return self._file.tell() - len(self._pending)

    def seek(self, offset):
        self._pending = None
        self._file.seek(offset)

    def readline(self):
        self._pending = None
        return self._file.readline()


def _counts(path):
    """Yield ``(key, count)`` for each key group of a key-sorted side."""
    with open(path, "rb") as stream:
        key = None
        count = 0
        for line in stream:
            found = _line_key(line)
            if key is None:
                key, count = found, 1
            elif found == key:
                count += 1
            else:
                yield (key, count)
                key, count = found, 1
        if key is not None:
            yield (key, count)


def _write_decisions(path, left_path, right_path, pair_budget):
    """Admit keys ascending while the shared budget covers their full result."""
    with closing(_counts(left_path)) as left_counts, \
            closing(_counts(right_path)) as right_counts, \
            path.open("w", encoding="ascii", newline="\n") as out:
        left = next(left_counts, None)
        right = next(right_counts, None)
        budget = pair_budget
        while left is not None or right is not None:
            if right is None or (left is not None and left[0] < right[0]):
                key, lcount, rcount = left[0], left[1], 0
                left = next(left_counts, None)
            elif left is None or right[0] < left[0]:
                key, lcount, rcount = right[0], 0, right[1]
                right = next(right_counts, None)
            else:
                key, lcount, rcount = left[0], left[1], right[1]
                left = next(left_counts, None)
                right = next(right_counts, None)
            required = lcount * rcount if lcount and rcount else lcount + rcount
            accepted = required <= budget
            if accepted:
                budget -= required
            out.write(json.dumps([key, 1 if accepted else 0], separators=(",", ":")) + "\n")


def _decisions(path):
    with open(path, "rb") as stream:
        for line in stream:
            entry = json.loads(line)
            yield (entry[0], bool(entry[1]))


def _joined_rows(decision_path, left_path, right_path):
    left = _LineReader(left_path)
    right = _LineReader(right_path)
    reread = _LineReader(right_path)
    try:
        with closing(_decisions(decision_path)) as decisions:
            for key, accepted in decisions:
                for reader in (left, right):
                    line = reader.peek()
                    while line is not None and _line_key(line) < key:
                        reader.pop()
                        line = reader.peek()
                if not accepted:
                    for reader in (left, right):
                        while (line := reader.peek()) is not None and _line_key(line) == key:
                            reader.pop()
                    continue
                left_line = left.peek()
                if left_line is not None and _line_key(left_line) == key:
                    right_line = right.peek()
                    if right_line is not None and _line_key(right_line) == key:
                        start = right.position()
                        while (left_line := left.peek()) is not None and _line_key(left_line) == key:
                            left.pop()
                            left_row = _row_from_line(left_line)
                            reread.seek(start)
                            while (right_line := reread.readline()):
                                if _line_key(right_line) != key:
                                    break
                                yield (left_row, _row_from_line(right_line))
                        while (right_line := right.peek()) is not None and _line_key(right_line) == key:
                            right.pop()
                    else:
                        while (left_line := left.peek()) is not None and _line_key(left_line) == key:
                            left.pop()
                            yield (_row_from_line(left_line), None)
                else:
                    while (right_line := right.peek()) is not None and _line_key(right_line) == key:
                        right.pop()
                        yield (None, _row_from_line(right_line))
    finally:
        left.close()
        right.close()
        reread.close()


def _deferred_rows(decision_path, rows_path):
    rows = _LineReader(rows_path)
    try:
        with closing(_decisions(decision_path)) as decisions:
            for key, accepted in decisions:
                line = rows.peek()
                while line is not None and _line_key(line) < key:
                    rows.pop()
                    line = rows.peek()
                if accepted:
                    while (line := rows.peek()) is not None and _line_key(line) == key:
                        rows.pop()
                else:
                    while (line := rows.peek()) is not None and _line_key(line) == key:
                        rows.pop()
                        yield _row_from_line(line)
    finally:
        rows.close()


@contextmanager
def budgeted_full_join(left, right, *, pair_budget, run_rows=128, fan_in=8,
                       max_payload=4096, scratch=None):
    """Yield ``(joined, deferred_left, deferred_right)`` cursors for one batch.

    Each key is admitted in ascending key order when the remaining pair budget
    covers its complete result (left*right pairs, or the single side's row
    count); otherwise the whole key is deferred to the matching cursor, which
    replays that side's original rows.  Cursors are one-shot and valid only
    inside the with-block.
    """
    _check_join_params(pair_budget, run_rows, fan_in, max_payload)
    with TemporaryDirectory(dir=scratch, prefix="row-join-") as base:
        root = Path(base)
        left_path = _spill_side(left, root, "left", run_rows=run_rows,
                                fan_in=fan_in, max_payload=max_payload)
        right_path = _spill_side(right, root, "right", run_rows=run_rows,
                                 fan_in=fan_in, max_payload=max_payload)
        decision_path = root / "decisions.jsonl"
        _write_decisions(decision_path, left_path, right_path, pair_budget)
        joined = _joined_rows(decision_path, left_path, right_path)
        deferred_left = _deferred_rows(decision_path, left_path)
        deferred_right = _deferred_rows(decision_path, right_path)
        try:
            yield joined, deferred_left, deferred_right
        finally:
            joined.close()
            deferred_left.close()
            deferred_right.close()


# ---------------------------------------------------------------------------
# Partial-run version compaction
#
# Two one-shot version streams (the selected run set being replaced and the
# untouched run sets) are each validated and externally sorted once.  A single
# streaming merge then walks every key group in (key, revision) order and
# decides, against at most 32 registered snapshot revisions plus the "latest"
# read point, which selected originals still have to ship in the replacement
# stream.  The batch cursors afterwards read only the spilled replacement
# records and the untouched spill: the original selected input is never kept
# around, neither in memory nor as a hidden file.

_VERSION_ROW_LIMIT = 200_000
_VERSION_MAX_PAYLOAD = 4096
_VERSION_MAX_REVISION = (1 << 63) - 1
_VERSION_MAX_SNAPSHOTS = 32
_LATEST = 1 << 63  # sentinel revision standing for the unbounded latest read


@dataclass(frozen=True)
class Version:
    key: int
    revision: int
    payload: object  # UTF-8-encodable str, or None for a delete marker


def _check_version_params(snapshots, run_rows, fan_in, scratch):
    if not isinstance(snapshots, (list, tuple)) or len(snapshots) > _VERSION_MAX_SNAPSHOTS:
        raise ValueError("snapshots must be a list or tuple of at most 32 revisions")
    seen = set()
    for revision in snapshots:
        if type(revision) is not int or not 0 <= revision <= _VERSION_MAX_REVISION:
            raise ValueError("snapshot revisions must be ints in [0, 2**63 - 1]")
        if revision in seen:
            raise ValueError("snapshot revisions must be unique")
        seen.add(revision)
    if type(run_rows) is not int or not 1 <= run_rows <= 256:
        raise ValueError("run_rows must be an int in [1, 256]")
    if type(fan_in) is not int or not 2 <= fan_in <= 16:
        raise ValueError("fan_in must be an int in [2, 16]")
    if scratch is not None and not isinstance(scratch, (str, Path)):
        raise ValueError("scratch must be None, a str or a Path")


def _version_entries(rows, source):
    """Validate one version stream; yield ``(key, revision, source, payload)``."""
    for seq, row in enumerate(rows):
        if seq >= _VERSION_ROW_LIMIT:
            raise ValueError(f"row {seq}: exceeds the {_VERSION_ROW_LIMIT}-row limit")
        if not isinstance(row, Version):
            raise ValueError(f"row {seq}: expected Version")
        if type(row.key) is not int or not -(1 << 63) <= row.key < (1 << 63):
            raise ValueError(f"row {seq}: key must be a signed 64-bit int")
        if type(row.revision) is not int or not 0 <= row.revision <= _VERSION_MAX_REVISION:
            raise ValueError(f"row {seq}: revision must be an int in [0, 2**63 - 1]")
        if row.payload is not None:
            if not isinstance(row.payload, str):
                raise ValueError(f"row {seq}: payload must be a str or None")
            encoded = row.payload.encode("utf-8")
            if len(encoded) > _VERSION_MAX_PAYLOAD:
                raise ValueError(
                    f"row {seq}: payload exceeds {_VERSION_MAX_PAYLOAD} UTF-8 bytes")
        yield (row.key, row.revision, source, row.payload)


def _version_merge(paths):
    """Merge spills by (key, revision); payloads never participate in ordering."""
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open(encoding="ascii")) for path in paths]
        yield from heapq.merge(*(_read(stream) for stream in streams),
                               key=lambda entry: (entry[0], entry[1]))


def _spill_versions(rows, root, name, source, *, run_rows, fan_in):
    """Sort one version stream into ``root/name.sorted.jsonl`` and return it."""

    def part(level, index):
        return root / f"{name}-{level}-{index}.jsonl"

    entries = _version_entries(rows, source)
    count = 0
    while True:
        chunk = list(islice(entries, run_rows))
        if not chunk:
            break
        chunk.sort(key=lambda entry: (entry[0], entry[1]))
        _write(part(0, count), chunk)
        count += 1
    level = 0
    while count > fan_in:
        written = 0
        for start in range(0, count, fan_in):
            group = [part(level, i) for i in range(start, min(start + fan_in, count))]
            with closing(_version_merge(group)) as merged:
                _write(part(level + 1, written), merged)
            for old in group:
                old.unlink()
            written += 1
        count, level = written, level + 1
    final = root / f"{name}.sorted.jsonl"
    if count == 0:
        final.write_bytes(b"")
    else:
        with closing(_version_merge([part(level, i) for i in range(count)])) as merged:
            _write(final, merged)
        for i in range(count):
            part(level, i).unlink()
    return final


def _version_from_entry(entry):
    return Version(entry[0], entry[1], entry[3])


class _KeyDecision:
    """Streaming per-key sweep shared by all registered points and latest.

    The key group is scanned once in ascending revision.  A monotonic cursor
    over the registered points decides which points each record wins; the
    delivered-view state (``last_kept``: the newest untouched row or kept
    selected row, deletes included) is carried across points, so later points
    reuse earlier decisions instead of rescanning history.

    Inclusion rules:
      * every untouched record belongs to the delivered view;
      * a selected value is included when it wins a registered point or latest;
      * a selected delete is included at the first point where the original
        view is invisible but the delivered view would still show a value --
        once a delete is the delivered winner it masks every older value, so a
        later delete re-covering an invisible point is dropped.
    """

    __slots__ = ("points", "point_index", "last_kept", "shipped", "out")

    def __init__(self, points, out):
        self.points = points              # registered revisions, ascending
        self.point_index = 0              # next registered point to resolve
        self.last_kept = None             # current winner of the delivered view
        self.shipped = None               # identity of the last shipped entry
        self.out = out                    # callable shipping a kept selected row

    def _ship(self, entry):
        # A selected entry can be nominated by several points but must be
        # delivered at most once.
        if self.shipped is not entry:
            self.shipped = entry
            self.out(entry)

    def _covered_points(self, entry, bound):
        """Advance the point cursor across points ``entry`` wins; return count."""
        points, index = self.points, self.point_index
        covered = 0
        while index < len(points) and points[index] <= bound:
            if points[index] >= entry[1]:
                covered += 1
            index += 1
        self.point_index = index
        return covered

    def add(self, entry, covered_through):
        """Fold one record (ascending revision) into the sweep.

        ``covered_through`` is the successor record's revision minus one.
        """
        covered = self._covered_points(entry, covered_through)
        self._transition(entry, covered, latest=False)

    def finish(self, entry):
        """Resolve points still covered by the final record, then latest."""
        covered = self._covered_points(entry, _VERSION_MAX_REVISION)
        self._transition(entry, covered, latest=False)
        # Latest is an additional release point resolved last, once per key.
        self._transition(entry, 1, latest=True)

    def _transition(self, entry, covered, latest):
        source, payload = entry[2], entry[3]
        if source == 1:
            # All untouched rows stay in place, so the delivered view always
            # advances to an untouched record even if it wins no registered
            # point in the original view.
            self.last_kept = entry
            return
        if payload is not None:
            # Rule 1: a selected value winning any release point (latest
            # included) ships and enters the delivered view.
            if covered:
                self._ship(entry)
                self.last_kept = entry
            return
        # Selected delete: rule 2.  Keep it at the first point where the
        # original view is invisible but the delivered view would show a
        # value.  A delete already in the delivered winner keeps masking all
        # older values, so another covering delete is never shipped.
        if covered and self.last_kept is not None and self.last_kept[3] is not None:
            self._ship(entry)
            self.last_kept = entry


def _write_replacement(merged_path, replacement_path, points):
    """One streaming pass over the (key, revision)-sorted two-side merge."""
    with open(merged_path, "rb") as stream, \
            replacement_path.open("w", encoding="ascii", newline="\n") as out:
        def ship(entry):
            out.write(json.dumps(list(entry), ensure_ascii=True,
                                 separators=(",", ":")) + "\n")

        decision = None
        current_key = None
        previous = None
        for line in stream:
            entry = tuple(json.loads(line))
            key = entry[0]
            if key != current_key:
                if decision is not None:
                    decision.finish(previous)
                current_key = key
                decision = _KeyDecision(points, ship)
                previous = entry
                continue
            if entry[1] == previous[1]:
                raise ValueError(
                    f"duplicate revision {entry[1]} for key {key} across inputs")
            decision.add(previous, entry[1] - 1)
            previous = entry
        if decision is not None:
            decision.finish(previous)


class _VersionBatch:
    """Re-openable replacement and point-query cursors over spilled results."""

    def __init__(self, replacement_path, untouched_path, points, root):
        self._replacement = replacement_path
        self._untouched = untouched_path
        self._points = points
        self._root = root
        self._closed = False
        self._active = set()

    def _track(self, generator):
        def wrap():
            try:
                yield from generator
            finally:
                self._active.discard(wrapped)

        wrapped = wrap()
        self._active.add(wrapped)
        return wrapped

    def replacement(self):
        if self._closed:
            raise RuntimeError("version batch has been closed")
        return self._track(self._replacement_cursor())

    def visible(self, at=None):
        if self._closed:
            raise RuntimeError("version batch has been closed")
        if at is not None:
            if type(at) is not int or not 0 <= at <= _VERSION_MAX_REVISION:
                raise ValueError("at must be None or a registered revision")
            index = bisect.bisect_left(self._points, at)
            if index == len(self._points) or self._points[index] != at:
                raise ValueError(f"{at} is not a registered snapshot revision")
        return self._track(self._visible_cursor(at))

    def _replacement_cursor(self):
        with open(self._replacement, "rb") as stream:
            for line in stream:
                yield _version_from_entry(tuple(json.loads(line)))

    def _visible_cursor(self, at):
        bound = _LATEST if at is None else at
        with open(self._untouched, "rb") as untouched_stream, \
                open(self._replacement, "rb") as replacement_stream:
            sources = (
                (tuple(json.loads(line)) for line in untouched_stream),
                (tuple(json.loads(line)) for line in replacement_stream),
            )
            current_key = None
            winner = None
            for entry in heapq.merge(*sources, key=lambda item: (item[0], item[1])):
                key = entry[0]
                if key != current_key:
                    if winner is not None and winner[3] is not None:
                        yield _version_from_entry(winner)
                    current_key = key
                    # Each key starts with no in-bound winner: a key whose
                    # records all exceed the read bound stays invisible and
                    # must not inherit the previous key's winner.
                    winner = None
                # Only in-bound records compete for the visible row; the last
                # such record wins, including when it is a delete.
                if entry[1] <= bound:
                    winner = entry
            if winner is not None and winner[3] is not None:
                yield _version_from_entry(winner)

    def _shutdown(self):
        self._closed = True
        # Any cursor still suspended inside the with-block is closed by the
        # generator close() below; its finally-block drops the registration.
        for token in list(self._active):
            token.close()


# ---------------------------------------------------------------------------
# Late-materialized grouped Top-K
#
# The query node receives a one-shot stream of narrow index hits while the
# wide payloads live on backend data pages.  Materialising every hit before
# ranking would read pages for candidates that later lose, and fetching row
# by row in rank order would re-read pages whose hits interleave.  This
# operator therefore ranks first and reads late: the index stream is
# validated and externally sorted once when the context is entered, the best
# hit per group is elected (highest score, ties by arrival order) and the
# first ``limit`` distinct groups are kept.  While the caller consumes that
# ranking, each backing page is read exactly once -- at the first rank that
# needs it, asking for precisely the slots the selected hits use on the
# page, including slots only needed by later ranks.  Payloads whose delivery
# ranks lie ahead are staged in this call's temp directory, so arbitrarily
# interleaved pages never force pending wide rows into memory.

_RANK_ROW_LIMIT = 200_000
_RANK_MAX_PAYLOAD = 4096


@dataclass(frozen=True)
class Hit:
    group: int
    score: int
    page: int
    slot: int


def _check_rank_params(limit, run_rows, fan_in, scratch):
    if type(limit) is not int or not 0 <= limit <= _RANK_ROW_LIMIT:
        raise ValueError("limit must be an int in [0, 200000]")
    if type(run_rows) is not int or not 1 <= run_rows <= 256:
        raise ValueError("run_rows must be an int in [1, 256]")
    if type(fan_in) is not int or not 2 <= fan_in <= 16:
        raise ValueError("fan_in must be an int in [2, 16]")
    if scratch is not None and not isinstance(scratch, (str, Path)):
        raise ValueError("scratch must be None, a str or a Path")


def _hit_entries(hits):
    """Validate the one-shot hit stream; yield ``[group, -score, seq, page, slot]``."""
    for seq, hit in enumerate(hits):
        if seq >= _RANK_ROW_LIMIT:
            raise ValueError(f"row {seq}: exceeds the {_RANK_ROW_LIMIT}-row limit")
        if not isinstance(hit, Hit):
            raise ValueError(f"row {seq}: expected Hit")
        if type(hit.group) is not int or not -(1 << 63) <= hit.group < (1 << 63):
            raise ValueError(f"row {seq}: group must be a signed 64-bit int")
        if type(hit.score) is not int or not -(1 << 63) <= hit.score < (1 << 63):
            raise ValueError(f"row {seq}: score must be a signed 64-bit int")
        if type(hit.page) is not int or not 0 <= hit.page < (1 << 63):
            raise ValueError(f"row {seq}: page must be an int in [0, 2**63 - 1]")
        if type(hit.slot) is not int or not 0 <= hit.slot <= 127:
            raise ValueError(f"row {seq}: slot must be an int in [0, 127]")
        yield [hit.group, -hit.score, seq, hit.page, hit.slot]


def _spill_records(entries, root, name, *, run_rows, fan_in):
    """Externally sort JSON int records into ``root/name.sorted.jsonl``."""
    def part(level, index):
        return root / f"{name}-{level}-{index}.jsonl"

    iterator = iter(entries)
    count = 0
    while True:
        chunk = list(islice(iterator, run_rows))
        if not chunk:
            break
        chunk.sort()
        _write(part(0, count), chunk)
        count += 1
    level = 0
    while count > fan_in:
        written = 0
        for start in range(0, count, fan_in):
            group = [part(level, i) for i in range(start, min(start + fan_in, count))]
            with closing(_merge(group)) as merged:
                _write(part(level + 1, written), merged)
            for old in group:
                old.unlink()
            written += 1
        count, level = written, level + 1
    final = root / f"{name}.sorted.jsonl"
    if count == 0:
        final.write_bytes(b"")
    else:
        with closing(_merge([part(level, i) for i in range(count)])) as merged:
            _write(final, merged)
        for i in range(count):
            part(level, i).unlink()
    return final


def _winner_entries(hits_path):
    """Fold the (group, -score, seq)-sorted hits into one winner per group."""
    previous = None
    with open(hits_path, encoding="ascii") as stream:
        for line in stream:
            group, neg_score, seq, page, slot = json.loads(line)
            if group != previous:
                previous = group
                yield [neg_score, seq, group, page, slot]


def _write_selected(winners_path, selected_path, limit):
    """Keep the first ``limit`` rank-ordered winners in ``selected_path``.

    Yields one ``[page, rank, slot]`` record per selected hit for the page
    index.
    """
    rank = 0
    with open(winners_path, encoding="ascii") as stream, \
            selected_path.open("w", encoding="ascii", newline="\n") as out:
        for line in stream:
            if rank >= limit:
                break
            neg_score, _seq, group, page, slot = json.loads(line)
            out.write(json.dumps([page, slot, group, -neg_score],
                                 separators=(",", ":")) + "\n")
            yield [page, rank, slot]
            rank += 1


def _plan_entries(bypage_path):
    """Aggregate the (page, rank, slot) stream into page-read plan records.

    Emits one ``[first_rank, 0, page, slots]`` record per page (the read
    request) and one ``[first_rank, 1, slot, rank]`` record per selected hit
    (the delivery assignments); the plan spill orders them by first use.
    """
    with open(bypage_path, encoding="ascii") as stream:
        page = None
        first_rank = 0
        slots = set()
        for line in stream:
            found_page, rank, slot = json.loads(line)
            if found_page != page:
                if page is not None:
                    yield [first_rank, 0, page, sorted(slots)]
                page, first_rank, slots = found_page, rank, set()
            slots.add(slot)
            yield [first_rank, 1, slot, rank]
        if page is not None:
            yield [first_rank, 0, page, sorted(slots)]


def _check_page_response(page, slots, response):
    if not isinstance(response, dict) or len(response) != len(slots):
        raise ValueError(
            f"page {page}: response must be a dict keyed by exactly "
            "the requested slots")
    wanted = set(slots)
    for key, value in response.items():
        if type(key) is not int or key not in wanted:
            raise ValueError(f"page {page}: unexpected slot {key!r} in response")
        if not isinstance(value, str):
            raise ValueError(f"page {page}: slot {key}: payload must be a str")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(
                f"page {page}: slot {key}: payload is not UTF-8 encodable") from None
        if len(encoded) > _RANK_MAX_PAYLOAD:
            raise ValueError(
                f"page {page}: slot {key}: payload exceeds "
                f"{_RANK_MAX_PAYLOAD} UTF-8 bytes")


def _ranked_rows(selected_path, plan_path, payloads_path, offsets_path, read_page):
    opened = []
    try:
        selected = open(selected_path, encoding="ascii")
        opened.append(selected)
        plan = open(plan_path, encoding="ascii")
        opened.append(plan)
        payloads = open(payloads_path, "w+b")
        opened.append(payloads)
        offsets = open(offsets_path, "w+b")
        opened.append(offsets)
        pending = plan.readline()
        for rank, line in enumerate(selected):
            page, slot, group, score = json.loads(line)
            if pending:
                record = json.loads(pending)
                if record[0] == rank:
                    # First use of this page: read it once, asking for every
                    # slot any selected hit will need, and validate the whole
                    # response before anything from it is staged or delivered.
                    slots = record[3]
                    response = read_page(record[2], tuple(slots))
                    _check_page_response(record[2], slots, response)
                    written = {}
                    while True:
                        pending = plan.readline()
                        if not pending:
                            break
                        record = json.loads(pending)
                        if record[0] != rank:
                            break
                        found_slot, found_rank = record[2], record[3]
                        location = written.get(found_slot)
                        if location is None:
                            data = response[found_slot].encode("utf-8")
                            payloads.seek(0, 2)
                            location = (payloads.tell(), len(data))
                            payloads.write(data)
                            written[found_slot] = location
                        offsets.seek(found_rank * 16)
                        offsets.write(location[0].to_bytes(8, "little"))
                        offsets.write(location[1].to_bytes(8, "little"))
                    del response, written
            offsets.seek(rank * 16)
            position = offsets.read(16)
            payloads.seek(int.from_bytes(position[:8], "little"))
            payload = payloads.read(int.from_bytes(position[8:], "little")).decode("utf-8")
            yield (Hit(group, score, page, slot), payload)
    finally:
        for stream in opened:
            stream.close()


@contextmanager
def late_ranked_rows(hits, read_page, *, limit, run_rows=128, fan_in=8,
                     scratch=None):
    """Yield a one-shot iterator of ``(hit, payload)`` for the grouped Top-K.

    The one-shot ``hits`` stream is consumed and validated when the context
    is entered; no page is read before the caller starts consuming.  Hits
    rank by descending score, ties by original arrival order; the best hit
    per group is elected and the first ``limit`` distinct groups are
    delivered in that order.  ``read_page(page, slots)`` is called at most
    once per page, only when the rank about to be delivered first needs it,
    with exactly the slots the selected hits use on that page.  The iterator
    is valid only within the with-block.
    """
    _check_rank_params(limit, run_rows, fan_in, scratch)
    with TemporaryDirectory(dir=scratch, prefix="row-rank-") as base:
        root = Path(base)
        hits_path = _spill_records(_hit_entries(hits), root, "hits",
                                   run_rows=run_rows, fan_in=fan_in)
        winners_path = _spill_records(_winner_entries(hits_path), root, "winners",
                                      run_rows=run_rows, fan_in=fan_in)
        hits_path.unlink()
        selected_path = root / "selected.jsonl"
        bypage_path = _spill_records(
            _write_selected(winners_path, selected_path, limit),
            root, "bypage", run_rows=run_rows, fan_in=fan_in)
        winners_path.unlink()
        plan_path = _spill_records(_plan_entries(bypage_path), root, "plan",
                                   run_rows=run_rows, fan_in=fan_in)
        bypage_path.unlink()
        cursor = _ranked_rows(selected_path, plan_path, root / "payloads.bin",
                              root / "offsets.bin", read_page)
        try:
            yield cursor
        finally:
            cursor.close()


@contextmanager
def compact_versions(selected, untouched, *, snapshots=(), run_rows=128,
                     fan_in=8, scratch=None):
    """Compact one replaceable run set while other run sets stay untouched.

    Yields a batch whose replacement cursor ships only the selected originals
    still needed to reproduce the registered historical read points and the
    latest view; point queries read the replacement spill together with the
    untouched spill, never the original selected input.
    """
    _check_version_params(snapshots, run_rows, fan_in, scratch)
    points = sorted(snapshots)
    with TemporaryDirectory(dir=scratch, prefix="version-compact-") as base:
        root = Path(base)
        selected_path = _spill_versions(
            selected, root, "selected", 0, run_rows=run_rows, fan_in=fan_in)
        untouched_path = _spill_versions(
            untouched, root, "untouched", 1, run_rows=run_rows, fan_in=fan_in)
        merged_path = root / "merged.jsonl"
        with closing(_version_merge([untouched_path, selected_path])) as merged:
            _write(merged_path, merged)
        replacement_path = root / "replacement.sorted.jsonl"
        _write_replacement(merged_path, replacement_path, points)
        merged_path.unlink()
        selected_path.unlink()
        batch = _VersionBatch(replacement_path, untouched_path, points, root)
        try:
            yield batch
        finally:
            batch._shutdown()
