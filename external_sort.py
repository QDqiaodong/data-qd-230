"""Disk-backed stable sort and budgeted full-join operators for keyed rows."""
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
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
