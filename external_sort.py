"""Disk-backed stable sort operator for integer-keyed rows."""
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


def _entries(rows, max_payload):
    for seq, row in enumerate(rows):
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
