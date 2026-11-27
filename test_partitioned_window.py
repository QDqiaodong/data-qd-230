"""Acceptance tests for the partitioned look-ahead window operator."""
import bisect
import threading
import time
import tracemalloc
import unittest
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from external_sort import Row, partitioned_window

MIN_KEY = -(1 << 63)
MAX_KEY = (1 << 63) - 1


def drain(batch, shard):
    items = []
    while True:
        item = batch.take(shard)
        if item is None:
            return items
        items.append(item)


def oracle(rows, preceding):
    """Reference window counts in arbitrary-precision integer arithmetic."""
    keys = sorted(row.key for row in rows)
    result = []
    for row in rows:
        low = row.key - preceding
        result.append(sum(1 for key in keys if low <= key <= row.key))
    return result


def expected_shards(rows, cuts, preceding):
    counts = oracle(rows, preceding)
    shards = [[] for _ in range(len(cuts) + 1)]
    for index, (row, count) in enumerate(zip(rows, counts)):
        shard = bisect.bisect_right(cuts, row.key)
        shards[shard].append((index, row, count))
    for items in shards:
        items.sort(key=lambda item: (item[1].key, item[0]))
    return shards


def directory_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


class PartitionedWindowSpecTest(unittest.TestCase):
    def test_headline_retirement_sequence_with_carry_one_and_two(self):
        keys = [12, 8, 10, 9, 8, 20, 10, 19, 24, 21]
        rows = [Row(key, f"payload-{i}") for i, key in enumerate(keys)]
        for carry in (1, 2):
            with self.subTest(carry=carry):
                with TemporaryDirectory() as scratch:
                    with partitioned_window(
                            iter(rows), [10, 20], preceding=12,
                            carry_keys=carry, run_rows=3, fan_in=2,
                            scratch=scratch) as batch:
                        p0 = batch.owner_path(0)
                        p1 = batch.owner_path(1)
                        p2 = batch.owner_path(2)
                        for path in (p0, p1, p2):
                            self.assertTrue(path.is_dir())

                        # First row of shard 1 while shard 0 is untouched.
                        self.assertEqual(batch.take(1), (Row(10, "payload-2"), 5))
                        # Drain shard 0 completely.
                        self.assertEqual(batch.take(0), (Row(8, "payload-1"), 2))
                        self.assertEqual(batch.take(0), (Row(8, "payload-4"), 2))
                        self.assertEqual(batch.take(0), (Row(9, "payload-3"), 3))
                        self.assertIsNone(batch.take(0))

                        first = batch.retire(0)
                        if carry == 1:
                            # Keys 8 and 9 are both still needed: two slots.
                            self.assertFalse(first)
                            self.assertTrue(p0.is_dir())
                        else:
                            if first:
                                self.assertFalse(p0.exists())
                            else:
                                self.assertTrue(p0.is_dir())

                        # Shard 2 has never been opened; its count must be exact.
                        self.assertEqual(batch.take(2), (Row(20, "payload-5"), 8))
                        self.assertEqual(batch.take(1), (Row(10, "payload-6"), 5))
                        self.assertEqual(batch.take(1), (Row(12, "payload-0"), 6))
                        self.assertEqual(batch.take(1), (Row(19, "payload-7"), 7))
                        self.assertIsNone(batch.take(1))

                        # Only key 9 still owes a count (to the row of key 21).
                        self.assertTrue(batch.retire(0))
                        self.assertFalse(p0.exists())
                        self.assertEqual(batch.take(2), (Row(21, "payload-9"), 7))
                        # Key 24 is deliberately still un-taken; retry is a
                        # no-op success on the already retired shard.
                        self.assertTrue(batch.retire(0))
                        self.assertEqual(batch.take(2), (Row(24, "payload-8"), 5))
                        self.assertIsNone(batch.take(2))
                        self.assertTrue(batch.retire(1))
                        self.assertFalse(p1.exists())
                        self.assertTrue(batch.retire(2))
                        self.assertFalse(p2.exists())
                    self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_two_finished_sources_contend_for_one_reserve_budget(self):
        rows = [Row(key, f"p{i}") for i, key in enumerate(
            [8, 9, 10, 11, 20, 21])]
        with TemporaryDirectory() as scratch:
            with partitioned_window(
                    iter(rows), [10, 20], preceding=12, carry_keys=2,
                    scratch=scratch) as batch:
                p0 = batch.owner_path(0)
                p1 = batch.owner_path(1)
                p2 = batch.owner_path(2)
                self.assertEqual(drain(batch, 0),
                                 [(Row(8, "p0"), 1), (Row(9, "p1"), 2)])
                self.assertEqual(drain(batch, 1),
                                 [(Row(10, "p2"), 3), (Row(11, "p3"), 4)])
                r0 = batch.retire(0)
                r1 = batch.retire(1)
                # Key 20 needs both source keys of each side; at most one
                # source can fit its two slots into the shared budget of two.
                self.assertLessEqual([r0, r1].count(True), 1)
                if r0:
                    self.assertFalse(p0.exists())
                    self.assertFalse(batch.retire(1))
                    self.assertTrue(p1.is_dir())
                else:
                    self.assertTrue(p0.is_dir())
                    self.assertTrue(r1)
                    self.assertFalse(p1.exists())
                    self.assertFalse(batch.retire(0))

                self.assertEqual(drain(batch, 2),
                                 [(Row(20, "p4"), 5), (Row(21, "p5"), 5)])
                self.assertTrue(batch.retire(0))
                self.assertTrue(batch.retire(1))
                self.assertTrue(batch.retire(2))
                for path in (p0, p1, p2):
                    self.assertFalse(path.exists())
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_zero_width_window_counts_only_same_key_rows(self):
        rows = [Row(5, "a"), Row(5, "b"), Row(4, "c"),
                Row(6, "d"), Row(5, "e")]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [5], preceding=0,
                                    carry_keys=0, scratch=scratch) as batch:
                # Rows of key 5 go to the right-hand shard.
                self.assertEqual(drain(batch, 0), [(Row(4, "c"), 1)])
                self.assertEqual(drain(batch, 1), [
                    (Row(5, "a"), 3), (Row(5, "b"), 3),
                    (Row(5, "e"), 3), (Row(6, "d"), 1)])

    def test_empty_shards_and_empty_input(self):
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(()), [10, 20], preceding=7,
                                    carry_keys=0, scratch=scratch) as batch:
                for shard in range(3):
                    self.assertIsNone(batch.take(shard))
                    self.assertTrue(batch.retire(shard))
                    self.assertFalse(batch.owner_path(shard).exists())
                    self.assertTrue(batch.retire(shard))
            self.assertEqual(list(Path(scratch).iterdir()), [])

        rows = [Row(0, "x"), Row(30, "y")]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [10, 20], preceding=100,
                                    carry_keys=0, scratch=scratch) as batch:
                self.assertEqual(drain(batch, 0), [(Row(0, "x"), 1)])
                self.assertEqual(drain(batch, 1), [])
                self.assertEqual(drain(batch, 2), [(Row(30, "y"), 2)])
                # The empty range retires for free even with zero budget.
                self.assertTrue(batch.retire(1))
                self.assertFalse(batch.owner_path(1).exists())

    def test_extreme_keys_use_mathematical_integer_bounds(self):
        rows = [Row(MAX_KEY, "a"), Row(MIN_KEY, "b"), Row(MIN_KEY, "c"),
                Row(MAX_KEY - 1, "d"), Row(0, "e")]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [0], preceding=MAX_KEY,
                                    carry_keys=0, scratch=scratch) as batch:
                got = drain(batch, 0) + drain(batch, 1)
                expected = sorted(
                    ((row, count) for row, count in zip(rows, oracle(rows, MAX_KEY))),
                    key=lambda item: item[0].key)
                self.assertEqual(got, expected)
        # Zero-width window at the minimum key must not wrap around.
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [], preceding=0,
                                    carry_keys=0, scratch=scratch) as batch:
                got = drain(batch, 0)
                self.assertEqual([count for _row, count in got],
                                 [2, 2, 1, 1, 1])

    def test_cut_belong_to_right_shard_and_inputs_are_not_mutated(self):
        cuts = [10, 20]
        cuts_copy = list(cuts)
        rows = [Row(20, "a"), Row(10, "b"), Row(9, "c"), Row(21, "d")]
        rows_copy = list(rows)
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), cuts, preceding=0,
                                    carry_keys=0, scratch=scratch) as batch:
                self.assertEqual(drain(batch, 0), [(Row(9, "c"), 1)])
                self.assertEqual(drain(batch, 1), [(Row(10, "b"), 1)])
                self.assertEqual(drain(batch, 2),
                                 [(Row(20, "a"), 1), (Row(21, "d"), 1)])
        self.assertEqual(cuts, cuts_copy)
        self.assertEqual(rows, rows_copy)

    def test_owner_path_is_stable_exclusive_and_carries_payloads_only_there(self):
        rows = [Row(key, f"payload-{key}") for key in range(9)]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [3, 6], preceding=2,
                                    carry_keys=0, scratch=scratch) as batch:
                paths = [batch.owner_path(s) for s in range(3)]
                # Fixed for the life of the batch and mutually distinct.
                self.assertEqual(paths, [batch.owner_path(s) for s in range(3)])
                self.assertEqual(len(set(paths)), 3)
                for path in paths:
                    self.assertTrue(path.is_dir())
                # Every payload lives in exactly one shard directory.
                for shard, path in enumerate(paths):
                    text = "".join(
                        p.read_text(errors="ignore")
                        for p in path.rglob("*") if p.is_file())
                    for key in range(9):
                        present = f"payload-{key}" in text
                        expected_shard = bisect.bisect_right([3, 6], key)
                        self.assertEqual(present, shard == expected_shard)
                # The integer-only bookkeeping database sits outside shard
                # directories and stores no payload bytes.
                db = next(Path(scratch).rglob("window.sqlite3"))
                for path in paths:
                    self.assertNotIn(db, path.parents)
                blob = db.read_bytes()
                self.assertNotIn(b"payload-", blob)

    def test_no_full_input_substitute_after_entry_completes(self):
        wide = "z" * 2048
        rows = [Row(key, wide if key % 2 == 0 else f"p{key}")
                for key in range(60)]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [20, 40], preceding=5,
                                    carry_keys=10, run_rows=7, fan_in=2,
                                    scratch=scratch) as batch:
                # Only per-shard spills plus the integer count database exist;
                # no file outside the shard dirs holds payload text.
                for path in Path(scratch).rglob("*"):
                    if not path.is_file():
                        continue
                    inside_shard = any(
                        batch.owner_path(s) in path.parents for s in range(3))
                    if not inside_shard:
                        self.assertNotIn(b"z" * 2048, path.read_bytes())
                # Retiring one shard removes the only on-disk copy of its
                # payloads; the remaining shards keep theirs intact.
                drain(batch, 0)
                self.assertTrue(batch.retire(0))
                self.assertFalse(batch.owner_path(0).exists())
                remaining = [
                    row for shard in (1, 2) for row, _ in drain(batch, shard)]
                self.assertEqual([row.key for row in remaining],
                                 list(range(20, 60)))


class PartitionedWindowLifecycleTest(unittest.TestCase):
    def test_retire_with_unread_rows_fails_and_keeps_source_usable(self):
        rows = [Row(key, f"p{key}") for key in range(6)]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [3], preceding=2,
                                    carry_keys=10, scratch=scratch) as batch:
                path = batch.owner_path(0)
                self.assertFalse(batch.retire(0))
                self.assertTrue(path.is_dir())
                # A failed retire changes no consumption position.
                self.assertEqual(batch.take(0), (Row(0, "p0"), 1))
                self.assertFalse(batch.retire(0))
                self.assertTrue(path.is_dir())
                self.assertEqual(drain(batch, 0),
                                 [(Row(1, "p1"), 2), (Row(2, "p2"), 3)])
                # Shard 1 is still fully un-taken: its first key 3 reaches back
                # to key 1 (window [1, 3]), so shard 0 still owes two distinct
                # keys.  With no carry budget it must stay on disk; with room
                # it may migrate the two counts and release the directory.
                result = batch.retire(0)
                if result:
                    self.assertFalse(path.exists())
                    held = batch._conn.execute(
                        "SELECT COUNT(*) FROM reserve_counts").fetchone()[0]
                    self.assertEqual(held, 2)
                else:
                    self.assertTrue(path.is_dir())
                self.assertEqual(drain(batch, 1), [
                    (Row(3, "p3"), 3), (Row(4, "p4"), 3), (Row(5, "p5"), 3)])
                self.assertTrue(batch.retire(0))
                self.assertFalse(path.exists())

    def test_failed_retire_reserves_nothing_and_keeps_source_intact(self):
        rows = [Row(key, f"p{key}") for key in range(10)]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [5], preceding=100,
                                    carry_keys=1, scratch=scratch) as batch:
                drain(batch, 0)
                # Five distinct left keys are still needed but only one slot
                # exists: every attempt must fail without partial reservation.
                for _ in range(3):
                    self.assertFalse(batch.retire(0))
                    self.assertTrue(batch.owner_path(0).is_dir())
                    held = batch._conn.execute(
                        "SELECT COUNT(*) FROM reserve_counts").fetchone()[0]
                    self.assertEqual(held, 0)
                # The untouched source keeps every future count exact.
                for t in range(5, 10):
                    row, count = batch.take(1)
                    self.assertEqual((row.key, count), (t, t + 1))
                # With no consumer left, the shard retires for free.
                self.assertTrue(batch.retire(0))
                self.assertFalse(batch.owner_path(0).exists())

    def test_stale_retained_slots_are_released_for_later_retirements(self):
        rows = [Row(key, f"p{key}") for key in range(12)]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [4, 8], preceding=2,
                                    carry_keys=2, scratch=scratch) as batch:
                drain(batch, 0)
                # Right frontier is key 5; keys 3 and 4 are retained.
                self.assertTrue(batch.retire(0))
                self.assertEqual(batch._conn.execute(
                    "SELECT COUNT(*) FROM reserve_counts").fetchone()[0], 2)
                drain(batch, 1)
                # Frontier moves to key 9: stale keys 3, 4 fall out of every
                # remaining window, freeing room for keys 7, 8 of shard 1.
                self.assertTrue(batch.retire(1))
                held = batch._conn.execute(
                    "SELECT COUNT(*) FROM reserve_counts").fetchone()[0]
                self.assertLessEqual(held, 2)
                drain(batch, 2)
                self.assertTrue(batch.retire(2))

    def test_take_after_retire_and_calls_after_close_raise(self):
        rows = [Row(1, "p")]
        with TemporaryDirectory() as scratch:
            with partitioned_window(iter(rows), [0], preceding=1,
                                    carry_keys=0, scratch=scratch) as batch:
                self.assertEqual(drain(batch, 1), [(Row(1, "p"), 1)])
                self.assertTrue(batch.retire(1))
                with self.assertRaises(RuntimeError):
                    batch.take(1)
                for bad in (-1, 2, "1", 1.0, True):
                    with self.assertRaises(ValueError):
                        batch.take(bad)
                    with self.assertRaises(ValueError):
                        batch.retire(bad)
                    with self.assertRaises(ValueError):
                        batch.owner_path(bad)
            for method in (batch.take, batch.retire, batch.owner_path):
                with self.assertRaises(RuntimeError):
                    method(0)

    def test_invalid_parameters_are_rejected_before_consumption(self):
        bad = [
            {"cuts": [1, 1]}, {"cuts": [2, 1]}, {"cuts": [1.0]},
            {"cuts": [True]}, {"cuts": list(range(8))}, {"cuts": "ab"},
            {"cuts": [1 << 63]}, {"cuts": [-(1 << 63) - 1]},
            {"preceding": -1}, {"preceding": 1.0}, {"preceding": True},
            {"preceding": 1 << 63},
            {"carry_keys": -1}, {"carry_keys": 1.0},
            {"carry_keys": 200_001}, {"carry_keys": True},
            {"run_rows": 0}, {"run_rows": 257}, {"run_rows": True},
            {"fan_in": 1}, {"fan_in": 17}, {"fan_in": False},
            {"scratch": 7},
        ]
        for override in bad:
            with self.subTest(override=override):
                kwargs = {"preceding": 1, "carry_keys": 1}
                kwargs.update(override)
                cuts = kwargs.pop("cuts", [])
                with self.assertRaises(ValueError):
                    with partitioned_window(iter(()), cuts, **kwargs):
                        self.fail("invalid parameters accepted")

    def test_invalid_rows_and_row_limit_leave_no_scratch(self):
        cases = [
            [Row("x", "p")],
            [Row(1, 1)],
            [Row(1, "x" * 4097)],
            [Row(1 << 63, "p")],
            ["not-a-row"],
        ]
        with TemporaryDirectory() as scratch:
            for bad_rows in cases:
                with self.assertRaises(ValueError):
                    with partitioned_window(iter(bad_rows), [], preceding=1,
                                            carry_keys=0, scratch=scratch):
                        self.fail("invalid row accepted")
                self.assertEqual(list(Path(scratch).iterdir()), [])

        def too_many():
            for index in range(200_001):
                yield Row(0, "")

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "row 200000"):
                with partitioned_window(too_many(), [], preceding=0,
                                        carry_keys=0, run_rows=256,
                                        scratch=scratch):
                    self.fail("row limit not enforced")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_upstream_failure_and_early_exit_clean_only_own_scratch(self):
        def failing():
            yield Row(0, "ok")
            raise RuntimeError("upstream gone")

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(RuntimeError, "upstream gone"):
                with partitioned_window(failing(), [0], preceding=1,
                                        carry_keys=0, scratch=scratch):
                    self.fail("upstream failure entered a batch")
            self.assertEqual(list(Path(scratch).iterdir()), [])

        with TemporaryDirectory() as scratch:
            sentinel = Path(scratch) / "caller.txt"
            sentinel.write_text("keep me")
            with self.assertRaisesRegex(RuntimeError, "consumer exited"):
                with partitioned_window(
                        (Row(key, "p") for key in range(40)), [20],
                        preceding=3, carry_keys=0, scratch=scratch) as batch:
                    self.assertEqual(batch.take(0), (Row(0, "p"), 1))
                    raise RuntimeError("consumer exited")
            self.assertEqual(sentinel.read_text(), "keep me")
            self.assertEqual(list(Path(scratch).iterdir()), [sentinel])


class PartitionedWindowOracleTest(unittest.TestCase):
    def test_random_interleaving_with_retirement_matches_oracle(self):
        rng = Random(20261127)
        for trial in range(120):
            n = rng.randrange(0, 150)
            span = rng.choice([4, 16, 100])
            base = rng.choice([MIN_KEY, -50, 0, MAX_KEY - 200])
            rows = [Row(base + rng.randrange(span), f"p{i}") for i in range(n)]
            present = sorted({row.key for row in rows})
            cut_count = rng.randrange(0, 4)
            cuts = (sorted(rng.sample(present, min(cut_count, len(present))))
                    if present else [])
            preceding = rng.choice([0, 1, 3, 10, 10**18, MAX_KEY])
            carry = rng.choice([0, 1, 2, 3, 50, 200_000])
            expected = expected_shards(rows, cuts, preceding)
            positions = [0] * (len(cuts) + 1)
            retired = [False] * (len(cuts) + 1)

            with TemporaryDirectory() as scratch:
                with partitioned_window(
                        iter(rows), cuts, preceding=preceding,
                        carry_keys=carry, run_rows=rng.choice([1, 2, 7, 64]),
                        fan_in=rng.choice([2, 3, 8]),
                        scratch=scratch) as batch:
                    paths = [batch.owner_path(s)
                             for s in range(len(cuts) + 1)]
                    steps = 0
                    while any(pos < len(exp)
                              for pos, exp in zip(positions, expected)):
                        steps += 1
                        self.assertLess(steps, 8000)
                        shard = rng.randrange(len(cuts) + 1)
                        if positions[shard] == len(expected[shard]):
                            if retired[shard]:
                                continue
                            if rng.random() < 0.5:
                                self.assertIsNone(batch.take(shard))
                            else:
                                result = batch.retire(shard)
                                if result:
                                    self.assertFalse(paths[shard].exists())
                                    retired[shard] = True
                            continue
                        if rng.random() < 0.8:
                            if retired[shard]:
                                with self.assertRaises(RuntimeError):
                                    batch.take(shard)
                                continue
                            row, count = batch.take(shard)
                            _index, exp_row, exp_count = expected[shard][
                                positions[shard]]
                            positions[shard] += 1
                            self.assertEqual((row, count),
                                             (exp_row, exp_count))
                        else:
                            result = batch.retire(shard)
                            self.assertFalse(result)
                            self.assertTrue(paths[shard].is_dir())
                    for shard in range(len(cuts) + 1):
                        self.assertTrue(batch.retire(shard))
                        self.assertFalse(paths[shard].exists())
                        held = batch._conn.execute(
                            "SELECT COUNT(*) FROM reserve_counts").fetchone()[0]
                        self.assertLessEqual(held, carry)
                self.assertEqual(list(Path(scratch).iterdir()), [])


class PartitionedWindowResourceTest(unittest.TestCase):
    def test_wide_generator_input_interleaved_retire_within_budgets(self):
        wide_rows = 3000          # 1000 UTF-8 bytes each -> ~3 MiB payloads
        narrow_rows = 12_000
        total_rows = wide_rows + narrow_rows
        payload_bytes = wide_rows * 1000 + narrow_rows
        produced = [0]

        def generate():
            for key in range(6000):
                produced[0] += 1
                yield Row(key, "a")
            for key in range(6000, 6000 + wide_rows):
                produced[0] += 1
                yield Row(key, "w" * 1000)
            for key in range(9000, 9000 + 6000):
                produced[0] += 1
                yield Row(key, "b")

        def oracle_count(rows, key, preceding):
            low = key - preceding
            return sum(1 for row in rows if low <= row.key <= key)

        with TemporaryDirectory() as scratch:
            peak_temp = [0]
            stop = threading.Event()

            def watch():
                while not stop.is_set():
                    peak_temp[0] = max(peak_temp[0],
                                       directory_bytes(Path(scratch)))
                    time.sleep(0.002)

            watcher = threading.Thread(target=watch, daemon=True)
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            watcher.start()
            try:
                with partitioned_window(
                        generate(), [6000, 9000], preceding=9000,
                        carry_keys=10_000, run_rows=64, fan_in=4,
                        scratch=scratch) as batch:
                    build_peak = tracemalloc.get_traced_memory()[1] - baseline
                    self.assertEqual(produced[0], total_rows)
                    build_temp = directory_bytes(Path(scratch))
                    p1 = batch.owner_path(1)

                    # Consume the rightmost shard first, keeping no output list.
                    first = batch.take(2)
                    self.assertEqual(first, (Row(9000, "b"),
                                             oracle_count(
                                                 [Row(k, "") for k in
                                                  list(range(6000)) +
                                                  list(range(6000, 9000)) +
                                                  list(range(9000, 15000))],
                                                 9000, 9000)))
                    for _ in range(9):
                        row, count = batch.take(2)
                        self.assertEqual(row.key, 9001 + _)
                    # Real retirement attempt on an unread shard must fail.
                    self.assertFalse(batch.retire(1))
                    self.assertTrue(p1.is_dir())
                    mid_temp = directory_bytes(Path(scratch))

                    # Drain the wide shard and retire it; payloads really leave
                    # the scratch directory even though windows remain open.
                    wide_seen = 0
                    while True:
                        item = batch.take(1)
                        if item is None:
                            break
                        row, count = item
                        self.assertEqual(len(row.payload), 1000)
                        wide_seen += 1
                    self.assertEqual(wide_seen, wide_rows)
                    self.assertTrue(batch.retire(1))
                    self.assertFalse(p1.exists())

                    # Counts after the source directory is gone stay exact:
                    # verify against an independent count computed by hand for
                    # a sampled set of remaining keys, without storing outputs.
                    all_keys = (list(range(6000)) +
                                list(range(6000, 9000)) +
                                list(range(9000, 15000)))
                    checked = 0
                    while True:
                        item = batch.take(2)
                        if item is None:
                            break
                        row, count = item
                        if checked % 997 == 0:
                            low = row.key - 9000
                            expected = sum(1 for k in all_keys if low <= k <= row.key)
                            self.assertEqual(count, expected)
                        checked += 1
                    self.assertEqual(checked, 6000 - 10)
                    while batch.take(0) is not None:
                        pass
                    read_peak = tracemalloc.get_traced_memory()[1] - baseline
            finally:
                stop.set()
                watcher.join(timeout=2)
                traced_peak = tracemalloc.get_traced_memory()[1] - baseline
                tracemalloc.stop()

            byte_bound = (16 * 1024 * 1024
                          + 16 * (payload_bytes + 128 * total_rows))
            self.assertLessEqual(build_peak, 12 * 1024 * 1024)
            self.assertLessEqual(read_peak, 12 * 1024 * 1024)
            self.assertLessEqual(traced_peak, 12 * 1024 * 1024)
            self.assertLessEqual(peak_temp[0], byte_bound)
            self.assertLessEqual(build_temp, byte_bound)
            self.assertLessEqual(mid_temp, byte_bound)
            self.assertEqual(list(Path(scratch).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
