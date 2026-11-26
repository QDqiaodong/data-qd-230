from contextlib import nullcontext
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import tracemalloc
import unittest

from external_sort import Row, budgeted_full_join, sorted_rows


def _scrambled_four_key_inputs():
    """Ascending keys need 1, 6, 4 and 1 slots; first is left-only, last right-only."""
    left_rows = [Row(2, "L2a"), Row(1, "L1a"), Row(0, "L0"),
                 Row(1, "L1b"), Row(2, "L2b")]
    right_rows = [Row(3, "R3"), Row(2, "R2a"), Row(1, "R1a"),
                  Row(2, "R2b"), Row(1, "R1b"), Row(1, "R1c")]
    return left_rows, right_rows


def _pairs(rows):
    return [(a and (a.key, a.payload), b and (b.key, b.payload)) for a, b in rows]


def _rows(rows):
    return [(row.key, row.payload) for row in rows]


class BudgetedFullJoinTest(unittest.TestCase):
    def test_budget_admits_prefix_by_cost_and_defers_whole_keys(self):
        left_rows, right_rows = _scrambled_four_key_inputs()
        with TemporaryDirectory() as scratch:
            with budgeted_full_join(iter(left_rows), iter(right_rows), pair_budget=5,
                                    run_rows=2, fan_in=2, scratch=scratch) as cursors:
                joined, deferred_left, deferred_right = cursors
                self.assertEqual(_pairs(joined), [
                    ((0, "L0"), None),
                    ((2, "L2a"), (2, "R2a")), ((2, "L2a"), (2, "R2b")),
                    ((2, "L2b"), (2, "R2a")), ((2, "L2b"), (2, "R2b")),
                ])
                self.assertEqual(_rows(deferred_left), [(1, "L1a"), (1, "L1b")])
                self.assertEqual(_rows(deferred_right),
                                [(1, "R1a"), (1, "R1b"), (1, "R1c"), (3, "R3")])
            self.assertEqual(list(Path(scratch).iterdir()), [])

            with budgeted_full_join(iter(left_rows), iter(right_rows), pair_budget=8,
                                    run_rows=2, fan_in=2, scratch=scratch) as cursors:
                joined, deferred_left, deferred_right = cursors
                self.assertEqual(_pairs(joined), [
                    ((0, "L0"), None),
                    ((1, "L1a"), (1, "R1a")), ((1, "L1a"), (1, "R1b")),
                    ((1, "L1a"), (1, "R1c")), ((1, "L1b"), (1, "R1a")),
                    ((1, "L1b"), (1, "R1b")), ((1, "L1b"), (1, "R1c")),
                    (None, (3, "R3")),
                ])
                self.assertEqual(_rows(deferred_left), [(2, "L2a"), (2, "L2b")])
                self.assertEqual(_rows(deferred_right), [(2, "R2a"), (2, "R2b")])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_empty_inputs_and_zero_budget(self):
        with budgeted_full_join(iter(()), iter(()), pair_budget=99) as cursor:
            self.assertEqual([list(part) for part in cursor], [[], [], []])
        with TemporaryDirectory() as scratch:
            with budgeted_full_join(iter([Row(0, "L")]), iter(()), pair_budget=0,
                                    scratch=scratch) as (joined, deferred_left, deferred_right):
                self.assertEqual(list(joined), [])
                self.assertEqual(_rows(deferred_left), [(0, "L")])
                self.assertEqual(list(deferred_right), [])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_deferred_streams_feed_another_batch(self):
        left_rows, right_rows = _scrambled_four_key_inputs()
        with TemporaryDirectory() as scratch:
            with TemporaryDirectory() as scratch2:
                with budgeted_full_join(iter(left_rows), iter(right_rows), pair_budget=5,
                                        run_rows=2, fan_in=2, scratch=scratch) as cursors:
                    joined, deferred_left, deferred_right = cursors
                    # Interleave cursors: joined, one row from each deferred stream.
                    self.assertEqual(next(joined), (Row(0, "L0"), None))
                    self.assertEqual(next(deferred_left), Row(1, "L1a"))
                    self.assertEqual(next(deferred_right), Row(1, "R1a"))
                    self.assertEqual(_roots(list(joined)), [2, 2, 2, 2])
                    # The remaining deferred keys form the input of a second batch;
                    # only the keys not admitted by the first batch show up.
                    with budgeted_full_join(deferred_left, deferred_right, pair_budget=3,
                                            run_rows=2, fan_in=2, scratch=scratch2) as (
                                            joined2, left_deferred2, right_deferred2):
                        self.assertEqual(_pairs(joined2), [
                            ((1, "L1b"), (1, "R1b")), ((1, "L1b"), (1, "R1c")),
                            (None, (3, "R3")),
                        ])
                        self.assertEqual(list(left_deferred2), [])
                        self.assertEqual(list(right_deferred2), [])
                self.assertEqual(list(Path(scratch).iterdir()), [])
                self.assertEqual(list(Path(scratch2).iterdir()), [])

    def test_only_one_cursor_used_and_cursors_stop_after_exit(self):
        left_rows, right_rows = _scrambled_four_key_inputs()
        with TemporaryDirectory() as scratch:
            with budgeted_full_join(iter(left_rows), iter(right_rows), pair_budget=5,
                                    run_rows=2, fan_in=2, scratch=scratch) as cursors:
                joined, deferred_left, deferred_right = cursors
                self.assertEqual(next(deferred_left), Row(1, "L1a"))
            self.assertEqual(list(joined), [])
            self.assertEqual(list(deferred_left), [])
            self.assertEqual(list(deferred_right), [])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_early_stop_and_consumer_failure_release_scratch(self):
        left_rows, right_rows = _scrambled_four_key_inputs()
        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(RuntimeError, "consumer exited"):
                with budgeted_full_join(iter(left_rows), iter(right_rows), pair_budget=5,
                                        run_rows=2, fan_in=2, scratch=scratch) as (joined, _, _):
                    self.assertEqual(next(joined), (Row(0, "L0"), None))
                    raise RuntimeError("consumer exited")
            self.assertEqual(list(Path(scratch).iterdir()), [])
        with TemporaryDirectory() as scratch:
            with budgeted_full_join(iter(left_rows), iter(right_rows), pair_budget=5,
                                    run_rows=2, fan_in=2, scratch=scratch) as (joined, dl, dr):
                self.assertEqual(next(joined), (Row(0, "L0"), None))
            self.assertEqual(list(joined), [])
            self.assertEqual(list(dl), [])
            self.assertEqual(list(dr), [])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_invalid_last_row_and_upstream_failure_leave_no_spills(self):
        with TemporaryDirectory() as scratch:
            def bad_rows():
                yield Row(0, "ok")
                yield Row(1, "ok")
                yield Row("not-an-int", "bad")

            with self.assertRaisesRegex(ValueError, "row 2"):
                with budgeted_full_join(bad_rows(), iter(()), pair_budget=10,
                                        run_rows=1, scratch=scratch):
                    self.fail("invalid row entered output")
            self.assertEqual(list(Path(scratch).iterdir()), [])

        with TemporaryDirectory() as scratch:
            def failing_rows():
                yield Row(0, "ok")
                raise RuntimeError("upstream failed")

            with self.assertRaisesRegex(RuntimeError, "upstream failed"):
                with budgeted_full_join(iter(()), failing_rows(), pair_budget=10,
                                        run_rows=1, scratch=scratch):
                    self.fail("upstream failure entered output")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_invalid_parameters_and_row_limits(self):
        for overrides in ({"pair_budget": True}, {"pair_budget": 1.0},
                          {"pair_budget": -1}, {"pair_budget": 10**9 + 1},
                          {"run_rows": 0}, {"run_rows": 257}, {"run_rows": True},
                          {"fan_in": 1}, {"fan_in": 17}, {"fan_in": False},
                          {"max_payload": 0}, {"max_payload": 4097},
                          {"max_payload": 1.5}):
            kwargs = {"pair_budget": 5, **overrides}
            with self.assertRaises(ValueError):
                with budgeted_full_join(iter(()), iter(()), **kwargs):
                    self.fail("invalid parameters accepted")

        def too_many_rows():
            for i in range(200_001):
                yield Row(0, "")

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "row 200000"):
                with budgeted_full_join(too_many_rows(), iter(()), pair_budget=10**9,
                                        run_rows=256, fan_in=8, max_payload=1,
                                        scratch=scratch):
                    self.fail("oversized side accepted")
            self.assertEqual(list(Path(scratch).iterdir()), [])

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                with budgeted_full_join(iter([Row(0, "苹果")]), iter(()),
                                        pair_budget=1, max_payload=4, scratch=scratch):
                    self.fail("oversized payload accepted")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_matches_oracle_over_budgets_and_interleavings(self):
        rng = Random(991)
        for _ in range(12):
            left_rows = [Row(rng.randrange(-2, 4), f"L{i}") for i in range(rng.randrange(0, 9))]
            right_rows = [Row(rng.randrange(-2, 4), f"R{i}") for i in range(rng.randrange(0, 9))]
            for budget in (0, 1, 2, 3, 6, 9, 30):
                with TemporaryDirectory() as scratch:
                    with budgeted_full_join(iter(left_rows), iter(right_rows),
                                            pair_budget=budget, run_rows=2, fan_in=2,
                                            scratch=scratch) as cursors:
                        # Drain the three cursors in an arbitrary order, one row at
                        # a time, to prove no cursor depends on another's progress.
                        collected = [[], [], []]
                        active = [0, 1, 2]
                        while active:
                            index = rng.choice(active)
                            try:
                                collected[index].append(next(cursors[index]))
                            except StopIteration:
                                active.remove(index)
                    self.assertEqual(list(Path(scratch).iterdir()), [])
                self.assertEqual(_pairs(collected[0]), _oracle(left_rows, right_rows, budget)[0])
                self.assertEqual(_rows(collected[1]), _oracle(left_rows, right_rows, budget)[1])
                self.assertEqual(_rows(collected[2]), _oracle(left_rows, right_rows, budget)[2])


def _roots(rows):
    return [row[0].key if row[0] is not None else row[1].key for row in rows]


def _oracle(left_rows, right_rows, budget):
    left_groups, right_groups = {}, {}
    for row in left_rows:
        left_groups.setdefault(row.key, []).append(row)
    for row in right_rows:
        right_groups.setdefault(row.key, []).append(row)
    joined, deferred_left, deferred_right = [], [], []
    remaining = budget
    for key in sorted(set(left_groups) | set(right_groups)):
        lrows = left_groups.get(key, [])
        rrows = right_groups.get(key, [])
        required = len(lrows) * len(rrows) if lrows and rrows else len(lrows) + len(rrows)
        if required <= remaining:
            remaining -= required
            if lrows and rrows:
                for lrow in lrows:
                    for rrow in rrows:
                        joined.append((lrow, rrow))
            else:
                for row in lrows or rrows:
                    joined.append((row, None) if lrows else (None, row))
        else:
            deferred_left.extend(lrows)
            deferred_right.extend(rrows)
    return ([(a and (a.key, a.payload), b and (b.key, b.payload)) for a, b in joined],
            [(row.key, row.payload) for row in deferred_left],
            [(row.key, row.payload) for row in deferred_right])


class BudgetedFullJoinResourceTest(unittest.TestCase):
    def test_hot_key_larger_than_runs_streams_within_budget(self):
        rows_per_side = 3000
        payload_bytes = [0]
        produced = [0]

        def side(prefix):
            for i in range(rows_per_side):
                produced[0] += 1
                row = Row(777, prefix + str(i))
                payload_bytes[0] += len(row.payload.encode("utf-8"))
                yield row

        with TemporaryDirectory() as scratch:
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            try:
                with budgeted_full_join(side("L"), side("R"),
                                        pair_budget=rows_per_side * rows_per_side,
                                        run_rows=32, fan_in=4, scratch=scratch) as cursors:
                    joined, deferred_left, deferred_right = cursors
                    head = [next(joined) for _ in range(5)]
                    self.assertEqual([(a.payload, b.payload) for a, b in head], [
                        ("L0", "R0"), ("L0", "R1"), ("L0", "R2"), ("L0", "R3"), ("L0", "R4"),
                    ])
                    live = sum(p.stat().st_size for p in Path(scratch).rglob("*") if p.is_file())
                    peak = tracemalloc.get_traced_memory()[1]
                    self.assertEqual(produced[0], 2 * rows_per_side)
                    bound = 16 * 1024 * 1024 + 16 * (payload_bytes[0] + 128 * 2 * rows_per_side)
                    self.assertLessEqual(live, bound)
                    self.assertLessEqual(peak - baseline, 32 * 1024 * 1024)
                    self.assertEqual(list(deferred_left), [])
                    self.assertEqual(list(deferred_right), [])
            finally:
                tracemalloc.stop()
            self.assertEqual(list(Path(scratch).iterdir()), [])


class ExternalSortTest(unittest.TestCase):
    def test_multipass_matches_stable_oracle(self):
        rng = Random(737)
        rows = [Row(rng.randrange(-4, 5), f"input\n{i}→") for i in range(83)]
        with TemporaryDirectory() as scratch:
            with sorted_rows(iter(rows), run_rows=3, fan_in=2, scratch=scratch) as out:
                run_dir, = Path(scratch).iterdir()
                files = list(run_dir.iterdir())
                self.assertLessEqual(len(files), 2)
                self.assertTrue(all(int(f.name.split("-")[0]) >= 2 for f in files))
                self.assertEqual(list(out), sorted(rows, key=lambda row: row.key))
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_closes_on_early_stop_and_consumer_failure(self):
        with TemporaryDirectory() as scratch:
            for fail in (False, True):
                expectation = self.assertRaisesRegex(RuntimeError, "consumer exited") if fail else nullcontext()
                with expectation:
                    with sorted_rows((Row(key, "") for key in range(9, -1, -1)),
                                     run_rows=2, fan_in=2, scratch=scratch) as out:
                        self.assertEqual(next(out), Row(0, ""))
                        if fail:
                            raise RuntimeError("consumer exited")
                self.assertEqual(list(out), [])
                self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_invalid_late_row_leaves_no_spills(self):
        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "row 2"):
                with sorted_rows([Row(0, "ok"), Row(1, "ok"), Row(2, "苹果")],
                                 run_rows=1, max_payload=4, scratch=scratch):
                    self.fail("invalid row entered output")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_empty_and_extreme_integer_keys(self):
        with sorted_rows(iter(())) as out:
            self.assertEqual(list(out), [])
        rows = [Row((1 << 63) - 1, "max"), Row(-(1 << 63), "min")]
        with sorted_rows(iter(rows), run_rows=1) as out:
            self.assertEqual(list(out), rows[::-1])
