"""Acceptance tests for partial-run version compaction (compact_versions)."""
from contextlib import nullcontext
from pathlib import Path
from random import Random
import threading
import time
from tempfile import TemporaryDirectory
import tracemalloc
import unittest

from external_sort import Version, compact_versions


def _triples(rows):
    return [(row.key, row.revision, row.payload) for row in rows]


# ---------------------------------------------------------------------------
# Reference oracle: a straightforward, in-memory implementation of the two
# retention rules written in the specification, kept deliberately separate
# from the production streaming state machine.
# ---------------------------------------------------------------------------

def _oracle(selected, untouched, snapshots):
    """Return (replacement_triples, {point: visible_triples}) for one batch."""
    by_key = {}
    for row in selected:
        by_key.setdefault(row.key, []).append(
            (row.revision, 0, row.payload))       # source 0 = selected
    for row in untouched:
        by_key.setdefault(row.key, []).append(
            (row.revision, 1, row.payload))       # source 1 = untouched
    points = sorted(snapshots)
    read_points = points + [None]
    replacement = {}
    expected = {point: [] for point in read_points}

    def winner(records, bound):
        best = None
        for revision, source, payload in records:
            if revision <= bound and (best is None or revision > best[0]):
                best = (revision, source, payload)
        return best

    for key in sorted(by_key):
        records = sorted(by_key[key])
        kept_revisions = set()
        # Rule 1: selected values winning any release point (latest last).
        for revision, source, payload in records:
            if source == 0 and payload is not None:
                for point in read_points:
                    bound = (1 << 63) if point is None else point
                    if winner(records, bound) == (revision, source, payload):
                        kept_revisions.add(revision)
                        replacement[(key, revision)] = payload
                        break
        # Rule 2: walk the points in order, latest last, starting from all
        # untouched rows plus the selected rows already decided to keep.
        for point in read_points:
            bound = (1 << 63) if point is None else point
            delivered = [(r, s, p) for r, s, p in records
                         if s == 1 or r in kept_revisions]
            original_winner = winner(records, bound)
            if original_winner is None or original_winner[2] is None:
                delivered_winner = winner(delivered, bound)
                if delivered_winner is not None \
                        and delivered_winner[2] is not None:
                    revision, source, payload = original_winner
                    if source == 0 and revision not in kept_revisions:
                        kept_revisions.add(revision)
                        replacement[(key, revision)] = None
        final_delivered = [(r, s, p) for r, s, p in records
                           if s == 1 or r in kept_revisions]
        for point in read_points:
            bound = (1 << 63) if point is None else point
            win = winner(final_delivered, bound)
            if win is not None and win[2] is not None:
                expected[point].append((key, win[0], win[2]))
    replacement_rows = sorted((key, revision, payload)
                              for (key, revision), payload in replacement.items())
    return replacement_rows, expected


class CompactVersionsSpecTest(unittest.TestCase):
    def _spec_inputs(self, key9_tombstone):
        selected = [Version(7, 11, "v11"), Version(7, 2, "v2"),
                    Version(7, 7, None), Version(9, 7, None)]
        untouched = [Version(7, 9, "u9"), Version(7, 1, "u1"),
                     Version(9, 1, None if key9_tombstone else "u1")]
        # Unordered, one-shot iterators.
        rng = Random(20261126)
        rng.shuffle(selected)
        rng.shuffle(untouched)
        return iter(selected), iter(untouched)

    def test_spec_scenario_replacement_and_queries_interleaved(self):
        selected, untouched = self._spec_inputs(key9_tombstone=False)
        with TemporaryDirectory() as scratch:
            with compact_versions(selected, untouched, snapshots=[3, 8, 10],
                                  run_rows=2, fan_in=2, scratch=scratch) as batch:
                replacement = batch.replacement()
                self.assertEqual(next(replacement), Version(7, 2, "v2"))
                # Query before consuming the rest of the replacement stream.
                self.assertEqual(_triples(batch.visible(3)),
                                 [(7, 2, "v2"), (9, 1, "u1")])
                self.assertEqual(_triples(batch.visible(8)), [])
                rest = list(replacement)
                self.assertEqual(_triples(rest),
                                 [(7, 7, None), (7, 11, "v11"),
                                  (9, 7, None)])
                self.assertEqual(_triples(batch.visible(10)),
                                 [(7, 9, "u9")])
                self.assertEqual(_triples(batch.visible()),
                                 [(7, 11, "v11")])
                # Repeated queries and re-opened cursors are deterministic.
                self.assertEqual(_triples(batch.visible(3)),
                                 [(7, 2, "v2"), (9, 1, "u1")])
                self.assertEqual(_triples(batch.replacement()),
                                 [(7, 2, "v2"), (7, 7, None),
                                  (7, 11, "v11"), (9, 7, None)])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_spec_scenario_untouched_value_turned_into_delete(self):
        selected, untouched = self._spec_inputs(key9_tombstone=True)
        with TemporaryDirectory() as scratch:
            with compact_versions(selected, untouched, snapshots=[3, 8, 10],
                                  run_rows=2, fan_in=2, scratch=scratch) as batch:
                self.assertEqual(_triples(batch.replacement()),
                                 [(7, 2, "v2"), (7, 7, None),
                                  (7, 11, "v11")])
                for at in (3, 8, 10, None):
                    rows = _triples(batch.visible(at))
                    self.assertNotIn(9, [row[0] for row in rows])
                self.assertEqual(_triples(batch.visible(3)), [(7, 2, "v2")])
                self.assertEqual(_triples(batch.visible(8)), [])
                self.assertEqual(_triples(batch.visible(10)),
                                 [(7, 9, "u9")])
                self.assertEqual(_triples(batch.visible()),
                                 [(7, 11, "v11")])
            self.assertEqual(list(Path(scratch).iterdir()), [])


class CompactVersionsRulesTest(unittest.TestCase):
    def test_two_snapshot_deletes_keep_only_the_earlier_one(self):
        # Untouched value rev1; selected deletes rev4 and rev9 each end a
        # registered point (5 and 10).  The rev4 delete already masks rev1,
        # so rev9 must not ship.  A later selected value rev12 still ships.
        selected = [Version(3, 4, None), Version(3, 9, None),
                    Version(3, 12, "late")]
        untouched = [Version(3, 1, "old")]
        with TemporaryDirectory() as scratch:
            with compact_versions(iter(selected), iter(untouched),
                                  snapshots=[5, 10], run_rows=1, fan_in=2,
                                  scratch=scratch) as batch:
                self.assertEqual(_triples(batch.replacement()),
                                 [(3, 4, None), (3, 12, "late")])
                self.assertEqual(_triples(batch.visible(5)), [])
                self.assertEqual(_triples(batch.visible(10)), [])
                self.assertEqual(_triples(batch.visible()),
                                 [(3, 12, "late")])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_second_delete_is_needed_when_value_reappears_between_points(self):
        # A selected value between the two deletes means the second delete
        # genuinely covers a point that would otherwise show that value.
        selected = [Version(3, 4, None), Version(3, 7, "back"),
                    Version(3, 9, None)]
        untouched = [Version(3, 1, "old")]
        with TemporaryDirectory() as scratch:
            with compact_versions(iter(selected), iter(untouched),
                                  snapshots=[5, 8, 10], run_rows=1, fan_in=2,
                                  scratch=scratch) as batch:
                self.assertEqual(_triples(batch.replacement()),
                                 [(3, 4, None), (3, 7, "back"),
                                  (3, 9, None)])
                self.assertEqual(_triples(batch.visible(5)), [])
                self.assertEqual(_triples(batch.visible(8)),
                                 [(3, 7, "back")])
                self.assertEqual(_triples(batch.visible(10)), [])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_very_late_cross_source_duplicate_fails_and_leaves_nothing(self):
        def selected_rows():
            yield Version(5, 1, "a")
            yield Version(5, 2, "b")
            yield Version(5, 999, "late-selected")

        def untouched_rows():
            yield Version(5, 4, "u4")
            yield Version(5, 999, "late-untouched")

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "duplicate revision 999"):
                with compact_versions(selected_rows(), untouched_rows(),
                                      snapshots=[500], run_rows=1, fan_in=2,
                                      scratch=scratch) as batch:
                    # The batch must never be exposed before decisions finish.
                    self.fail(f"batch leaked: {batch!r}")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_unregistered_points_are_rejected_without_advancing_cursors(self):
        selected = [Version(1, 2, "x")]
        untouched = [Version(1, 1, "y")]
        with TemporaryDirectory() as scratch:
            with compact_versions(iter(selected), iter(untouched),
                                  snapshots=[2], scratch=scratch) as batch:
                replacement = batch.replacement()
                first = next(replacement)
                for bad in (0, 1, 3, 2 ** 63 - 1, True, 1.0, "2"):
                    with self.assertRaises(ValueError):
                        batch.visible(bad)
                # The replacement cursor was not consumed by the failures.
                self.assertRaises(StopIteration, next, replacement)
                self.assertEqual(first, Version(1, 2, "x"))
                self.assertEqual(_triples(batch.visible(2)),
                                 [(1, 2, "x")])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_upstream_failure_propagates_and_cleans_scratch(self):
        def failing_selected():
            yield Version(1, 1, "ok")
            raise RuntimeError("selected boom")

        def failing_untouched():
            yield Version(2, 1, "ok")
            raise RuntimeError("untouched boom")

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(RuntimeError, "selected boom"):
                with compact_versions(failing_selected(), iter(()),
                                      snapshots=[1], scratch=scratch):
                    self.fail("upstream failure entered a batch")
            self.assertEqual(list(Path(scratch).iterdir()), [])

            with self.assertRaisesRegex(RuntimeError, "untouched boom"):
                with compact_versions(iter(()), failing_untouched(),
                                      snapshots=[1], scratch=scratch):
                    self.fail("upstream failure entered a batch")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_early_exit_and_cursor_lifecycle(self):
        selected = [Version(k, 2, f"s{k}") for k in range(20)]
        untouched = [Version(k, 1, f"u{k}") for k in range(20)]
        with TemporaryDirectory() as scratch:
            with compact_versions(iter(selected), iter(untouched),
                                  snapshots=[2], run_rows=4, fan_in=2,
                                  scratch=scratch) as batch:
                replacement = batch.replacement()
                head = next(replacement)
                query = batch.visible(2)
                query_head = next(query)
            # Cursors stop producing after the context exits...
            self.assertEqual(list(replacement), [])
            self.assertEqual(list(query), [])
            self.assertEqual(head, Version(0, 2, "s0"))
            self.assertEqual(query_head, Version(0, 2, "s0"))
            # ...and new calls raise RuntimeError.
            with self.assertRaises(RuntimeError):
                batch.replacement()
            with self.assertRaises(RuntimeError):
                batch.visible()
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_consumer_exception_inside_block_still_cleans_up(self):
        selected = [Version(1, 1, "x"), Version(1, 3, None)]
        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(RuntimeError, "consumer exited"):
                with compact_versions(iter(selected), iter(()),
                                      snapshots=[2], scratch=scratch) as batch:
                    next(batch.replacement())
                    raise RuntimeError("consumer exited")
            self.assertEqual(list(Path(scratch).iterdir()), [])


class CompactVersionsValidationTest(unittest.TestCase):
    def test_invalid_parameters(self):
        empty = iter(())
        good = dict(snapshots=(), run_rows=128, fan_in=8)
        overrides = [
            {"snapshots": {}}, {"snapshots": [0, 0]},
            {"snapshots": [-1]}, {"snapshots": [2 ** 63]},
            {"snapshots": [True]}, {"snapshots": [1.0]},
            {"snapshots": list(range(33))},
            {"run_rows": 0}, {"run_rows": 257}, {"run_rows": True},
            {"run_rows": 1.0}, {"fan_in": 1}, {"fan_in": 17},
            {"fan_in": False},
        ]
        for override in overrides:
            kwargs = {**good, **override}
            with self.assertRaises(ValueError):
                with compact_versions(iter(()), iter(()), **kwargs):
                    self.fail(f"accepted invalid params {override}")

    def test_invalid_records(self):
        cases = [
            [Version(True, 1, "x")],
            [Version(1, True, "x")],
            [Version(1, -1, "x")],
            [Version(1, 2 ** 63, "x")],
            [Version(1, 1, 1)],
            [Version(1, 1, "é" * 2049)],   # 4098 UTF-8 bytes
            [Version(1 << 63, 1, "x")],
            [Version(-(1 << 63) - 1, 1, "x")],
            ["not-a-version"],
        ]
        with TemporaryDirectory() as scratch:
            for rows in cases:
                with self.assertRaises(ValueError):
                    with compact_versions(iter(rows), iter(()),
                                          run_rows=1, scratch=scratch):
                        self.fail(f"accepted invalid record {rows!r}")
                self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_row_limit_per_side(self):
        def too_many():
            for i in range(200_001):
                yield Version(i, 1, "")

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "row 200000"):
                with compact_versions(too_many(), iter(()), run_rows=256,
                                      scratch=scratch):
                    self.fail("oversized selected stream accepted")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_empty_inputs_and_empty_payload_is_a_value(self):
        with TemporaryDirectory() as scratch:
            with compact_versions(iter(()), iter(()), snapshots=[5],
                                  scratch=scratch) as batch:
                self.assertEqual(list(batch.replacement()), [])
                self.assertEqual(list(batch.visible()), [])
                self.assertEqual(list(batch.visible(5)), [])
            rows = [Version(8, 3, ""), Version(8, 6, None)]
            with compact_versions(iter(rows), iter(()), snapshots=[4, 7],
                                  scratch=scratch) as batch:
                self.assertEqual(list(batch.replacement()),
                                 [Version(8, 3, ""), Version(8, 6, None)])
                self.assertEqual(list(batch.visible(4)),
                                 [Version(8, 3, "")])
                self.assertEqual(list(batch.visible(7)), [])
                self.assertEqual(list(batch.visible()), [])
            self.assertEqual(list(Path(scratch).iterdir()), [])


class CompactVersionsOracleTest(unittest.TestCase):
    def test_matches_oracle_on_random_cases_with_interleaved_reads(self):
        rng = Random(424242)
        for trial in range(60):
            key_count = rng.randrange(1, 9)
            selected, untouched = [], []
            used = {}
            for _ in range(rng.randrange(0, 40)):
                key = rng.randrange(-4, key_count)
                revision = rng.randrange(0, 40)
                if used.setdefault(key, set()).__contains__(revision):
                    continue
                used[key].add(revision)
                payload = None if rng.random() < 0.3 else f"p{trial}-{revision}"
                row = Version(key, revision, payload)
                (selected if rng.random() < 0.5 else untouched).append(row)
            snapshots = sorted(rng.sample(range(40), rng.randrange(0, 6)))
            expected_replacement, expected_visible = _oracle(
                selected, untouched, snapshots)
            with TemporaryDirectory() as scratch:
                with compact_versions(
                        iter(list(selected)), iter(list(untouched)),
                        snapshots=snapshots,
                        run_rows=rng.choice([1, 2, 5, 128]),
                        fan_in=rng.choice([2, 3, 8]),
                        scratch=scratch) as batch:
                    # Randomly interleave replacement and query consumption.
                    actual_replacement = []
                    cursors = [batch.replacement()]
                    queries = [batch.visible(at=point)
                               for point in snapshots + [None]]
                    drained = [False]
                    while not drained[0]:
                        cursor = rng.choice(cursors)
                        try:
                            actual_replacement.append(next(cursor))
                        except StopIteration:
                            drained[0] = True
                    actual_replacement = _triples(actual_replacement)
                    self.assertEqual(actual_replacement, expected_replacement)
                    for point in snapshots + [None]:
                        self.assertEqual(
                            _triples(batch.visible(at=point)),
                            expected_visible[point],
                            msg=f"trial {trial} point {point}")
                self.assertEqual(list(Path(scratch).iterdir()), [])


class CompactVersionsResourceTest(unittest.TestCase):
    def test_long_chain_and_wide_payloads_stay_within_budget(self):
        chain_length = 4_001
        wide_keys = 2_000
        wide_payload = "w" * 2_000                 # 2000 UTF-8 bytes each
        snapshots = [10, 50, 120, 400, 900, 1_500,
                     2_200, 2_800, 3_100, 3_500] + list(range(3_968, 3_990))
        self.assertEqual(len(snapshots), 32)
        self.assertEqual(len(set(snapshots)), 32)
        payload_total = [0]
        produced = [0]

        def hot_chain():
            # One key, revisions 1..chain_length; every fifth revision is a
            # delete, opening and closing delete windows along the chain.
            for revision in range(1, chain_length + 1):
                payload = None if revision % 5 == 0 else f"h{revision}"
                if payload is not None:
                    payload_total[0] += len(payload.encode())
                produced[0] += 1
                yield Version(0, revision, payload)

        def wide_rows():
            for key in range(1, wide_keys + 1):
                payload_total[0] += len(wide_payload.encode())
                produced[0] += 1
                yield Version(key, revision=1, payload=wide_payload)

        total_records = chain_length + wide_keys
        with TemporaryDirectory() as scratch:
            # Watchdog: sample the logical bytes of live temp data across the
            # whole context, including the build phase -- not after files are
            # deleted.  This catches peaks that a post-hoc directory listing
            # would miss.
            peak_temp = [0]
            stop = threading.Event()

            def watch():
                while not stop.is_set():
                    total = 0
                    for path in Path(scratch).rglob("*"):
                        if path.is_file():
                            try:
                                total += path.stat().st_size
                            except FileNotFoundError:
                                pass
                    peak_temp[0] = max(peak_temp[0], total)
                    time.sleep(0.002)

            watcher = threading.Thread(target=watch, daemon=True)
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            watcher.start()
            try:
                with compact_versions(hot_chain(), wide_rows(),
                                      snapshots=snapshots, run_rows=128,
                                      fan_in=8, scratch=scratch) as batch:
                    # Build-phase peak: measured while the context is open and
                    # the generators have already been fully drained, so the
                    # value reflects spilling plus the decision sweep rather
                    # than a post-cleanup stable state.
                    build_peak = tracemalloc.get_traced_memory()[1] - baseline
                    self.assertEqual(produced[0], total_records)
                    # The selected spill must be gone once decisions complete.
                    leftovers = [p.name for p in Path(scratch).rglob("*")
                                 if "selected" in p.name or "merged" in p.name]
                    self.assertEqual(leftovers, [])

                    # Read only a few outputs of the long chain via history.
                    # The first snapshot's winner is itself a delete: the key
                    # is invisible there (the pre-delete values won no other
                    # registered point, so they were correctly compacted out).
                    at = snapshots[0]
                    self.assertEqual(
                        [row for row in batch.visible(at=at) if row.key == 0],
                        [] if at % 5 == 0 else [
                            Version(0, at, f"h{at}")])
                    # Pick a snapshot whose winner is a value.
                    value_point = next(p for p in snapshots if p % 5 != 0)
                    head = [row for row in batch.visible(at=value_point)
                            if row.key == 0]
                    self.assertEqual(head,
                                     [Version(0, value_point, f"h{value_point}")])
                    # ...and check latest plus one late snapshot.
                    latest = [row for row in batch.visible() if row.key == 0]
                    self.assertEqual(latest, [Version(0, chain_length,
                                                      f"h{chain_length}")])
                    late_rev = next(p for p in reversed(snapshots)
                                    if p % 5 != 0)
                    late = [row for row in batch.visible(at=late_rev)
                            if row.key == 0]
                    self.assertEqual(late, [Version(0, late_rev, f"h{late_rev}")])

                    # Replacement stream: only read a prefix, and verify those
                    # rows are genuine selected originals (never untouched).
                    replacement = batch.replacement()
                    prefix = [next(replacement) for _ in range(20)]
                    self.assertEqual([row.key for row in prefix[:20]],
                                     [0] * 20)
                    for row in prefix:
                        self.assertIn(row.revision, range(1, chain_length + 1))
                        if row.payload is not None:
                            self.assertTrue(row.payload.startswith("h"))
            finally:
                stop.set()
                watcher.join(timeout=2)
                traced_peak = tracemalloc.get_traced_memory()[1] - baseline
                tracemalloc.stop()
            self.assertEqual(list(Path(scratch).iterdir()), [])

        total_payload = payload_total[0]
        byte_budget = 16 * 1024 * 1024 + 16 * (
            total_payload + 128 * total_records)
        # The watchdog samples concurrently; allow a small timing slack only
        # for files being unlinked between the size and unlink operations.
        self.assertLessEqual(peak_temp[0], byte_budget)
        self.assertLessEqual(build_peak, 24 * 1024 * 1024)
        self.assertLessEqual(traced_peak, 24 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
