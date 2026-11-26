"""Acceptance tests for late-materialized grouped Top-K (late_ranked_rows)."""
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import tracemalloc
import unittest

from external_sort import Hit, late_ranked_rows


class _Pages:
    """read_page double: records every call, serves deterministic payloads."""

    def __init__(self, behavior=None):
        self.calls = []
        self._behavior = behavior

    def __call__(self, page, slots):
        self.calls.append((page, slots))
        if self._behavior is not None:
            return self._behavior(page, slots)
        return {slot: f"page{page}-slot{slot}" for slot in slots}


def _spec_hits(top_score):
    """Spec scenario: group 1 has two candidates, groups 2-4 one each.

    The lower-score groups arrive first; group 1's two candidates arrive
    last.  With ``top_score`` 100 the page-9 candidate wins its group; with
    95 the page-8 candidate (score 99) takes over.
    """
    return [
        Hit(2, 98, 4, 1),
        Hit(3, 97, 9, 6),
        Hit(4, 96, 7, 5),
        Hit(1, top_score, 9, 2),
        Hit(1, 99, 8, 3),
    ]


def _oracle(hits, limit):
    """In-memory reference: best hit per group, then the first ``limit``."""
    best = {}
    for seq, hit in enumerate(hits):
        key = (-hit.score, seq)
        if hit.group not in best or key < best[hit.group][0]:
            best[hit.group] = (key, hit)
    return [hit for _, hit in sorted(best.values(), key=lambda item: item[0])[:limit]]


class LateRankedReadScheduleTest(unittest.TestCase):
    def test_top_candidate_wins_and_pages_are_read_lazily(self):
        pages = _Pages()
        with TemporaryDirectory() as scratch:
            with late_ranked_rows(iter(_spec_hits(100)), pages, limit=3,
                                  run_rows=1, fan_in=2, scratch=scratch) as rows:
                # Nothing is read before consumption starts.
                self.assertEqual(pages.calls, [])
                first = next(rows)
                self.assertEqual(first, (Hit(1, 100, 9, 2), "page9-slot2"))
                # Page 9 is read once, for its own slot and the later rank's.
                self.assertEqual(pages.calls, [(9, (2, 6))])
                second = next(rows)
                self.assertEqual(second, (Hit(2, 98, 4, 1), "page4-slot1"))
                self.assertEqual(pages.calls, [(9, (2, 6)), (4, (1,))])
                third = next(rows)
                self.assertEqual(third, (Hit(3, 97, 9, 6), "page9-slot6"))
                # The third rank reuses the already-read page 9.
                self.assertEqual(pages.calls, [(9, (2, 6)), (4, (1,))])
                self.assertRaises(StopIteration, next, rows)
            # Page 7 (the group that lost to the limit) stayed unread.
            self.assertNotIn(7, [page for page, _ in pages.calls])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_dethroned_candidate_changes_the_read_schedule(self):
        pages = _Pages()
        with TemporaryDirectory() as scratch:
            with late_ranked_rows(iter(_spec_hits(95)), pages, limit=3,
                                  run_rows=1, fan_in=2, scratch=scratch) as rows:
                self.assertEqual(pages.calls, [])
                # Score 95 loses the group to the score-99 candidate on page 8.
                self.assertEqual(next(rows), (Hit(1, 99, 8, 3), "page8-slot3"))
                self.assertEqual(pages.calls, [(8, (3,))])
                self.assertEqual(next(rows), (Hit(2, 98, 4, 1), "page4-slot1"))
                self.assertEqual(pages.calls, [(8, (3,)), (4, (1,))])
                self.assertEqual(next(rows), (Hit(3, 97, 9, 6), "page9-slot6"))
                # Page 9 is now read late and only for the winner's slot.
                self.assertEqual(pages.calls, [(8, (3,)), (4, (1,)), (9, (6,))])
                self.assertRaises(StopIteration, next, rows)
            self.assertNotIn(7, [page for page, _ in pages.calls])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_tied_scores_keep_arrival_order_across_distant_inputs(self):
        hits = [
            Hit(1, 50, 11, 1),          # seq 0
            Hit(8, 90, 12, 2),
            Hit(9, 80, 13, 3),
            Hit(2, 50, 14, 4),          # seq 3, ties group 1
            Hit(1, 50, 15, 5),          # seq 4, same group+score: loses to seq 0
            Hit(3, 50, 16, 6),          # seq 5, ties groups 1 and 2
        ]
        pages = _Pages()
        with late_ranked_rows(iter(hits), pages, limit=5,
                              run_rows=1, fan_in=2) as rows:
            got = list(rows)
        self.assertEqual([hit for hit, _ in got], [
            Hit(8, 90, 12, 2), Hit(9, 80, 13, 3),
            Hit(1, 50, 11, 1), Hit(2, 50, 14, 4), Hit(3, 50, 16, 6),
        ])
        self.assertEqual([payload for _, payload in got], [
            "page12-slot2", "page13-slot3",
            "page11-slot1", "page14-slot4", "page16-slot6",
        ])

    def test_stopping_after_the_first_row_leaves_later_pages_unread(self):
        pages = _Pages()
        with TemporaryDirectory() as scratch:
            with late_ranked_rows(iter(_spec_hits(100)), pages, limit=3,
                                  run_rows=1, fan_in=2, scratch=scratch) as rows:
                self.assertEqual(next(rows), (Hit(1, 100, 9, 2), "page9-slot2"))
                self.assertEqual(pages.calls, [(9, (2, 6))])
            # Pages 4, 8 and 7 were never reached.
            self.assertEqual(list(rows), [])
            self.assertEqual(pages.calls, [(9, (2, 6))])
            self.assertEqual(list(Path(scratch).iterdir()), [])


class LateRankedPageFailureTest(unittest.TestCase):
    def _two_page_hits(self):
        return [Hit(1, 100, 10, 1), Hit(2, 99, 20, 2)]

    def test_missing_slot_for_a_later_rank_fails_the_whole_page(self):
        hits = [Hit(1, 100, 10, 1), Hit(2, 99, 20, 2),
                Hit(3, 98, 20, 3), Hit(4, 97, 30, 1)]

        def behavior(page, slots):
            if page == 20:
                return {2: "only-slot-2"}       # slot 3 (needed at rank 2) missing
            return {slot: f"page{page}-slot{slot}" for slot in slots}

        pages = _Pages(behavior)
        with TemporaryDirectory() as scratch:
            with late_ranked_rows(iter(hits), pages, limit=4, run_rows=1,
                                  fan_in=2, scratch=scratch) as rows:
                first = next(rows)
                self.assertEqual(first, (Hit(1, 100, 10, 1), "page10-slot1"))
                # The page is rejected as a whole: not even its first rank
                # is delivered, and the iterator is terminated.
                with self.assertRaises(ValueError):
                    next(rows)
                self.assertRaises(StopIteration, next, rows)
                # Page 30, due only later, was never read.
                self.assertEqual(pages.calls, [(10, (1,)), (20, (2, 3))])
            # The previously delivered rank is not revoked.
            self.assertEqual(first, (Hit(1, 100, 10, 1), "page10-slot1"))
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_reader_exception_propagates_and_terminates(self):
        def behavior(page, slots):
            raise RuntimeError("backend down")

        pages = _Pages(behavior)
        with TemporaryDirectory() as scratch:
            with late_ranked_rows(iter(self._two_page_hits()), pages, limit=2,
                                  run_rows=1, fan_in=2, scratch=scratch) as rows:
                with self.assertRaisesRegex(RuntimeError, "backend down"):
                    next(rows)
                self.assertRaises(StopIteration, next, rows)
                self.assertEqual(pages.calls, [(10, (1,))])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_invalid_responses_are_rejected(self):
        bad_responses = [
            ("missing slot", {}),
            ("extra slot", {1: "a", 2: "b"}),
            ("wrong slot", {2: "a"}),
            ("boolean key", {True: "a"}),
            ("non-string payload", {1: 17}),
            ("oversized payload", {1: "y" * 4097}),
            ("non-utf8 payload", {1: "\ud800"}),
            ("non-dict response", [("slot", "payload")]),
        ]
        for name, bad in bad_responses:
            with self.subTest(name=name):
                pages = _Pages(lambda page, slots: bad)
                with TemporaryDirectory() as scratch:
                    with late_ranked_rows(iter(self._two_page_hits()), pages,
                                          limit=2, run_rows=1, fan_in=2,
                                          scratch=scratch) as rows:
                        with self.assertRaises(ValueError):
                            next(rows)
                        self.assertRaises(StopIteration, next, rows)
                        self.assertEqual(pages.calls, [(10, (1,))])
                    self.assertEqual(list(Path(scratch).iterdir()), [])


class LateRankedValidationTest(unittest.TestCase):
    def test_invalid_parameters(self):
        pages = _Pages()
        good = {"limit": 1, "run_rows": 1, "fan_in": 2}
        overrides = [
            {"limit": -1}, {"limit": 200_001}, {"limit": True},
            {"limit": 1.5}, {"limit": "3"},
            {"run_rows": 0}, {"run_rows": 257}, {"run_rows": True},
            {"run_rows": 1.0},
            {"fan_in": 1}, {"fan_in": 17}, {"fan_in": False}, {"fan_in": 2.0},
            {"scratch": 123}, {"scratch": b"dir"},
        ]
        for override in overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    with late_ranked_rows(iter(()), pages,
                                          **{**good, **override}):
                        self.fail("invalid parameters accepted")
        self.assertEqual(pages.calls, [])

    def test_invalid_hit_fields(self):
        bad_hits = [
            Hit(True, 1, 1, 1), Hit(1, True, 1, 1),
            Hit(1, 1, True, 1), Hit(1, 1, 1, True),
            Hit(1 << 63, 1, 1, 1), Hit(-(1 << 63) - 1, 1, 1, 1),
            Hit(1, 1 << 63, 1, 1), Hit(1, -(1 << 63) - 1, 1, 1),
            Hit(1, 1, -1, 1), Hit(1, 1, 1 << 63, 1),
            Hit(1, 1, 1, -1), Hit(1, 1, 1, 128),
            "not-a-hit", None,
        ]
        for bad in bad_hits:
            with self.subTest(bad=bad):
                pages = _Pages()
                with self.assertRaises(ValueError):
                    with late_ranked_rows(iter([bad]), pages, limit=1,
                                          run_rows=1, fan_in=2):
                        self.fail("invalid hit accepted")
                self.assertEqual(pages.calls, [])

    def test_boundary_hit_fields_are_accepted(self):
        hits = [Hit(-(1 << 63), -(1 << 63), 0, 0),
                Hit((1 << 63) - 1, (1 << 63) - 1, (1 << 63) - 1, 127)]
        pages = _Pages()
        with late_ranked_rows(iter(hits), pages, limit=2,
                              run_rows=1, fan_in=2) as rows:
            got = list(rows)
        self.assertEqual([hit for hit, _ in got], [hits[1], hits[0]])
        self.assertEqual(pages.calls, [((1 << 63) - 1, (127,)), (0, (0,))])

    def test_invalid_last_hit_leaves_no_backend_reads(self):
        def hits():
            yield Hit(1, 100, 10, 1)
            yield Hit(2, 99, 20, 2)
            yield Hit(3, 98, 30, 3)
            yield Hit(4, 97, 40, 128)       # invalid slot on the last row

        pages = _Pages()
        with TemporaryDirectory() as scratch:
            with self.assertRaises(ValueError):
                with late_ranked_rows(hits(), pages, limit=2, run_rows=1,
                                      fan_in=2, scratch=scratch):
                    self.fail("invalid stream entered the context")
            self.assertEqual(pages.calls, [])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_upstream_failure_propagates_without_reads(self):
        def hits():
            yield Hit(1, 100, 10, 1)
            raise RuntimeError("index stream failed")

        pages = _Pages()
        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(RuntimeError, "index stream failed"):
                with late_ranked_rows(hits(), pages, limit=2, run_rows=1,
                                      fan_in=2, scratch=scratch):
                    self.fail("failing stream entered the context")
            self.assertEqual(pages.calls, [])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_zero_limit_still_validates_the_stream(self):
        pages = _Pages()
        with late_ranked_rows(iter([Hit(1, 100, 10, 1)]), pages, limit=0,
                              run_rows=1, fan_in=2) as rows:
            self.assertEqual(list(rows), [])
        self.assertEqual(pages.calls, [])
        with self.assertRaises(ValueError):
            with late_ranked_rows(iter([Hit(1, 100, 10, 128)]), pages, limit=0,
                                  run_rows=1, fan_in=2):
                self.fail("invalid stream accepted with limit=0")

    def test_row_limit(self):
        def hits():
            for seq in range(200_001):
                yield Hit(seq, seq, seq, 0)

        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "row 200000"):
                with late_ranked_rows(hits(), _Pages(), limit=1, run_rows=256,
                                      fan_in=16, scratch=scratch):
                    self.fail("oversized stream accepted")
            self.assertEqual(list(Path(scratch).iterdir()), [])


class LateRankedLifecycleTest(unittest.TestCase):
    def test_cursor_stops_after_context_exit(self):
        pages = _Pages()
        with TemporaryDirectory() as scratch:
            with late_ranked_rows(iter(_spec_hits(100)), pages, limit=3,
                                  run_rows=1, fan_in=2, scratch=scratch) as rows:
                self.assertEqual(next(rows), (Hit(1, 100, 9, 2), "page9-slot2"))
            self.assertEqual(list(rows), [])
            self.assertEqual(pages.calls, [(9, (2, 6))])
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_consumer_exception_propagates_and_cleans_up(self):
        pages = _Pages()
        with TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(RuntimeError, "consumer exited"):
                with late_ranked_rows(iter(_spec_hits(100)), pages, limit=3,
                                      run_rows=1, fan_in=2, scratch=scratch) as rows:
                    next(rows)
                    raise RuntimeError("consumer exited")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_scratch_keeps_caller_files(self):
        with TemporaryDirectory() as scratch:
            keep = Path(scratch) / "caller-file.txt"
            keep.write_text("caller data", encoding="utf-8")
            with late_ranked_rows(iter(_spec_hits(100)), _Pages(), limit=3,
                                  run_rows=1, fan_in=2, scratch=scratch) as rows:
                self.assertEqual(len(list(rows)), 3)
            self.assertEqual(keep.read_text(encoding="utf-8"), "caller data")
            self.assertEqual([p.name for p in Path(scratch).iterdir()],
                             ["caller-file.txt"])

    def test_shared_location_is_read_once_and_served_twice(self):
        hits = [Hit(1, 100, 5, 2), Hit(2, 99, 5, 2), Hit(3, 98, 5, 1)]
        pages = _Pages()
        with late_ranked_rows(iter(hits), pages, limit=3,
                              run_rows=1, fan_in=2) as rows:
            got = list(rows)
        self.assertEqual(got, [(Hit(1, 100, 5, 2), "page5-slot2"),
                               (Hit(2, 99, 5, 2), "page5-slot2"),
                               (Hit(3, 98, 5, 1), "page5-slot1")])
        self.assertEqual(pages.calls, [(5, (1, 2))])

    def test_empty_and_max_size_payloads(self):
        responses = {0: "", 127: "z" * 4096}
        pages = _Pages(lambda page, slots: {slot: responses[slot] for slot in slots})
        hits = [Hit(1, 100, 10, 0), Hit(2, 99, 10, 127)]
        with late_ranked_rows(iter(hits), pages, limit=2,
                              run_rows=1, fan_in=2) as rows:
            self.assertEqual(list(rows), [(Hit(1, 100, 10, 0), ""),
                                          (Hit(2, 99, 10, 127), "z" * 4096)])
        self.assertEqual(pages.calls, [(10, (0, 127))])

    def test_duplicate_heavy_stream(self):
        hits = [Hit(7, 100 - (i % 3), i % 4, i % 128) for i in range(5000)]
        pages = _Pages()
        with late_ranked_rows(iter(hits), pages, limit=5,
                              run_rows=4, fan_in=2) as rows:
            got = list(rows)
        # One group, one winner: highest score, earliest arrival.
        self.assertEqual(got, [(Hit(7, 100, 0, 0), "page0-slot0")])
        self.assertEqual(pages.calls, [(0, (0,))])


class LateRankedOracleTest(unittest.TestCase):
    def test_matches_oracle_with_read_schedule(self):
        rng = Random(20261126)
        for trial in range(30):
            hits = [Hit(rng.randrange(-5, 30), rng.randrange(-100, 101),
                        rng.randrange(0, 12), rng.randrange(0, 6))
                    for _ in range(rng.randrange(0, 120))]
            limit = rng.randrange(0, 25)
            expected = _oracle(hits, limit)
            pages = _Pages()
            with TemporaryDirectory() as scratch:
                with late_ranked_rows(iter(hits), pages, limit=limit,
                                      run_rows=rng.choice([1, 2, 5]),
                                      fan_in=rng.choice([2, 3]),
                                      scratch=scratch) as rows:
                    actual = list(rows)
                self.assertEqual([hit for hit, _ in actual], expected,
                                 msg=f"trial {trial}")
                for hit, payload in actual:
                    self.assertEqual(payload, f"page{hit.page}-slot{hit.slot}")
                requested = {}
                for page, slots in pages.calls:
                    self.assertNotIn(page, requested, msg=f"trial {trial}")
                    self.assertEqual(list(slots), sorted(set(slots)))
                    requested[page] = slots
                wanted = {}
                for hit in expected:
                    wanted.setdefault(hit.page, set()).add(hit.slot)
                self.assertEqual(
                    requested,
                    {page: tuple(sorted(slots)) for page, slots in wanted.items()},
                    msg=f"trial {trial}")
                self.assertEqual(list(Path(scratch).iterdir()), [])


class LateRankedResourceTest(unittest.TestCase):
    def test_interleaved_pages_keep_pending_payloads_on_disk(self):
        page_count = 3400
        payload_size = 4096
        # Ranks 0..N-1 each first-use a fresh page (slot 1); ranks N..2N-1
        # deliver slot 2 of the same pages in the same page order.  The
        # pending slot-2 payloads total ~13.3 MiB, above the 12 MiB budget.
        calls = []

        def hits():
            for index in range(page_count):
                yield Hit(2 * index, 2 * page_count - index, index, 1)
            for index in range(page_count):
                yield Hit(2 * index + 1, page_count - 1 - index, index, 2)

        def read_page(page, slots):
            calls.append((page, slots))
            return {slot: chr(ord("A") + slot) * payload_size for slot in slots}

        with TemporaryDirectory() as scratch:
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            try:
                with late_ranked_rows(hits(), read_page, limit=2 * page_count,
                                      run_rows=128, fan_in=8,
                                      scratch=scratch) as rows:
                    seen = 0
                    for rank, (hit, payload) in enumerate(rows):
                        if rank < page_count:
                            index, slot = rank, 1
                            group, score = 2 * index, 2 * page_count - index
                        else:
                            index, slot = rank - page_count, 2
                            group, score = 2 * index + 1, page_count - 1 - index
                        self.assertEqual(hit, Hit(group, score, index, slot))
                        self.assertEqual(payload,
                                         chr(ord("A") + slot) * payload_size)
                        seen = rank + 1
                    self.assertEqual(seen, 2 * page_count)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            # Every page read exactly once, in first-use order, both slots.
            self.assertEqual(calls, [(page, (1, 2)) for page in range(page_count)])
            self.assertLessEqual(peak - baseline, 12 * 1024 * 1024)
            self.assertEqual(list(Path(scratch).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
