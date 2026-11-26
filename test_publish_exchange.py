"""Acceptance tests for ordered_exchange's page-publishing loader mode."""
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import tracemalloc
import unittest

from external_sort import Row, ordered_exchange


MIB = 1024 * 1024


def directory_logical_bytes(root):
    """Sum the current sizes of all regular files below root."""
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # A watcher can race an unlink; the next sample is enough.
            pass
    return total


def wide_payload(round_index, index):
    """A deterministic, varying payload of exactly 4000 UTF-8 bytes."""
    prefix = f"r{round_index:03d}-{index:05d}-".encode("ascii")
    payload = (prefix + b"0123456789abcdef" * 300)[:4000]
    return payload.decode("ascii")


class _RecordingLoader:
    """A loader that records committed rows without retaining the payloads.

    ``commit_before_raise`` models a loader that persists the whole page and
    then fails, so the exception does not reveal whether the page committed.
    The loader is idempotent per ``(page_id, signature)``: a repeat with the
    same identity and order never adds a second load.
    """

    def __init__(self, fail_page=None, fail_times=1, commit_before_raise=False):
        self.fail_page = fail_page
        self.fail_times = fail_times
        self.failures_left = fail_times if fail_page is not None else 0
        self.commit_before_raise = commit_before_raise
        self.pages = {}          # page_id -> tuple of (shard, seq, key)
        self.loaded = []         # (page_id, shard, seq, key, payload_len)
        self.loaded_rows = 0
        self.attempts = []       # (page_id, signature, failed)
        self.failed_pages = {}   # page_id -> rows tuple captured at failure

    def __call__(self, page_id, rows):
        signature = tuple(
            (shard, seq, row.key, len(row.payload))
            for shard, seq, row in rows)
        failed = (
            self.fail_page is not None
            and page_id == self.fail_page
            and self.failures_left > 0)
        self.attempts.append((page_id, signature, failed))
        if failed:
            self.failures_left -= 1
            self.failed_pages.setdefault(page_id, rows)
            if not self.commit_before_raise:
                # Nothing was written; the retry must reproduce this page.
                raise RuntimeError(f"injected failure on page {page_id}")

        identity = tuple((shard, seq, row.key) for shard, seq, row in rows)
        if page_id in self.pages:
            assert self.pages[page_id] == identity, (
                "retry changed the page identity or order")
        else:
            self.pages[page_id] = identity
            for shard, seq, row in rows:
                self.loaded.append(
                    (page_id, shard, seq, row.key, len(row.payload)))
                self.loaded_rows += 1

        if failed:
            # The page landed but the caller still sees a failure.
            raise RuntimeError(f"injected failure on page {page_id}")


class _CountingLoader:
    """A loader that keeps only per-page summaries, never full payloads.

    It still lets the test verify cross-page ordering and uniqueness by
    tracking a strictly increasing key and contiguous page ranges while the
    callback runs, without retaining any returned payload.
    """

    def __init__(self, fail_page=None):
        self.fail_page = fail_page
        self.failed_once = False
        self.loaded_rows = 0
        self.pages = {}          # page_id -> (count, first_key, last_key)
        self.attempts = 0
        self.failures = 0
        self._last_key = None
        self._last_page = None

    def __call__(self, page_id, rows):
        self.attempts += 1
        if page_id == self.fail_page and not self.failed_once:
            self.failed_once = True
            self.failures += 1
            raise RuntimeError(f"injected failure on page {page_id}")
        count = 0
        first = last = None
        for shard, seq, row in rows:
            key = row.key
            if self._last_key is not None:
                assert key > self._last_key, "global order regressed"
                if page_id == self._last_page:
                    assert key == self._last_key + 1, "page not contiguous"
            self._last_key = key
            self._last_page = page_id
            if first is None:
                first = key
            last = key
            count += 1
        self.pages[page_id] = (count, first, last)
        self.loaded_rows += count


class PublishHeadlineTest(unittest.TestCase):
    def _run(self, delivery_bytes):
        # The loader fully records page 0 and only then raises, so the
        # exception does not reveal whether the page committed.
        loader = _RecordingLoader(fail_page=0, commit_before_raise=True)
        payload = "a" * 8  # cost = 32 + 8 = 40
        observations = []

        with ordered_exchange([0, 0], shared_bytes=120,
                              publisher=loader, page_rows=4,
                              delivery_bytes=delivery_bytes) as exchange:
            self.assertTrue(exchange.offer(1, Row(2, payload)))
            self.assertTrue(exchange.offer(1, Row(3, payload)))
            self.assertTrue(exchange.offer(0, Row(2, payload)))
            exchange.advance(0, 3)
            exchange.advance(1, 3)

            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                exchange.flush()
            # Page 0 held only the two key-2 rows, shard 0 first, and they
            # are already recorded exactly once.
            self.assertEqual(loader.pages[0], ((0, 0, 2), (1, 0, 2)))
            self.assertEqual(loader.loaded_rows, 2)
            self.assertEqual(
                [(s, q) for _, s, q, _, _ in loader.loaded], [(0, 0), (1, 0)])
            exchange.advance(0, 4)
            exchange.advance(1, 4)

            accepted_4 = exchange.offer(0, Row(4, payload))
            accepted_5 = exchange.offer(0, Row(5, payload))
            observations.append(
                f"delivery={delivery_bytes}: key4 accepted={accepted_4}, "
                f"key5 accepted={accepted_5}")
            self.assertEqual(accepted_4, accepted_5)
            if delivery_bytes == 79:
                self.assertFalse(accepted_4)
            else:
                self.assertIn(accepted_4, (True, False))

            # Retry must resend exactly the original page with no new writes.
            confirmed = exchange.flush()
            self.assertEqual(confirmed, 2)
            self.assertEqual(loader.loaded_rows, 2)
            self.assertEqual(loader.pages[0], ((0, 0, 2), (1, 0, 2)))

            if not accepted_4:
                self.assertTrue(exchange.offer(0, Row(4, payload)))
                self.assertTrue(exchange.offer(0, Row(5, payload)))
                observations.append(
                    f"delivery={delivery_bytes}: re-offered key4/key5 accepted")

            # Production quota is full; confirming the page must not release
            # budget a second time.
            self.assertFalse(exchange.offer(0, Row(6, payload)))
            observations.append(
                f"delivery={delivery_bytes}: key6 rejected after confirm")

            exchange.finish(0)
            exchange.finish(1)
            self.assertEqual(exchange.flush(), 3)
            page1 = loader.pages[1]
            self.assertEqual(page1, ((1, 1, 3), (0, 1, 4), (0, 2, 5)))
            self.assertEqual(
                [(s, q) for _, s, q, _, _ in loader.loaded],
                [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)])
            self.assertTrue(exchange.done())
            self.assertEqual(exchange.flush(), 0)

        landed = [
            (page_id, shard, seq, key)
            for page_id, shard, seq, key, _ in loader.loaded]
        observations.append(
            f"delivery={delivery_bytes}: loaded {loader.loaded_rows} rows "
            f"across pages {sorted(loader.pages)}: {landed}")
        print("\n".join(observations))
        self.assertEqual(loader.loaded_rows, 5)
        self.assertEqual(sorted(loader.pages), [0, 1])

    def test_headline_delivery_79(self):
        self._run(79)

    def test_headline_delivery_80(self):
        self._run(80)


class PublishFailureTest(unittest.TestCase):
    def test_write_before_failure_not_done_until_retry_then_once(self):
        loader = _RecordingLoader(fail_page=0, fail_times=2)
        with TemporaryDirectory() as scratch:
            with ordered_exchange([4096], shared_bytes=0, scratch=scratch,
                                  publisher=loader, page_rows=8) as exchange:
                self.assertTrue(exchange.offer(0, Row(0, "zero")))
                self.assertTrue(exchange.offer(0, Row(1, "one")))

                # No safe rows yet: flush must not call the loader.
                self.assertEqual(exchange.flush(), 0)
                self.assertEqual(loader.attempts, [])

                exchange.advance(0, 2)

                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    exchange.flush()
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    exchange.flush()
                # Consecutive failures keep page id, order and payload intact.
                first_page, _, first_failed = loader.attempts[0]
                second_page, second_sig, second_failed = loader.attempts[1]
                self.assertTrue(first_failed and second_failed)
                self.assertEqual(first_page, 0)
                self.assertEqual(second_page, 0)
                self.assertEqual(loader.attempts[0][1], second_sig)
                self.assertEqual(loader.loaded_rows, 0)

                # Even after every shard ends, an unconfirmed page blocks done.
                exchange.finish(0)
                self.assertFalse(exchange.done())
                with self.assertRaises(RuntimeError):
                    exchange.take()

                self.assertEqual(exchange.flush(), 2)
                self.assertEqual(loader.loaded_rows, 2)
                self.assertEqual(
                    loader.pages[0], ((0, 0, 0), (0, 1, 1)))
                self.assertTrue(exchange.done())
                # A confirmed page is not republished and done stays true.
                self.assertEqual(exchange.flush(), 0)
                self.assertEqual(loader.loaded_rows, 2)

            print(
                f"write-before-failure: attempts={len(loader.attempts)}, "
                f"pages={sorted(loader.pages)}, loaded={loader.loaded_rows}")
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_take_is_forbidden_and_normal_mode_flush_rejected(self):
        loader = _RecordingLoader()
        with ordered_exchange([4096], shared_bytes=0,
                              publisher=loader, page_rows=8) as exchange:
            self.assertTrue(exchange.offer(0, Row(0, "zero")))
            exchange.advance(0, 1)
            with self.assertRaises(RuntimeError):
                exchange.take()
            # The rejected take changed nothing; publish still works.
            self.assertEqual(exchange.flush(), 1)
            self.assertEqual(loader.loaded_rows, 1)

        with ordered_exchange([4096], shared_bytes=0) as exchange:
            exchange.offer(0, Row(0, "zero"))
            with self.assertRaises(RuntimeError):
                exchange.flush()

    def test_unconfirmed_page_quit_cleans_staging(self):
        loader = _RecordingLoader(fail_page=0)
        references = []
        with TemporaryDirectory() as scratch:
            sentinel = Path(scratch) / "caller.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with ordered_exchange([4096], shared_bytes=0, scratch=scratch,
                                  publisher=loader, page_rows=8) as exchange:
                references.append(exchange)
                self.assertTrue(exchange.offer(0, Row(0, "zero")))
                exchange.advance(0, 1)
                with self.assertRaises(RuntimeError):
                    exchange.flush()
                self.assertFalse(exchange.done())
                # Leave with an unconfirmed page on purpose.
            exchange = references[0]
            for operation in (
                lambda: exchange.offer(0, Row(1, "one")),
                lambda: exchange.advance(0, 2),
                lambda: exchange.finish(0),
                lambda: exchange.flush(),
                lambda: exchange.take(),
                exchange.done,
            ):
                with self.assertRaises(RuntimeError):
                    operation()
            self.assertEqual(
                sorted(p.name for p in Path(scratch).iterdir()),
                ["caller.txt"])
            # The publisher never confirmed a page.
            self.assertEqual(loader.loaded_rows, 0)
        print(
            f"unconfirmed-exit: attempts={len(loader.attempts)}, "
            f"loaded={loader.loaded_rows}")

    def test_committed_then_raised_retry_does_not_double_load(self):
        loader = _RecordingLoader(fail_page=0, commit_before_raise=True)
        with ordered_exchange([4096], shared_bytes=0,
                              publisher=loader, page_rows=8) as exchange:
            self.assertTrue(exchange.offer(0, Row(0, "zero")))
            self.assertTrue(exchange.offer(0, Row(1, "one")))
            exchange.advance(0, 2)
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                exchange.flush()
            # The loader had already written the page.
            self.assertEqual(loader.loaded_rows, 2)
            # Retrying the identical page must not add a second load.
            self.assertEqual(exchange.flush(), 2)
            self.assertEqual(loader.loaded_rows, 2)
            self.assertEqual(loader.pages[0], ((0, 0, 0), (0, 1, 1)))
            exchange.finish(0)
            self.assertTrue(exchange.done())
        print(
            f"commit-before-raise: attempts={len(loader.attempts)}, "
            f"pages={sorted(loader.pages)}, loaded={loader.loaded_rows}")

    def test_parameter_validation(self):
        invalid = [
            {"publisher": 5},
            {"publisher": "loader"},
            {"page_rows": 0},
            {"page_rows": 129},
            {"page_rows": True},
            {"page_rows": 4.0},
            {"delivery_bytes": -1},
            {"delivery_bytes": MIB + 1},
            {"delivery_bytes": False},
            {"delivery_bytes": 1.0},
        ]
        for extra in invalid:
            with self.assertRaises(ValueError):
                with ordered_exchange([32], shared_bytes=0, **extra):
                    self.fail(f"accepted invalid arguments {extra!r}")
        # Non-publishing mode still validates the new arguments.
        with self.assertRaises(ValueError):
            with ordered_exchange([32], shared_bytes=0, page_rows=0):
                self.fail("accepted invalid page_rows without a publisher")


class PublishGeneratedResourceTest(unittest.TestCase):
    ROUNDS = 6
    ROWS = 2200
    PAYLOAD = 4000
    PAGE_ROWS = 128
    DELIVERY = MIB

    def test_generated_wide_workload_page_order_and_resources(self):
        scratch_context = TemporaryDirectory()
        scratch = scratch_context.name
        loader = _CountingLoader(fail_page=0)
        peak_staged = [directory_logical_bytes(scratch)]
        stop_watcher = threading.Event()

        def watch_directory():
            while not stop_watcher.wait(0.001):
                peak_staged[0] = max(
                    peak_staged[0], directory_logical_bytes(scratch))

        watcher = threading.Thread(
            target=watch_directory, name="publish-watchdog", daemon=True)
        watcher.start()

        started_trace = not tracemalloc.is_tracing()
        if started_trace:
            tracemalloc.start()
        if hasattr(tracemalloc, "reset_peak"):
            tracemalloc.reset_peak()
        memory_baseline = tracemalloc.get_traced_memory()[0]
        memory_peak = 0
        offered_total = 0
        published_total = 0
        sampled_with_pending = []
        failure_measured = False

        try:
            with ordered_exchange(
                    [0, 0], shared_bytes=16 * MIB, scratch=scratch,
                    publisher=loader, page_rows=self.PAGE_ROWS,
                    delivery_bytes=self.DELIVERY) as exchange:
                for round_index in range(self.ROUNDS):
                    base = round_index * self.ROWS
                    for index in range(self.ROWS):
                        key = base + index
                        shard = key & 1
                        payload = wide_payload(round_index, index)
                        self.assertTrue(
                            exchange.offer(shard, Row(key, payload)))
                        offered_total += 1

                    # Pending delivery exists now; sample with rows in transit.
                    sampled_with_pending.append(
                        directory_logical_bytes(scratch))

                    exchange.advance(0, base + self.ROWS)
                    exchange.advance(1, base + self.ROWS)

                    while True:
                        try:
                            published = exchange.flush()
                        except RuntimeError:
                            # A page is pending in both exchange and loader.
                            sampled_with_pending.append(
                                directory_logical_bytes(scratch))
                            peak_staged[0] = max(
                                peak_staged[0], sampled_with_pending[-1])
                            if not failure_measured:
                                failure_measured = True
                                self.assertEqual(loader.loaded_rows, 0)
                                print(
                                    "injected page failure observed: "
                                    f"staged={sampled_with_pending[-1] / MIB:.2f}"
                                    f"MiB, loaded={loader.loaded_rows}")
                            continue
                        if published == 0:
                            break
                        published_total += published

                    _, absolute_peak = tracemalloc.get_traced_memory()
                    memory_peak = max(
                        memory_peak, absolute_peak - memory_baseline)
                    self.assertEqual(exchange.flush(), 0)
                    self.assertLessEqual(
                        directory_logical_bytes(scratch), 64 * MIB)

                exchange.finish(0)
                exchange.finish(1)
                while True:
                    published = exchange.flush()
                    if not published:
                        break
                    published_total += published
                self.assertTrue(exchange.done())
                _, absolute_peak = tracemalloc.get_traced_memory()
                memory_peak = max(memory_peak, absolute_peak - memory_baseline)

            # The counting loader already enforced strictly increasing keys
            # and contiguous in-page ranges for every page it accepted.
            self.assertEqual(loader.loaded_rows, offered_total)
            self.assertEqual(published_total, offered_total)
            self.assertEqual(loader.failures, 1)
            page_ids = sorted(loader.pages)
            self.assertEqual(page_ids, list(range(len(page_ids))))
            for page_id in page_ids:
                count, first, last = loader.pages[page_id]
                self.assertGreater(count, 0)
                self.assertLessEqual(count, self.PAGE_ROWS)
                self.assertEqual(last - first + 1, count)
            # Every accepted page except the very last is full up to the
            # requested size; round boundaries may end one early.
            full_pages = sum(
                1 for count, _, _ in loader.pages.values()
                if count == self.PAGE_ROWS)
            self.assertGreater(full_pages, 0)

            self.assertEqual(list(Path(scratch).iterdir()), [])
            _, absolute_peak = tracemalloc.get_traced_memory()
            memory_peak = max(memory_peak, absolute_peak - memory_baseline)
        finally:
            stop_watcher.set()
            watcher.join(timeout=2)
            if watcher.is_alive():
                raise AssertionError("publish watchdog did not stop")
            if started_trace:
                tracemalloc.stop()
            scratch_context.cleanup()

        peak_staged_bytes = peak_staged[0]
        delivered_payload = offered_total * self.PAYLOAD
        print(
            f"generated publish: offered={offered_total}, "
            f"loaded={loader.loaded_rows}, pages={len(loader.pages)}, "
            f"delivered payload={delivered_payload / MIB:.2f}MiB, "
            f"tracemalloc peak={memory_peak / 1024:.0f}KiB, "
            f"staged peak={peak_staged_bytes / MIB:.2f}MiB, "
            f"pending samples={[round(b / MIB, 2) for b in sampled_with_pending]}"
        )

        # Business invariants demanded by the task.
        self.assertGreater(self.ROWS * self.PAYLOAD, 8 * MIB)
        self.assertGreater(delivered_payload, 32 * MIB)
        self.assertTrue(failure_measured)
        # Every offered row landed exactly once.
        self.assertEqual(loader.loaded_rows, offered_total)
        # Resource ceilings.
        self.assertLessEqual(memory_peak, 8 * MIB)
        self.assertLessEqual(peak_staged_bytes, 64 * MIB)
        self.assertTrue(any(b > 8 * MIB for b in sampled_with_pending))
        for sample in sampled_with_pending:
            self.assertLessEqual(sample, 64 * MIB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
