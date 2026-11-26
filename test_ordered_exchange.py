"""Ordered, budgeted external exchange acceptance tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import tracemalloc
import unittest

from external_sort import Row, ordered_exchange


MIB = 1024 * 1024
KEY_MAX = (1 << 63) - 1


def directory_logical_bytes(root):
    """Sum current sizes of all regular files below root."""
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # A watcher can race an unlink; the next sample is sufficient.
            pass
    return total


def wide_payload(round_index, kind, row_index):
    """Return a deterministic, varying payload of exactly 4000 UTF-8 bytes."""
    prefix = f"r{round_index:02d}-{kind}-{row_index:04d}-".encode("ascii")
    payload = (prefix + b"0123456789abcdef" * 300)[:4000]
    return payload.decode("ascii")


class OrderedExchangeSpecTest(unittest.TestCase):
    def test_headline_budget_order_retry_seq_and_completion(self):
        payload = "a" * 8  # cost = 32 + 8 = 40

        with ordered_exchange([40, 40], shared_bytes=40) as exchange:
            # Shard 1 deliberately arrives out of key order.
            self.assertTrue(exchange.offer(1, Row(9, payload)))
            self.assertTrue(exchange.offer(1, Row(2, payload)))

            # Shard 1's reserve and all shared overflow are now used.
            self.assertFalse(exchange.offer(1, Row(10, payload)))

            # Shard 0 still has its own idle reserve available.
            self.assertTrue(exchange.offer(0, Row(2, payload)))

            exchange.advance(1, 10)
            exchange.advance(0, 2)
            self.assertIsNone(exchange.take())

            exchange.advance(0, 3)

            first = exchange.take()
            self.assertIsNotNone(first)
            shard, seq, row = first
            self.assertEqual((shard, seq, row), (0, 0, Row(2, payload)))

            # Shard 1's key 2 has not yet been consumed/deducted, so the
            # rejected key 10 still cannot be admitted.
            self.assertFalse(exchange.offer(1, Row(10, payload)))

            second = exchange.take()
            self.assertEqual(second, (1, 1, Row(2, payload)))

            # The failed attempt did not consume sequence number 2.
            self.assertTrue(exchange.offer(1, Row(10, payload)))

            # Finish shard 0 first.  Shard 1's bound 10 still releases key 9
            # but strictly excludes the equal-bound key 10.
            exchange.finish(0)
            self.assertEqual(exchange.take(), (1, 0, Row(9, payload)))
            self.assertIsNone(exchange.take())
            self.assertFalse(exchange.done())

            exchange.finish(1)
            self.assertEqual(exchange.take(), (1, 2, Row(10, payload)))
            self.assertIsNone(exchange.take())
            self.assertTrue(exchange.done())

        with self.assertRaises(RuntimeError):
            exchange.take()

    def test_idle_reserve_is_not_lent_to_another_shard(self):
        payload = "a" * 8  # cost 40
        with ordered_exchange([40, 40], shared_bytes=0) as exchange:
            self.assertTrue(exchange.offer(1, Row(0, payload)))
            # Shard 0's unused 40-byte reserve cannot cover shard 1 overflow.
            self.assertFalse(exchange.offer(1, Row(1, payload)))
            self.assertTrue(exchange.offer(0, Row(2, payload)))

    def test_backward_bound_late_row_bad_payload_and_finish_atomicity(self):
        with ordered_exchange([64, 64], shared_bytes=0) as exchange:
            payload = "a" * 8
            self.assertTrue(exchange.offer(0, Row(1, payload)))
            self.assertFalse(exchange.offer(0, Row(2, payload)))

            exchange.advance(0, 10)
            with self.assertRaises(ValueError):
                exchange.advance(0, 9)

            # The rejected backward move did not alter the current bound.
            with self.assertRaises(ValueError):
                exchange.offer(0, Row(9, payload))
            # The original budget-rejected row is now late as well.
            with self.assertRaises(ValueError):
                exchange.offer(0, Row(2, payload))
            # A key equal to the bound is legal, but still budget-blocked.
            self.assertFalse(exchange.offer(0, Row(10, payload)))

            bad_rows = (
                Row(11, "x" * 4097),
                Row(11, "\ud800"),
                Row(True, payload),
                Row(11, 123),
                object(),
            )
            for bad_row in bad_rows:
                with self.assertRaises(ValueError):
                    exchange.offer(0, bad_row)

            # Remove the other shard's initial minimum bound, consume key 1,
            # and only then retry key 10.
            exchange.finish(1)
            self.assertEqual(exchange.take(), (0, 0, Row(1, payload)))
            self.assertTrue(exchange.offer(0, Row(10, payload)))

            exchange.finish(0)
            exchange.finish(0)  # repeated finish is legal
            with self.assertRaises(ValueError):
                exchange.offer(0, Row(12, payload))
            with self.assertRaises(ValueError):
                exchange.advance(0, 12)

            # Only the two successful shard-0 rows are present.  The budget
            # rejection, late row and bad rows added nothing.
            self.assertEqual(exchange.take(), (0, 1, Row(10, payload)))
            self.assertIsNone(exchange.take())
            self.assertTrue(exchange.done())

    def test_early_finish_does_not_override_remaining_bound(self):
        with ordered_exchange([1024, 1024], shared_bytes=0) as exchange:
            exchange.offer(0, Row(1, "one"))
            exchange.offer(0, Row(100, "hundred"))
            exchange.finish(0)

            exchange.offer(1, Row(50, "fifty"))
            exchange.advance(1, 50)  # strict inequality: key 50 still blocked

            self.assertEqual(exchange.take(), (0, 0, Row(1, "one")))
            self.assertIsNone(exchange.take())
            self.assertFalse(exchange.done())

            exchange.advance(1, 51)
            self.assertEqual(exchange.take(), (1, 0, Row(50, "fifty")))
            # Bound 51 still blocks shard 0's key 100 through the unfinished
            # shard 1, even though shard 0 itself finished early.
            self.assertIsNone(exchange.take())

            exchange.advance(1, 101)
            self.assertEqual(exchange.take(), (0, 1, Row(100, "hundred")))
            self.assertIsNone(exchange.take())

            exchange.finish(1)
            self.assertTrue(exchange.done())

    def test_same_key_orders_by_shard_then_arrival_sequence(self):
        with ordered_exchange([4096, 4096], shared_bytes=0) as exchange:
            for payload in ("a", "b", "c"):
                exchange.offer(0, Row(7, payload))
            exchange.offer(1, Row(7, "x"))
            exchange.advance(0, 8)
            exchange.advance(1, 8)

            expected = [
                (0, 0, Row(7, "a")),
                (0, 1, Row(7, "b")),
                (0, 2, Row(7, "c")),
                (1, 0, Row(7, "x")),
            ]
            for wanted in expected:
                self.assertEqual(exchange.take(), wanted)
            self.assertIsNone(exchange.take())

    def test_maximum_key_is_safe_only_after_every_shard_finishes(self):
        with ordered_exchange([4096, 4096], shared_bytes=0) as exchange:
            exchange.offer(0, Row(KEY_MAX, "max-0"))
            exchange.offer(0, Row(1, "low"))
            exchange.offer(1, Row(KEY_MAX, "max-1"))

            exchange.advance(0, KEY_MAX)
            exchange.advance(1, KEY_MAX)

            self.assertEqual(exchange.take(), (0, 1, Row(1, "low")))
            self.assertIsNone(exchange.take())

            exchange.finish(0)
            # One unfinished shard still has bound KEY_MAX.
            self.assertIsNone(exchange.take())

            exchange.finish(1)
            self.assertEqual(exchange.take(), (0, 0, Row(KEY_MAX, "max-0")))
            self.assertEqual(exchange.take(), (1, 0, Row(KEY_MAX, "max-1")))
            self.assertTrue(exchange.done())

    def test_invalid_arguments_and_entry_time_snapshot(self):
        invalid_cases = [
            {"reserved_bytes": [], "shared_bytes": 0},
            {"reserved_bytes": list(range(9)), "shared_bytes": 0},
            {"reserved_bytes": [-1], "shared_bytes": 0},
            {"reserved_bytes": [True], "shared_bytes": 0},
            {"reserved_bytes": [1.5], "shared_bytes": 0},
            {"reserved_bytes": (32,), "shared_bytes": -1},
            {"reserved_bytes": (32,), "shared_bytes": True},
            {"reserved_bytes": (32,), "shared_bytes": 1.5},
            {"reserved_bytes": "32", "shared_bytes": 0},
            {"reserved_bytes": [16 * MIB], "shared_bytes": 1},
            {"reserved_bytes": [8 * MIB, 8 * MIB], "shared_bytes": 1},
        ]
        for kwargs in invalid_cases:
            with self.assertRaises(ValueError):
                with ordered_exchange(**kwargs):
                    self.fail(f"accepted invalid arguments: {kwargs!r}")

        # The mutable argument is read when __enter__ runs, not in __init__.
        reserved = [-1]
        context = ordered_exchange(reserved, shared_bytes=0)
        reserved[0] = 32
        with context as exchange:
            self.assertTrue(exchange.offer(0, Row(0, "")))

    def test_invalid_shards_and_equal_bounds(self):
        with ordered_exchange([32, 32], shared_bytes=0) as exchange:
            for bad_shard in (-1, 2, True, 1.0, "0"):
                with self.assertRaises(ValueError):
                    exchange.offer(bad_shard, Row(0, ""))
                with self.assertRaises(ValueError):
                    exchange.advance(bad_shard, 0)
                with self.assertRaises(ValueError):
                    exchange.finish(bad_shard)

            exchange.advance(0, 5)
            exchange.advance(0, 5)  # repeating the same bound is legal
            self.assertTrue(exchange.offer(0, Row(5, "")))

    def test_context_exception_and_early_exit_clean_scratch(self):
        with TemporaryDirectory() as scratch:
            sentinel = Path(scratch) / "caller-file.txt"
            sentinel.write_text("keep me", encoding="utf-8")

            exchange_ref = []
            with self.assertRaisesRegex(RuntimeError, "caller failure"):
                with ordered_exchange([32], shared_bytes=0,
                                      scratch=scratch) as exchange:
                    exchange_ref.append(exchange)
                    self.assertTrue(exchange.offer(0, Row(1, "")))
                    raise RuntimeError("caller failure")

            exchange = exchange_ref[0]
            for operation in (
                lambda: exchange.offer(0, Row(2, "")),
                lambda: exchange.advance(0, 2),
                lambda: exchange.finish(0),
                lambda: exchange.take(),
                exchange.done,
            ):
                with self.assertRaises(RuntimeError):
                    operation()

            self.assertEqual(
                sorted(p.name for p in Path(scratch).iterdir()),
                ["caller-file.txt"],
            )

        with TemporaryDirectory() as scratch:
            with ordered_exchange([64], shared_bytes=0,
                                  scratch=scratch) as exchange:
                exchange.offer(0, Row(1, "a"))
                exchange.offer(0, Row(2, "b"))
                exchange.advance(0, 3)
                self.assertEqual(exchange.take()[2], Row(1, "a"))
                # Deliberately leave the second row and exit early.

            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_200000_row_limit_is_not_consumable_backpressure(self):
        with ordered_exchange([8 * MIB], shared_bytes=0) as exchange:
            for seq in range(200_000):
                if not exchange.offer(0, Row(seq, "")):
                    self.fail(f"budget rejected row {seq}")

            with self.assertRaises(ValueError):
                exchange.offer(0, Row(200_000, ""))

            # Consumption cannot restore the hard receive limit.
            exchange.advance(0, 1)
            self.assertIsNotNone(exchange.take())
            with self.assertRaises(ValueError):
                exchange.offer(0, Row(200_001, ""))


class OrderedExchangeGeneratedResourceTest(unittest.TestCase):
    ROUNDS = 9
    LOW_ROWS = 3200
    HIGH_ROWS = 50
    BLOCK = 100_000
    LOW_CUTOFF_ADD = 4096
    HIGH_BASE = 1_000_000_000

    def test_nine_round_generated_wide_workload(self):
        scratch_context = TemporaryDirectory()
        scratch = scratch_context.name
        peak_staged = [directory_logical_bytes(scratch)]
        stop_watcher = threading.Event()

        def watch_directory():
            while not stop_watcher.wait(0.001):
                peak_staged[0] = max(
                    peak_staged[0], directory_logical_bytes(scratch))

        watcher = threading.Thread(
            target=watch_directory, name="scratch-watchdog", daemon=True)
        watcher.start()

        started_trace = not tracemalloc.is_tracing()
        if started_trace:
            tracemalloc.start()
        if hasattr(tracemalloc, "reset_peak"):
            tracemalloc.reset_peak()
        memory_baseline = tracemalloc.get_traced_memory()[0]
        received_total = 0
        delivered_total = 0
        pending_cost = 0
        max_pending_cost = 0
        memory_peak = 0
        stage_samples = []

        try:
            with ordered_exchange([0, 0], shared_bytes=16 * MIB,
                                  scratch=scratch) as exchange:
                for round_index in range(self.ROUNDS):
                    base = round_index * self.BLOCK
                    cutoff = base + self.LOW_CUTOFF_ADD

                    # Descending indices make the low-key input out of order
                    # without retaining a list of wide payloads.
                    for reverse_index in range(self.LOW_ROWS):
                        index = self.LOW_ROWS - 1 - reverse_index
                        shard = index & 1
                        payload = wide_payload(round_index, "low", index)
                        accepted = exchange.offer(
                            shard, Row(base + index, payload))
                        self.assertTrue(accepted)
                        received_total += 1
                        pending_cost += 32 + len(payload.encode("utf-8"))

                    for index in range(self.HIGH_ROWS):
                        shard = index & 1
                        payload = wide_payload(round_index, "high", index)
                        accepted = exchange.offer(
                            shard,
                            Row(self.HIGH_BASE + round_index * 1000 + index,
                                payload))
                        self.assertTrue(accepted)
                        received_total += 1
                        pending_cost += 32 + len(payload.encode("utf-8"))

                    max_pending_cost = max(max_pending_cost, pending_cost)
                    filled_bytes = directory_logical_bytes(scratch)
                    peak_staged[0] = max(peak_staged[0], filled_bytes)

                    # This round's low prefix is now safe; old prefixes have
                    # already gone and future/high keys remain beyond cutoff.
                    exchange.advance(0, cutoff)
                    exchange.advance(1, cutoff)

                    consumed_this_round = 0
                    mid_consume_bytes = filled_bytes
                    while True:
                        item = exchange.take()
                        if item is None:
                            break
                        shard, seq, row = item
                        index = consumed_this_round
                        expected_key = base + index
                        expected_shard = index & 1
                        if expected_shard == 0:
                            within_round_seq = (
                                self.LOW_ROWS - 2 - index) // 2
                        else:
                            within_round_seq = (
                                self.LOW_ROWS - 1 - index) // 2
                        expected_seq = (
                            round_index
                            * (self.LOW_ROWS // 2 + self.HIGH_ROWS // 2)
                            + within_round_seq
                        )

                        self.assertEqual(row.key, expected_key)
                        self.assertEqual(shard, expected_shard)
                        self.assertEqual(seq, expected_seq)
                        self.assertEqual(
                            row.payload,
                            wide_payload(round_index, "low", index))

                        consumed_this_round += 1
                        delivered_total += 1
                        pending_cost -= 4032
                        if consumed_this_round == self.LOW_ROWS // 2:
                            mid_consume_bytes = directory_logical_bytes(scratch)

                    self.assertEqual(consumed_this_round, self.LOW_ROWS)
                    self.assertIsNone(exchange.take())
                    self.assertFalse(exchange.done())

                    recycled_bytes = directory_logical_bytes(scratch)
                    peak_staged[0] = max(
                        peak_staged[0], mid_consume_bytes, recycled_bytes)
                    stage_samples.append(
                        (filled_bytes, mid_consume_bytes, recycled_bytes))

                    _, absolute_peak = tracemalloc.get_traced_memory()
                    round_memory_peak = absolute_peak - memory_baseline
                    memory_peak = max(memory_peak, round_memory_peak)
                    print(
                        f"第{round_index + 1}/9轮: "
                        f"累计接收{received_total}，本轮交付{consumed_this_round}，"
                        f"待交付额度{pending_cost / 1024:.0f}KiB；"
                        f"暂存 填{filled_bytes / MIB:.2f}/"
                        f"中{mid_consume_bytes / MIB:.2f}/"
                        f"收{recycled_bytes / MIB:.2f}MiB，"
                        f"内存峰{round_memory_peak / MIB:.2f}MiB"
                    )

                    # Delivered low data must be recycled rather than growing
                    # another full low range every round.  A broad slack still
                    # allows SQLite metadata/page overhead.
                    retained_high_payload = (
                        (round_index + 1) * self.HIGH_ROWS * 4000)
                    self.assertLessEqual(
                        recycled_bytes,
                        retained_high_payload + 4 * MIB)
                    self.assertLess(recycled_bytes, filled_bytes)

                # High rows are still blocked by the last bounds, even though
                # every low prefix has been consumed.
                self.assertFalse(exchange.done())
                self.assertIsNone(exchange.take())

                exchange.finish(0)
                exchange.finish(1)

                high_delivered = 0
                for round_index in range(self.ROUNDS):
                    for index in range(self.HIGH_ROWS):
                        item = exchange.take()
                        self.assertIsNotNone(item)
                        shard, seq, row = item
                        expected_shard = index & 1
                        expected_seq = (
                            round_index
                            * (self.LOW_ROWS // 2 + self.HIGH_ROWS // 2)
                            + self.LOW_ROWS // 2
                            + index // 2
                        )
                        expected_key = (
                            self.HIGH_BASE + round_index * 1000 + index)
                        self.assertEqual(row.key, expected_key)
                        self.assertEqual(shard, expected_shard)
                        self.assertEqual(seq, expected_seq)
                        self.assertEqual(
                            row.payload,
                            wide_payload(round_index, "high", index))
                        high_delivered += 1
                        delivered_total += 1
                        pending_cost -= 4032

                self.assertEqual(high_delivered, self.ROUNDS * self.HIGH_ROWS)
                self.assertIsNone(exchange.take())
                self.assertTrue(exchange.done())
                self.assertEqual(pending_cost, 0)
                self.assertEqual(received_total, self.ROUNDS * 3250)
                self.assertEqual(delivered_total, received_total)

            # The private exchange directory is removed by context exit.
            self.assertEqual(list(Path(scratch).iterdir()), [])
            _, absolute_peak = tracemalloc.get_traced_memory()
            memory_peak = max(
                memory_peak, absolute_peak - memory_baseline)
        finally:
            stop_watcher.set()
            watcher.join(timeout=2)
            if watcher.is_alive():
                # The loop has a 1ms wait; joining should be immediate.
                raise AssertionError("scratch watchdog did not stop")
            if started_trace:
                tracemalloc.stop()
            scratch_context.cleanup()

        peak_staged_bytes = peak_staged[0]
        print(
            f"全程: 接收并交付{received_total}行；"
            f"最大待交付额度{max_pending_cost / MIB:.2f}MiB；"
            f"tracemalloc新增峰值{memory_peak / 1024:.0f}KiB；"
            f"暂存采样峰值{peak_staged_bytes / MIB:.2f}MiB"
        )

        # The generated business invariants.
        self.assertGreater(self.LOW_ROWS * 4000, 8 * MIB)
        self.assertGreater(
            self.ROUNDS * (self.LOW_ROWS + self.HIGH_ROWS) * 4000,
            64 * MIB)
        self.assertLessEqual(max_pending_cost, 16 * MIB)

        # Required resource ceilings.
        self.assertLessEqual(memory_peak, 8 * MIB)
        self.assertLessEqual(peak_staged_bytes, 64 * MIB)

        # Every explicitly measured phase must also obey the staging ceiling.
        for filled, middle, recycled in stage_samples:
            self.assertLessEqual(filled, 64 * MIB)
            self.assertLessEqual(middle, 64 * MIB)
            self.assertLessEqual(recycled, 64 * MIB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
