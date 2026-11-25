from contextlib import nullcontext
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest

from external_sort import Row, sorted_rows


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
