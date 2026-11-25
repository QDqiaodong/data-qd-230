from pathlib import Path
from tempfile import TemporaryDirectory

from external_sort import Row, sorted_rows


if __name__ == "__main__":
    data = [(4, "A"), (2, "B"), (4, "C"), (0, "D"), (2, "E"), (4, "F"), (1, "G")]
    with TemporaryDirectory() as scratch:
        with sorted_rows((Row(*t) for t in data), run_rows=1, fan_in=2,
                         scratch=scratch) as result:
            ordered = [(row.key, row.payload) for row in result]
        assert ordered == [(0, "D"), (1, "G"), (2, "B"), (2, "E"), (4, "A"), (4, "C"), (4, "F")]
        assert not list(Path(scratch).iterdir())
        print("ordered:", ordered)
        print("scratch files remaining: 0")
