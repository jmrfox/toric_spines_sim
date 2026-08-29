from pathlib import Path
import sys

"""
Converts NFF pointsets to text files.
"""


def convert_nff_to_txt() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from toric_spines_sim.utils import read_nff_points_and_write_txt_file

    src_dir = base_dir / "data" / "nff"
    dst_dir = base_dir / "data" / "pointsets"

    for nff_path in sorted(src_dir.glob("*.nff")):
        out_path = dst_dir / f"{nff_path.stem}.txt"
        read_nff_points_and_write_txt_file(nff_path, out_path)


def main() -> None:
    convert_nff_to_txt()


if __name__ == "__main__":
    main()
