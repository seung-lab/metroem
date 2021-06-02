import json
import pandas as pd
import argparse
from pathlib import Path


def parse_df(df):
    """Given CSV from neuroglancer export, extract and clean points"""
    c1 = "Coordinate 1"
    coord_cols = ["x0", "y0", "z0"]
    exported = c1 in df.columns
    if exported:
        df[["x0", "y0", "z0"]] = (
            df[c1]
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.split(", ", expand=True)
            .astype(int)
        )
    for coord_col in coord_cols:
        assert coord_col in df.columns
    return df[coord_cols]


def adjust_coord(coord, max_mip):
    """Snap MIP0 coord to nearest MAX_MIP coord & convert to int"""
    return int((coord // 2 ** max_mip) * 2 ** max_mip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add pairs to dataset spec")
    parser.add_argument("--src_spec_path", type=str)
    parser.add_argument("--dst_spec_path", type=str)
    parser.add_argument("--points_path", type=str)
    parser.add_argument("--offsets", type=int, nargs="+")
    parser.add_argument("--permute_pairs", action="store_true")

    args = parser.parse_args()
    with open(args.src_spec_path, "r") as f:
        spec = json.load(f)
    max_mip = spec["max_mip"]
    df = pd.read_csv(args.points_path, dtype=str)
    df = parse_df(df)
    for _, row in df.iterrows():
        for i in args.offsets:
            iters = 1
            if args.permute_pairs:
                iters = 2
            for j in range(iters):
                pair_offsets = [1 + i, i]
                if j % 2 == 1:
                    pair_offsets = pair_offsets[::-1]
                pair = []
                for k in pair_offsets:
                    img = {}
                    img["x"] = adjust_coord(row["x0"], max_mip)
                    img["y"] = adjust_coord(row["y0"], max_mip)
                    img["z"] = int(row["z0"] + k)
                    pair.append(img)
                spec["pairs"].append(pair)

    dst_spec_path = Path(args.dst_spec_path)
    dst_spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst_spec_path, "w") as f:
        json.dump(spec, f, indent=4)
