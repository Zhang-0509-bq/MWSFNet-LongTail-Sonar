"""Point the reference loader at local dataset files using the archived fixed splits."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def prepare(dataset, image_root, destination, check_only):
    specs = json.loads((ROOT / "configs" / "canonical.json").read_text(encoding="utf-8"))
    spec = specs["datasets"][dataset]
    source = ROOT / "splits" / dataset
    files = {}
    for name, checksum in spec["manifest_sha256"].items():
        raw = (source / name).read_bytes()
        if sha256(raw).hexdigest() != checksum:
            raise ValueError(f"{name}: manifest checksum mismatch")
        if name.startswith("kfold_"):
            files[name] = raw
            continue

        paths = []
        for row in raw.decode("utf-8").splitlines():
            logical, label = row.rsplit(maxsplit=1)
            relative = PurePosixPath(logical)
            if relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts:
                raise ValueError(f"Invalid dataset-relative path: {logical}")
            candidate = image_root.joinpath(*relative.parts)
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing dataset image: {candidate}")
            if any(c.isspace() for c in str(candidate)):
                raise ValueError("The original loader uses whitespace-delimited entries; choose an image root without spaces.")
            paths.append(f"{candidate.as_posix()} {int(label)}\n")
        files[name] = "".join(paths).encode("utf-8")

    if not check_only:
        for name, content in files.items():
            target = destination / name
            if target.exists() and target.read_bytes() != content:
                raise FileExistsError(f"An existing file differs from the archived manifest: {target}")
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            target = destination / name
            if not target.exists():
                target.write_bytes(content)

    train, val, test = spec["counts"]
    action = "Checked" if check_only else "Prepared"
    print(f"{action} {dataset} p0/k0: {train} training, {val} validation, {test} test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("NKSID", "FLSMDD"))
    parser.add_argument("image_root", type=Path, help="Folder containing the original category folders.")
    parser.add_argument("--output", type=Path, help="Manifest output directory (default: data/<dataset>).")
    parser.add_argument("--check-only", action="store_true", help="Validate file paths without writing files.")
    args = parser.parse_args()
    prepare(args.dataset, args.image_root.resolve(),
            args.output or ROOT / "data" / args.dataset, args.check_only)


if __name__ == "__main__":
    main()
