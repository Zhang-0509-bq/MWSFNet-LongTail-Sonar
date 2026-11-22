"""Verify archived predictions without images, checkpoints, downloads, or training."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from ensemble_eval import cal_gmean, softmax

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("accuracy", "gmean", "macro_f1")


def read_partition(path, expected_hash):
    contents = path.read_bytes()
    if sha256(contents).hexdigest() != expected_hash:
        raise ValueError(f"Manifest checksum mismatch: {path.name}")
    return [line.rsplit(maxsplit=1) for line in contents.decode("utf-8").splitlines()]


def evaluate(truth, predictions, n_classes):
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "gmean": cal_gmean(truth, predictions),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
    }, recall_score(truth, predictions, labels=range(n_classes), average=None, zero_division=0)


def require_metrics(actual, recorded, description):
    for key in METRICS:
        if not np.isclose(actual[key], recorded[key], rtol=0, atol=1e-10):
            raise ValueError(f"{description}: {key} = {actual[key]:.12g}, recorded {recorded[key]:.12g}")


def verify(experiment, spec, dataset, per_class):
    archive = ROOT / "artifacts" / f"{experiment}.npz"
    if sha256(archive.read_bytes()).hexdigest() != spec["archive_sha256"]:
        raise ValueError(f"{experiment}: archive checksum mismatch")
    name = f'test_abs{dataset["suffix"]}.txt'
    rows = read_partition(ROOT / "splits" / spec["dataset"] / name,
                          dataset["manifest_sha256"][name])
    ids, labels = zip(*rows)
    true_labels = np.array(labels, dtype=int)
    n_classes = len(dataset["classes"])
    if len(set(ids)) != len(ids) or sorted(set(true_labels)) != list(range(n_classes)):
        raise ValueError(f"{experiment}: invalid test manifest")

    with np.load(archive, allow_pickle=False) as data:
        if not np.array_equal(data["sample_ids"], ids) or not np.array_equal(data["y_true"], true_labels):
            raise ValueError(f"{experiment}: archive samples or labels differ from the fixed test split")
        recorded = spec["recorded_runs"]
        if [r["seed"] for r in recorded] != [0, 1, 2] or not np.array_equal(data["seeds"], [0, 1, 2]):
            raise ValueError(f"{experiment}: seed order mismatch")
        logits = data["logits"]
        if logits.shape != (3, len(ids), n_classes) or not np.isfinite(logits).all():
            raise ValueError(f"{experiment}: invalid logits")
        predictions = data["y_pred"]
        if not np.array_equal(logits.argmax(axis=2), predictions):
            raise ValueError(f"{experiment}: archived predictions disagree with logits")

        runs = []
        for record, prediction in zip(recorded, predictions):
            run, recalls = evaluate(true_labels, prediction, n_classes)
            require_metrics(run, record["metrics"], f'{experiment} seed {record["seed"]}')
            if not np.allclose(recalls, record["recall"], rtol=0, atol=5.1e-7):
                raise ValueError(f"{experiment}: archived class recalls disagree")
            runs.append(run)
            if per_class:
                print(f'{experiment} seed {record["seed"]} class recall (%): ' +
                      ", ".join(f"{name} {recall:.2%}" for name, recall in zip(dataset["classes"], recalls)))

        mean_std = {}
        for k in METRICS:
            samples = np.array([m[k] for m in runs])
            expected = spec["recorded_summary"][k]
            if not np.allclose(samples, expected["runs"], rtol=0, atol=1e-10):
                raise ValueError(f"{experiment}: seed-wise {k} mismatch in summary")
            actual = np.array([samples.mean(), samples.std(ddof=1)])
            if not np.allclose(actual, [expected["mean"], expected["std"]], rtol=0, atol=1e-10):
                raise ValueError(f"{experiment}: {k} mean/std mismatch")
            mean_std[k] = f"{actual[0]:.2%} ± {actual[1]:.2%}"
        print(experiment + " (n=3): " + ", ".join(f"{key} {value}" for key, value in mean_std.items()))

        if "ensemble" in spec:
            first, second, third = map(softmax, logits)
            ensemble_pred = ((first + second + third) / 3).argmax(axis=1)
            if not np.array_equal(ensemble_pred, data["ensemble_y_pred"]):
                raise ValueError(f"{experiment}: soft vote disagrees with the archived ensemble predictions")
            voted, _ = evaluate(true_labels, ensemble_pred, n_classes)
            require_metrics(voted, spec["ensemble"]["metrics"], f"{experiment} ensemble")
            print(experiment + " ensemble: " + ", ".join(f"{key} {value:.2%}" for key, value in voted.items()))


def main():
    with (ROOT / "configs" / "canonical.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    experiments = metadata["experiments"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=experiments, help="Verify one ID instead of all recordings.")
    parser.add_argument("--per-class", action="store_true", help="Print each class recall for every seed.")
    args = parser.parse_args()

    for name in [args.experiment] if args.experiment else experiments:
        spec = experiments[name]
        verify(name, spec, metadata["datasets"][spec["dataset"]], args.per_class)
    print("Archived metrics verified with no model execution and no result files written.")


if __name__ == "__main__":
    main()
