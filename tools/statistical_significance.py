"""Compute paired sequence-level statistical significance for mAP metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path


TEMPLATE_HEADER = ["sequence_id", "map50"]


@dataclass(frozen=True)
class ModelInput:
    name: str
    path: Path


@dataclass(frozen=True)
class ComparisonResult:
    baseline: str
    candidate: str
    metric: str
    n_sequences: int
    mean_baseline: float
    mean_candidate: float
    mean_difference: float
    normality_test: str
    normality_p: float | None
    test_name: str
    statistic: float
    p_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sequence-level metrics with paired t-test or Wilcoxon signed-rank test."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeat as NAME=path/to/sequence_metrics.csv. CSV must include sequence_id and the metric column.",
    )
    parser.add_argument("--baseline", default=None, help="Baseline model name. Defaults to the first --model.")
    parser.add_argument("--pair", action="append", default=[], help="Optional explicit pair BASELINE=CANDIDATE.")
    parser.add_argument("--metric", default="map50", help="Metric column to compare.")
    parser.add_argument("--sequence-column", default="sequence_id", help="Sequence ID column.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Normality-test alpha.")
    parser.add_argument("--normality-test", choices=["shapiro", "none"], default="shapiro")
    parser.add_argument("--allow-missing", action="store_true", help="Use sequence intersection instead of failing.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV.")
    parser.add_argument("--write-template", type=Path, default=None, help="Write a blank input CSV template and exit.")
    return parser.parse_args()


def parse_model(value: str) -> ModelInput:
    if "=" not in value:
        raise ValueError(f"--model must use NAME=CSV syntax: {value}")
    name, path = value.split("=", 1)
    if not name.strip():
        raise ValueError(f"Empty model name in {value}")
    return ModelInput(name.strip(), Path(path).expanduser())


def read_metric_csv(path: Path, sequence_column: str, metric: str) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing_columns = {sequence_column, metric} - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            sequence_id = (row.get(sequence_column) or "").strip()
            if not sequence_id:
                raise ValueError(f"{path}:{row_number}: empty sequence ID")
            if sequence_id in values:
                raise ValueError(f"{path}:{row_number}: duplicate sequence ID {sequence_id}")
            try:
                values[sequence_id] = float(row[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: invalid {metric} value {row.get(metric)!r}") from exc
    if not values:
        raise ValueError(f"No rows found in {path}")
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compare_pair(
    baseline_name: str,
    candidate_name: str,
    baseline_values: dict[str, float],
    candidate_values: dict[str, float],
    *,
    metric: str,
    alpha: float,
    normality_test: str,
    allow_missing: bool,
) -> ComparisonResult:
    baseline_only = sorted(set(baseline_values) - set(candidate_values))
    candidate_only = sorted(set(candidate_values) - set(baseline_values))
    if (baseline_only or candidate_only) and not allow_missing:
        raise ValueError(
            f"Sequence mismatch for {baseline_name} vs {candidate_name}: "
            f"missing from candidate={baseline_only[:10]}, missing from baseline={candidate_only[:10]}"
        )
    sequence_ids = sorted(set(baseline_values) & set(candidate_values))
    if len(sequence_ids) < 2:
        raise ValueError(f"Need at least two paired sequences for {baseline_name} vs {candidate_name}")
    baseline = [baseline_values[sequence_id] for sequence_id in sequence_ids]
    candidate = [candidate_values[sequence_id] for sequence_id in sequence_ids]
    differences = [candidate_value - baseline_value for baseline_value, candidate_value in zip(baseline, candidate)]

    try:
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError("statistical_significance.py requires scipy. Install it with `pip install scipy`.") from exc

    normality_p: float | None = None
    normality_label = "none"
    if normality_test == "shapiro":
        normality_label = "Shapiro-Wilk"
        if len(differences) >= 3:
            _normality_stat, normality_p = stats.shapiro(differences)
            normality_p = float(normality_p)
        else:
            normality_p = None

    use_ttest = normality_test == "none" or normality_p is None or normality_p >= alpha
    if use_ttest:
        statistic, p_value = stats.ttest_rel(candidate, baseline)
        test_name = "paired t-test"
    else:
        if all(abs(value) < 1e-12 for value in differences):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(candidate, baseline, zero_method="wilcox", alternative="two-sided")
        test_name = "Wilcoxon signed-rank"

    return ComparisonResult(
        baseline=baseline_name,
        candidate=candidate_name,
        metric=metric,
        n_sequences=len(sequence_ids),
        mean_baseline=mean(baseline),
        mean_candidate=mean(candidate),
        mean_difference=mean(differences),
        normality_test=normality_label,
        normality_p=normality_p,
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
    )


def write_results(path: Path, rows: list[ComparisonResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline",
        "candidate",
        "metric",
        "n_sequences",
        "mean_baseline",
        "mean_candidate",
        "mean_difference",
        "normality_test",
        "normality_p",
        "test_name",
        "statistic",
        "p_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TEMPLATE_HEADER)


def main() -> int:
    args = parse_args()
    if args.write_template:
        write_template(args.write_template)
        print(f"wrote_template: {args.write_template}")
        return 0
    try:
        model_inputs = [parse_model(value) for value in args.model]
        if len(model_inputs) < 2:
            raise ValueError("Provide at least two --model NAME=CSV inputs.")
        model_values = {
            model.name: read_metric_csv(model.path, args.sequence_column, args.metric) for model in model_inputs
        }
        baseline = args.baseline or model_inputs[0].name
        if baseline not in model_values:
            raise ValueError(f"Unknown baseline model: {baseline}")
        pairs: list[tuple[str, str]]
        if args.pair:
            pairs = []
            for pair in args.pair:
                if "=" not in pair:
                    raise ValueError(f"--pair must use BASELINE=CANDIDATE syntax: {pair}")
                left, right = pair.split("=", 1)
                pairs.append((left.strip(), right.strip()))
        else:
            pairs = [(baseline, name) for name in model_values if name != baseline]
        rows = [
            compare_pair(
                left,
                right,
                model_values[left],
                model_values[right],
                metric=args.metric,
                alpha=args.alpha,
                normality_test=args.normality_test,
                allow_missing=args.allow_missing,
            )
            for left, right in pairs
        ]
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for row in rows:
        print(
            f"{row.candidate} vs {row.baseline}: n={row.n_sequences}, "
            f"mean_delta={row.mean_difference:.6f}, {row.test_name}, "
            f"statistic={row.statistic:.6f}, p={row.p_value:.6g}"
        )
    if args.output:
        write_results(args.output, rows)
        print(f"wrote_output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
