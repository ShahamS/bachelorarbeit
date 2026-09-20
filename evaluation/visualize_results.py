
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "plots"

DATASET_LABELS = {
    "Hospital_log.xes": "Hospital",
    "Sepsis Cases - Event Log.xes": "Sepsis",
    "Road_Traffic_Fine_Management_Process.xes": "Traffic",
    "artificial_log.xes": "Artificial",
}

METRIC_LABELS = {
    "generalization": "Generalisierung",
    "fitness": "Fitness",
    "fitness_generalization_gap": "Fitness-Generalization-Gap",
    "avg_event_time_ms": "Zeit pro Event (ms)",
    "elapsed_s": "Gesamtlaufzeit (s)",
    "remote_calls": "Remote Calls",
    "route_calls": "Route Calls",
    "matrix_entries": "Footprint-Eintraege",
    "places": "Places",
    "transitions": "Transitionen",
    "arcs": "Kanten",
    "score_drop": "Score-Drop",
    "manipulated_score": "Score (Manipulation)",
}

METHOD_ORDER = [
    "CheckMyFlow",
    "TBR alpha",
    "TBR heuristics",
    "TBR inductive",
]

METHOD_COLORS = {
    "CheckMyFlow": "#1f77b4",
    "TBR alpha": "#d62728",
    "TBR heuristics": "#2ca02c",
    "TBR inductive": "#9467bd",
}

METHOD_MARKERS = {
    "CheckMyFlow": "o",
    "TBR alpha": "s",
    "TBR heuristics": "^",
    "TBR inductive": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PDF plots from CheckMyFlow and Token Replay CSV results.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing evaluation result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PDF plots are written.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Optional explicit CSV files. If omitted, current result files are auto-discovered.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        choices=["pdf", "png", "svg"],
        help="Output formats for every plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _explicit_paths(args.inputs) if args.inputs else _discover_csvs(Path(args.results_dir))
    training = _load_training_results(csv_paths)
    robustness = _load_robustness_results(csv_paths)

    written = []
    if not training.empty:
        written.extend(_plot_training_results(training, output_dir, args.formats))
    if not robustness.empty:
        written.extend(_plot_robustness_results(robustness, output_dir, args.formats))

    if not written:
        print("No plots written. Check whether result CSV files exist.")
        return

    print(f"Wrote {len(written)} plot files:")
    for path in written:
        print(f"  {path}")


def _explicit_paths(inputs: list[str]) -> list[Path]:
    return [Path(path) for path in inputs]


def _discover_csvs(results_dir: Path) -> list[Path]:
    csv_paths = []
    for path in sorted(results_dir.rglob("*.csv")):
        parts = set(path.parts)
        if "alteErgebnisse" in parts:
            continue
        if path.name.startswith("old_"):
            continue
        if path.name in {"dataset_location_summary.csv", "dataset_location_values.csv"}:
            continue
        csv_paths.append(path)
    return csv_paths


def _load_training_results(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        if "robustness" in path.parts:
            continue
        data = _read_csv(path)
        if data.empty or "training_split" not in data.columns:
            continue
        data["source_file"] = path.name
        data["method_label"] = data.apply(_method_label, axis=1)
        data["dataset_label"] = data["dataset"].map(DATASET_LABELS).fillna(data["dataset"])
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return _numeric(pd.concat(frames, ignore_index=True))


def _load_robustness_results(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        if "robustness" not in path.parts:
            continue
        data = _read_csv(path)
        if data.empty or "manipulation_probability" not in data.columns:
            continue
        data["source_file"] = path.name
        data["method_label"] = data.apply(_method_label, axis=1)
        data["dataset_label"] = data["dataset"].map(DATASET_LABELS).fillna(data["dataset"])
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return _numeric(pd.concat(frames, ignore_index=True))


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _numeric(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        if column in {"dataset", "location_key", "method", "strategy", "discovery_algorithm",
                      "requested_discovery_algorithm", "discovery_parameters", "approach",
                      "scenario", "manipulation_type", "source_file", "method_label",
                      "dataset_label"}:
            continue
        try:
            data[column] = pd.to_numeric(data[column])
        except (TypeError, ValueError):
            pass
    return data


def _method_label(row: pd.Series) -> str:
    approach = str(row.get("approach", "")).strip().lower()
    method = str(row.get("method", "")).strip().lower()
    algorithm = str(row.get("discovery_algorithm", "")).strip().lower()
    source_file = str(row.get("source_file", "")).strip().lower()

    if approach == "checkmyflow" or "checkmyflow" in source_file:
        return "CheckMyFlow"
    if method == "token_replay" or "token_replay" in source_file:
        if algorithm and algorithm != "nan":
            return f"TBR {algorithm}"
        for candidate in ("alpha", "heuristics", "heuristic", "inductive"):
            if candidate in source_file:
                normalized = "heuristics" if candidate == "heuristic" else candidate
                return f"TBR {normalized}"
        return "TBR"
    return row.get("approach") or row.get("method") or source_file


def _plot_training_results(data: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    written = []
    written.extend(_plot_metric_grid(
        data,
        x="training_split",
        y="generalization",
        output_dir=output_dir,
        stem="training_generalization",
        formats=formats,
        title="Generalisierung nach Trainingsanteil",
    ))
    written.extend(_plot_metric_grid(
        data,
        x="training_split",
        y="fitness",
        output_dir=output_dir,
        stem="training_fitness",
        formats=formats,
        title="Fitness nach Trainingsanteil",
    ))
    written.extend(_plot_metric_grid(
        data,
        x="training_split",
        y="fitness_generalization_gap",
        output_dir=output_dir,
        stem="training_fitness_generalization_gap",
        formats=formats,
        title="Fitness-Generalization-Gap",
    ))
    written.extend(_plot_metric_grid(
        data,
        x="training_split",
        y="avg_event_time_ms",
        output_dir=output_dir,
        stem="training_avg_event_time_ms",
        formats=formats,
        title="Verarbeitungszeit pro Event",
    ))
    written.extend(_plot_metric_grid(
        data,
        x="training_split",
        y="remote_calls",
        output_dir=output_dir,
        stem="training_remote_calls",
        formats=formats,
        title="Remote Calls nach Trainingsanteil",
    ))
    written.extend(_plot_model_size(data, output_dir, formats))
    return written


def _plot_robustness_results(data: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    written = []
    filtered = data[data["scenario"].astype(str) != "baseline"].copy()
    if filtered.empty:
        filtered = data.copy()

    for manipulation_type, group in filtered.groupby("manipulation_type", dropna=False):
        safe_type = _safe_name(str(manipulation_type))
        written.extend(_plot_metric_bars(
            group,
            y="manipulated_score",
            output_dir=output_dir,
            stem=f"robustness_{safe_type}_score",
            formats=formats,
            title=f"Robustheit: Score bei {manipulation_type}",
        ))
        written.extend(_plot_metric_bars(
            group,
            y="score_drop",
            output_dir=output_dir,
            stem=f"robustness_{safe_type}_score_drop",
            formats=formats,
            title=f"Robustheit: Score-Drop bei {manipulation_type}",
        ))
    return written


def _plot_metric_grid(
    data: pd.DataFrame,
    x: str,
    y: str,
    output_dir: Path,
    stem: str,
    formats: list[str],
    title: str,
) -> list[Path]:
    if x not in data.columns or y not in data.columns:
        return []
    plot_data = data.dropna(subset=[x, y])
    if plot_data.empty:
        return []

    datasets = list(plot_data["dataset_label"].dropna().unique())
    fig, axes = _make_dataset_axes(datasets)
    for axis, dataset in zip(axes, datasets):
        dataset_data = plot_data[plot_data["dataset_label"] == dataset]
        for method in _ordered_methods(dataset_data):
            method_data = dataset_data[dataset_data["method_label"] == method].sort_values(x)
            if method_data.empty:
                continue
            series = method_data.groupby(x, as_index=False)[y].mean()
            axis.plot(
                series[x],
                series[y],
                label=method,
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method),
                linewidth=1.8,
            )
        axis.set_title(dataset)
        axis.set_xlabel(METRIC_LABELS.get(x, x))
        axis.set_ylabel(METRIC_LABELS.get(y, y))
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    _hide_unused_axes(axes, len(datasets))
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, output_dir, stem, formats)


def _plot_metric_bars(
    data: pd.DataFrame,
    y: str,
    output_dir: Path,
    stem: str,
    formats: list[str],
    title: str,
) -> list[Path]:
    if y not in data.columns:
        return []
    plot_data = data.dropna(subset=[y])
    if plot_data.empty:
        return []

    datasets = list(plot_data["dataset_label"].dropna().unique())
    fig, axes = _make_dataset_axes(datasets)
    for axis, dataset in zip(axes, datasets):
        dataset_data = plot_data[plot_data["dataset_label"] == dataset]
        methods = _ordered_methods(dataset_data)
        series = (
            dataset_data.groupby("method_label", as_index=False)[y]
            .mean()
            .set_index("method_label")
        )
        values = [series.loc[method, y] for method in methods if method in series.index]
        labels = [method for method in methods if method in series.index]
        colors = [METHOD_COLORS.get(method) for method in labels]

        axis.bar(labels, values, color=colors, width=0.72)
        axis.set_title(dataset)
        axis.set_xlabel("Algorithmus")
        axis.set_ylabel(METRIC_LABELS.get(y, y))
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=25)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    _hide_unused_axes(axes, len(datasets))
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, output_dir, stem, formats)


def _plot_model_size(data: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    rows = []
    for _, row in data.iterrows():
        method = row["method_label"]
        if method == "CheckMyFlow" and pd.notna(row.get("matrix_entries")):
            rows.append({
                "dataset_label": row["dataset_label"],
                "method_label": method,
                "training_split": row["training_split"],
                "model_size": row["matrix_entries"],
            })
        elif method != "CheckMyFlow" and pd.notna(row.get("arcs")):
            rows.append({
                "dataset_label": row["dataset_label"],
                "method_label": method,
                "training_split": row["training_split"],
                "model_size": row["arcs"],
            })
    if not rows:
        return []
    model_data = pd.DataFrame(rows)
    return _plot_metric_grid(
        model_data,
        x="training_split",
        y="model_size",
        output_dir=output_dir,
        stem="training_model_size",
        formats=formats,
        title="Modellgroesse nach Trainingsanteil",
    )


def _make_dataset_axes(datasets: list[str]):
    cols = min(2, max(1, len(datasets)))
    rows = (len(datasets) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.8 * rows), squeeze=False)
    return fig, list(axes.ravel())


def _hide_unused_axes(axes, used_count: int) -> None:
    for axis in axes[used_count:]:
        axis.set_visible(False)


def _ordered_methods(data: pd.DataFrame) -> list[str]:
    present = list(data["method_label"].dropna().unique())
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(method for method in present if method not in ordered)
    return ordered


def _save(fig, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def _safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


if __name__ == "__main__":
    main()
