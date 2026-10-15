from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "step61_figures_v1.yaml"
)

EXPECTED_FIGURE_CONTRACT_SHA256 = (
    "0ea1f1a945bbf3fd2fdcd171497eab064f0942235b08d3375db78551c6fd49d9"
)

STATISTICS_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "step60_statistics_v1.yaml"
)
EXPECTED_STATISTICS_CONTRACT_SHA256 = (
    "651ac8f6b78073796858cb369cb99b9b16f7f20ef53125374e3405d39316e1a7"
)

AGG_ROOT = (
    PROJECT_ROOT / "results" / "step60-aggregation-v1"
)

AGG_MANIFEST_PATH = (
    AGG_ROOT / "aggregation_manifest.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "step61-figures-v1"
)

CORE_ROOT = OUTPUT_ROOT / "core"
SUPP_ROOT = OUTPUT_ROOT / "supplementary"
DATA_ROOT = OUTPUT_ROOT / "figure_data"

RAW_RATINGS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml-1m"
    / "v1"
    / "ratings.csv"
)

TRAIN_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "ml-1m"
    / "v1"
    / "train.csv"
)

VALIDATION_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "ml-1m"
    / "v1"
    / "validation.csv"
)

TEST_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "ml-1m"
    / "v1"
    / "test.csv"
)

MODEL_ORDER = (
    "LogisticPCA",
    "BernoulliMF",
    "p-emb",
    "PoissonMF",
    "b-emb",
    "HPF",
    "PCA/SVD",
    "ap-emb",
    "Popularity",
    "GaussianMF",
    "ng-emb",
    "g-emb",
    "Random",
)

NORMALIZATION_ORDER = (
    "raw",
    "sqrt",
    "mean",
)

CORE_IDS = tuple(f"F{i:02d}" for i in range(1, 17))
SUPPLEMENT_IDS = tuple(f"S{i:02d}" for i in range(1, 13))
ALL_FIGURE_IDS = CORE_IDS + SUPPLEMENT_IDS

PNG_DPI = 600

RANKING_METRICS_K10 = (
    "precision_at_10_mean",
    "recall_at_10_mean",
    "hit_rate_at_10_mean",
    "ndcg_at_10_mean",
    "mrr_at_10_mean",
    "map_at_10_mean",
)

FIGURE_FILENAMES = {
    "F01": "fig01_primary_ndcg10_ranking",
    "F02": "fig02_primary_ndcg_by_k",
    "F03": "fig03_primary_recall_by_k",
    "F04": "fig04_primary_metric_heatmap_k10",
    "F05": "fig05_primary_vs_preference_ndcg10",
    "F06": "fig06_preference_user_ndcg10_distribution",
    "F07": "fig07_gaussian_stability_ndcg10",
    "F08": "fig08_gaussian_stability_rmse",
    "F09": "fig09_gaussian_context_norm_p99",
    "F10": "fig10_gaussian_max_gradient",
    "F11": "fig11_movielens_rating_distribution",
    "F12": "fig12_user_history_distribution",
    "F13": "fig13_user_history_ecdf",
    "F14": "fig14_split_composition",
    "F15": "fig15_preference_relevant_items_distribution",
    "F16": "fig16_test_rating_distribution_with_threshold",
    "S01": "supp01_primary_seed_ndcg10",
    "S02": "supp02_preference_seed_ndcg10",
    "S03": "supp03_rank_shift_primary_to_preference",
    "S04": "supp04_gaussian_native_rmse",
    "S05": "supp05_gaussian_native_mae",
    "S06": "supp06_gaussian_max_prediction",
    "S07": "supp07_gaussian_context_norm_max",
    "S08": "supp08_preference_candidate_count_distribution",
    "S09": "supp09_primary_precision_by_k",
    "S10": "supp10_primary_map_by_k",
    "S11": "supp11_primary_mrr_by_k",
    "S12": "supp12_primary_hit_rate_by_k",
}

FIGURE_TITLES = {
    "F01": "Primary full-catalog ranking performance",
    "F02": "Primary NDCG across ranking cutoffs",
    "F03": "Primary Recall across ranking cutoffs",
    "F04": "Primary metric profile at K=10",
    "F05": "Primary versus preference-aware NDCG@10",
    "F06": "User-level preference-aware NDCG@10",
    "F07": "Gaussian EF-EMB ranking stability by context normalization",
    "F08": "Gaussian EF-EMB RMSE by context normalization",
    "F09": "Gaussian EF-EMB context-norm scale by normalization",
    "F10": "Maximum pre-clipping gradient norm by normalization",
    "F11": "MovieLens 1M rating distribution",
    "F12": "MovieLens 1M user-history length distribution",
    "F13": "ECDF of MovieLens 1M user-history lengths",
    "F14": "Canonical MovieLens 1M split composition",
    "F15": "Relevant items per eligible user in preference sensitivity",
    "F16": "Test-set rating distribution and preference threshold",
    "S01": "Primary NDCG@10 by individual frozen seed",
    "S02": "Preference-aware NDCG@10 by individual frozen seed",
    "S03": "Model rank shift under preference-aware relevance",
    "S04": "Final-test Gaussian-family RMSE",
    "S05": "Final-test Gaussian-family MAE",
    "S06": "Maximum validation prediction magnitude by normalization",
    "S07": "Maximum training context norm by normalization",
    "S08": "Candidate-set size distribution in preference sensitivity",
    "S09": "Primary Precision across ranking cutoffs",
    "S10": "Primary MAP across ranking cutoffs",
    "S11": "Primary MRR across ranking cutoffs",
    "S12": "Primary Hit Rate across ranking cutoffs",
}

FIGURE_ROLES = {
    "F01": "principal_result",
    "F02": "cutoff_sensitivity",
    "F03": "retrieval_depth",
    "F04": "multi_metric_overview",
    "F05": "relevance_definition_sensitivity",
    "F06": "user_level_heterogeneity",
    "F07": "normalization_ranking_effect",
    "F08": "normalization_numeric_stability",
    "F09": "mechanism_context_scaling",
    "F10": "mechanism_optimization_stability",
    "F11": "dataset_description",
    "F12": "dataset_and_context_scale",
    "F13": "dataset_and_context_scale",
    "F14": "experimental_design",
    "F15": "sensitivity_population_description",
    "F16": "preference_sensitivity_definition",
    "S01": "seed_level_primary_detail",
    "S02": "seed_level_preference_detail",
    "S03": "rank_sensitivity_detail",
    "S04": "native_metric_detail",
    "S05": "native_metric_detail",
    "S06": "stability_diagnostic_detail",
    "S07": "stability_diagnostic_detail",
    "S08": "sensitivity_population_detail",
    "S09": "cutoff_metric_detail",
    "S10": "cutoff_metric_detail",
    "S11": "cutoff_metric_detail",
    "S12": "cutoff_metric_detail",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        isinstance(value, dict),
        f"YAML root is not a mapping: {path}",
    )

    return value


def validate_contract() -> dict[str, Any]:
    require(
        FIGURE_CONTRACT_PATH.is_file(),
        f"Missing figure contract: {FIGURE_CONTRACT_PATH}",
    )

    actual = sha256_file(
        FIGURE_CONTRACT_PATH
    )

    require(
        actual == EXPECTED_FIGURE_CONTRACT_SHA256,
        (
            "Step 61 figure-contract SHA mismatch. "
            f"Expected={EXPECTED_FIGURE_CONTRACT_SHA256}; actual={actual}"
        ),
    )

    contract = load_yaml(
        FIGURE_CONTRACT_PATH
    )

    require(
        contract.get("freeze_id")
        ==
        "paper2-step61-figures-v1",
        "Figure-contract freeze_id mismatch",
    )

    require(
        contract.get("status") == "frozen",
        "Figure contract is not frozen",
    )

    require(
        contract.get("purpose", {}).get(
            "target_journal_known"
        )
        is False,
        "Figure contract unexpectedly declares a target journal",
    )

    require(
        contract.get("purpose", {}).get(
            "training_permitted"
        )
        is False,
        "Figure contract unexpectedly allows training",
    )

    require(
        contract.get("statistical_display", {}).get(
            "confidence_intervals"
        )
        is False,
        "Figure contract unexpectedly enables confidence intervals",
    )

    require(
        contract.get("statistical_display", {}).get(
            "p_values"
        )
        is False,
        "Figure contract unexpectedly enables p-values",
    )

    require(
        len(contract.get("core_candidates", {}))
        ==
        len(CORE_IDS),
        "Figure contract does not contain exactly 16 core candidates",
    )

    require(
        len(contract.get("supplementary_candidates", {}))
        ==
        len(SUPPLEMENT_IDS),
        "Figure contract does not contain exactly 12 supplementary candidates",
    )

    for figure_id, filename in FIGURE_FILENAMES.items():
        section = (
            "core_candidates"
            if figure_id.startswith("F")
            else "supplementary_candidates"
        )

        spec = contract[section][figure_id]

        require(
            spec.get("filename") == filename,
            f"{figure_id}: filename mismatch in figure contract",
        )

    return contract


def validate_statistics_contract() -> None:
    require(
        STATISTICS_CONTRACT_PATH.is_file(),
        f"Missing statistics contract: {STATISTICS_CONTRACT_PATH}",
    )

    actual = sha256_file(
        STATISTICS_CONTRACT_PATH
    )

    require(
        actual == EXPECTED_STATISTICS_CONTRACT_SHA256,
        (
            "Step 60 statistics-contract SHA mismatch. "
            f"Expected={EXPECTED_STATISTICS_CONTRACT_SHA256}; actual={actual}"
        ),
    )


def validate_aggregation_manifest() -> dict[str, Any]:
    require(
        AGG_MANIFEST_PATH.is_file(),
        f"Missing aggregation manifest: {AGG_MANIFEST_PATH}",
    )

    manifest = json.loads(
        AGG_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        manifest.get("freeze_id")
        ==
        "paper2-step60-aggregation-v1",
        "Step 60 aggregation freeze_id mismatch",
    )

    require(
        manifest.get("statistics_contract_sha256")
        ==
        EXPECTED_STATISTICS_CONTRACT_SHA256,
        "Step 60 aggregation statistics-contract hash mismatch",
    )

    require(
        manifest.get(
            "statistical_definitions",
            {},
        ).get("inferential_tests")
        is False,
        "Step 60 aggregation unexpectedly enables inferential tests",
    )

    require(
        manifest.get(
            "statistical_definitions",
            {},
        ).get("confidence_intervals")
        is False,
        "Step 60 aggregation unexpectedly enables confidence intervals",
    )

    for name, metadata in manifest.get(
        "artifacts",
        {},
    ).items():
        path = AGG_ROOT / name

        require(
            path.is_file(),
            f"Missing Step 60 artifact: {path}",
        )

        require(
            sha256_file(path)
            ==
            metadata.get("sha256"),
            f"Step 60 artifact SHA mismatch: {name}",
        )

    return manifest


def validate_raw_data() -> None:
    for path in (
        RAW_RATINGS,
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TEST_SPLIT,
    ):
        require(
            path.is_file(),
            f"Missing frozen raw/split source: {path}",
        )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def model_order(
    frame: pd.DataFrame,
    rank_column: str | None = None,
) -> list[str]:
    if rank_column is not None and rank_column in frame.columns:
        ordered = (
            frame[
                ["model", rank_column]
            ]
            .drop_duplicates()
            .sort_values(
                [rank_column, "model"],
                kind="stable",
            )
        )
        return ordered["model"].astype(str).tolist()

    present = set(
        frame["model"].astype(str)
    )

    return [
        model
        for model in MODEL_ORDER
        if model in present
    ]


def save_plot_data(
    frame: pd.DataFrame,
    *,
    figure_id: str,
    output_root: Path,
) -> Path:
    path = (
        output_root
        / "figure_data"
        / f"{figure_id.lower()}_{FIGURE_FILENAMES[figure_id]}.csv"
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    return path


def save_figure(
    fig: plt.Figure,
    *,
    figure_id: str,
    output_root: Path,
) -> list[Path]:
    subdir = (
        "core"
        if figure_id.startswith("F")
        else "supplementary"
    )

    base = (
        output_root
        / subdir
        / FIGURE_FILENAMES[figure_id]
    )

    outputs: list[Path] = []

    for extension in ("pdf", "svg", "png"):
        path = base.with_suffix(f".{extension}")

        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
        }

        if extension == "png":
            kwargs["dpi"] = PNG_DPI

        fig.savefig(
            path,
            **kwargs,
        )

        outputs.append(path)

    plt.close(fig)

    return outputs


def add_errorbar_note(ax: plt.Axes) -> None:
    ax.text(
        1.0,
        -0.12,
        "Error bars: sample SD across 3 frozen seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )


def render_horizontal_point_errorbar(
    *,
    frame: pd.DataFrame,
    mean_column: str,
    sd_column: str,
    rank_column: str,
    xlabel: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    data = frame.copy()
    data = data.sort_values(
        [rank_column, "model"],
        kind="stable",
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    y = np.arange(len(data))

    ax.errorbar(
        data[mean_column],
        y,
        xerr=data[sd_column],
        fmt="o",
        capsize=3,
        linewidth=1.0,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        data["model"]
    )
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    add_errorbar_note(ax)

    return fig, data


def render_k_lines(
    *,
    frame: pd.DataFrame,
    metric_base: str,
    rank_column: str,
    ylabel: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    ordered_models = model_order(
        frame,
        rank_column,
    )

    ks = (5, 10, 20)

    rows: list[dict[str, Any]] = []

    fig, ax = plt.subplots(
        figsize=(8.4, 5.6)
    )

    for model in ordered_models:
        record = frame.loc[
            frame["model"] == model
        ].iloc[0]

        means = [
            float(
                record[
                    f"{metric_base}_at_{k}_mean"
                ]
            )
            for k in ks
        ]

        sds = [
            float(
                record[
                    f"{metric_base}_at_{k}_sd"
                ]
            )
            for k in ks
        ]

        ax.errorbar(
            ks,
            means,
            yerr=sds,
            marker="o",
            capsize=2.5,
            linewidth=1.0,
            label=model,
        )

        for k, mean, sd in zip(
            ks,
            means,
            sds,
            strict=True,
        ):
            rows.append(
                {
                    "model": model,
                    "k": k,
                    "mean": mean,
                    "sd": sd,
                }
            )

    ax.set_xticks(
        list(ks)
    )
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )
    add_errorbar_note(ax)

    return fig, pd.DataFrame(rows)


def render_heatmap(
    *,
    frame: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    ordered = frame.sort_values(
        ["primary_rank", "model"],
        kind="stable",
    )

    display_names = (
        "Precision",
        "Recall",
        "Hit Rate",
        "NDCG",
        "MRR",
        "MAP",
    )

    matrix = ordered[
        list(RANKING_METRICS_K10)
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 5.8)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        norm=Normalize(
            vmin=0.0,
            vmax=float(
                np.nanmax(matrix)
            ),
        ),
    )

    ax.grid(False)
    ax.set_xticks(
        np.arange(len(display_names))
    )
    ax.set_xticklabels(
        display_names
    )

    ax.set_yticks(
        np.arange(len(ordered))
    )
    ax.set_yticklabels(
        ordered["model"]
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.04,
        pad=0.03,
    )
    cbar.set_label(
        "Mean metric value"
    )

    plot_data = (
        ordered[
            ["model", "primary_rank"]
            +
            list(RANKING_METRICS_K10)
        ]
        .copy()
    )

    return fig, plot_data


def render_primary_vs_preference(
    frame: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    data = frame.sort_values(
        ["primary_rank", "model"],
        kind="stable",
    )

    fig, ax = plt.subplots(
        figsize=(7.6, 5.4)
    )

    y = np.arange(len(data))

    for index, row in enumerate(
        data.itertuples()
    ):
        ax.plot(
            [
                row.primary_ndcg_at_10_mean,
                row.preference_ndcg_at_10_mean,
            ],
            [index, index],
            linewidth=1.0,
            alpha=0.8,
        )

    ax.errorbar(
        data["primary_ndcg_at_10_mean"],
        y,
        xerr=data["primary_ndcg_at_10_sd"],
        fmt="o",
        capsize=3,
        label="Primary",
    )

    ax.errorbar(
        data["preference_ndcg_at_10_mean"],
        y,
        xerr=data["preference_ndcg_at_10_sd"],
        fmt="s",
        capsize=3,
        label="Preference-aware",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        data["model"]
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "NDCG@10"
    )
    ax.legend(
        frameon=False,
        loc="lower right",
    )
    add_errorbar_note(ax)

    return fig, data


def render_violin_box(
    frame: pd.DataFrame,
    preference_summary: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    order = model_order(
        preference_summary,
        "preference_rank",
    )

    values = [
        frame.loc[
            frame["model"] == model,
            "ndcg_at_10_mean",
        ].to_numpy(
            dtype=float
        )
        for model in order
    ]

    fig, ax = plt.subplots(
        figsize=(9.2, 5.8)
    )

    positions = np.arange(
        1,
        len(order) + 1,
    )

    ax.violinplot(
        values,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    ax.boxplot(
        values,
        positions=positions,
        widths=0.16,
        showfliers=False,
        patch_artist=False,
    )

    ax.set_xticks(
        positions
    )
    ax.set_xticklabels(
        order,
        rotation=45,
        ha="right",
    )
    ax.set_ylabel(
        "User-level mean NDCG@10"
    )
    ax.set_xlabel("")
    ax.set_ylim(
        -0.02,
        1.02,
    )

    plot_data = frame[
        [
            "model",
            "userId",
            "n_relevant",
            "n_candidates",
            "ndcg_at_10_mean",
            "ndcg_at_10_sd",
        ]
    ].copy()

    return fig, plot_data


def aggregate_diagnostic_metric(
    frame: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    grouped = frame.groupby(
        [
            "model_name",
            "context_normalization",
        ],
        sort=False,
    )

    for (model, normalization), group in grouped:
        values = pd.to_numeric(
            group[value_column],
            errors="raise",
        ).astype(float)

        rows.append(
            {
                "model": str(model),
                "normalization": str(normalization),
                "mean": float(
                    values.mean()
                ),
                "sd": float(
                    values.std(ddof=1)
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                ),
                "n_seeds": int(
                    len(values)
                ),
            }
        )

    return pd.DataFrame(rows)


def render_grouped_stability(
    *,
    frame: pd.DataFrame,
    value_column: str,
    sd_column: str,
    ylabel: str,
    log_y: bool,
) -> tuple[plt.Figure, pd.DataFrame]:
    data = frame.copy()

    fig, ax = plt.subplots(
        figsize=(6.8, 4.8)
    )

    x = np.arange(
        len(NORMALIZATION_ORDER)
    )

    offsets = {
        "g-emb": -0.08,
        "ng-emb": 0.08,
    }

    for model in (
        "g-emb",
        "ng-emb",
    ):
        subset = (
            data.loc[
                data["model"] == model
            ]
            .set_index("normalization")
            .loc[list(NORMALIZATION_ORDER)]
        )

        ax.errorbar(
            x + offsets[model],
            subset[value_column],
            yerr=subset[sd_column],
            fmt="o",
            capsize=3,
            linewidth=1.0,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        NORMALIZATION_ORDER
    )
    ax.set_xlabel(
        "Context normalization"
    )
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")

    ax.legend(
        frameon=False,
    )
    add_errorbar_note(ax)

    return fig, data


def render_grouped_diagnostic(
    *,
    diagnostic_frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    log_y: bool,
) -> tuple[plt.Figure, pd.DataFrame]:
    data = aggregate_diagnostic_metric(
        diagnostic_frame,
        value_column=value_column,
    )

    fig, ax = plt.subplots(
        figsize=(6.8, 4.8)
    )

    x = np.arange(
        len(NORMALIZATION_ORDER)
    )

    offsets = {
        "g-emb": -0.08,
        "ng-emb": 0.08,
    }

    for model in (
        "g-emb",
        "ng-emb",
    ):
        subset = (
            data.loc[
                data["model"] == model
            ]
            .set_index("normalization")
            .loc[list(NORMALIZATION_ORDER)]
        )

        ax.errorbar(
            x + offsets[model],
            subset["mean"],
            yerr=subset["sd"],
            fmt="o",
            capsize=3,
            linewidth=1.0,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        NORMALIZATION_ORDER
    )
    ax.set_xlabel(
        "Context normalization"
    )
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")

    ax.legend(
        frameon=False,
    )
    add_errorbar_note(ax)

    return fig, data


def render_bar_counts(
    *,
    labels: Sequence[Any],
    counts: Sequence[float],
    xlabel: str,
    ylabel: str = "Count",
    annotation: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=(6.4, 4.6)
    )

    x = np.arange(
        len(labels)
    )

    ax.bar(
        x,
        counts,
        width=0.72,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(value) for value in labels]
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if annotation is not None:
        ax.text(
            0.99,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    return fig


def render_histogram(
    values: np.ndarray,
    *,
    xlabel: str,
    log_x: bool = False,
    bins: int | Sequence[float] = 30,
) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=(6.6, 4.8)
    )

    if log_x:
        positive = values[
            values > 0
        ]

        bin_edges = np.logspace(
            np.log10(
                float(
                    positive.min()
                )
            ),
            np.log10(
                float(
                    positive.max()
                )
            ),
            35,
        )

        ax.hist(
            positive,
            bins=bin_edges,
        )

        ax.set_xscale(
            "log"
        )
    else:
        ax.hist(
            values,
            bins=bins,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "Users"
    )

    return fig


def render_ecdf(
    values: np.ndarray,
    *,
    xlabel: str,
    log_x: bool,
) -> tuple[plt.Figure, pd.DataFrame]:
    x = np.sort(
        values.astype(float)
    )

    y = np.arange(
        1,
        len(x) + 1,
        dtype=float,
    ) / len(x)

    fig, ax = plt.subplots(
        figsize=(6.6, 4.8)
    )

    ax.step(
        x,
        y,
        where="post",
    )

    if log_x:
        ax.set_xscale(
            "log"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "Empirical CDF"
    )
    ax.set_ylim(
        0.0,
        1.01,
    )

    data = pd.DataFrame(
        {
            "history_length": x,
            "ecdf": y,
        }
    )

    return fig, data


def render_seed_dotplot(
    *,
    seed_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    rank_column: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    order = model_order(
        summary_frame,
        rank_column,
    )

    y_map = {
        model: index
        for index, model in enumerate(order)
    }

    fig, ax = plt.subplots(
        figsize=(7.4, 5.3)
    )

    markers = {
        2026: "o",
        2027: "s",
        2028: "^",
    }

    for seed in (
        2026,
        2027,
        2028,
    ):
        subset = seed_frame.loc[
            seed_frame["seed"] == seed
        ]

        ax.scatter(
            subset["ndcg_at_10"],
            [
                y_map[model]
                for model in subset["model"]
            ],
            marker=markers[seed],
            label=str(seed),
            s=28,
        )

    ax.set_yticks(
        np.arange(len(order))
    )
    ax.set_yticklabels(
        order
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "NDCG@10"
    )
    ax.set_ylabel("")
    ax.legend(
        title="Seed",
        frameon=False,
    )

    return fig, seed_frame.copy()


def render_rank_slopegraph(
    frame: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    data = frame.copy()

    fig, ax = plt.subplots(
        figsize=(7.2, 6.0)
    )

    left_x = 0.0
    right_x = 1.0

    for row in data.itertuples():
        y1 = int(
            row.primary_rank
        )
        y2 = int(
            row.preference_rank
        )

        ax.plot(
            [left_x, right_x],
            [y1, y2],
            marker="o",
            linewidth=1.0,
        )

        ax.text(
            left_x - 0.03,
            y1,
            row.model,
            ha="right",
            va="center",
            fontsize=8,
        )

        ax.text(
            right_x + 0.03,
            y2,
            row.model,
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(
        -0.42,
        1.42,
    )
    ax.set_ylim(
        13.7,
        0.3,
    )
    ax.set_xticks(
        [left_x, right_x]
    )
    ax.set_xticklabels(
        ["Primary", "Preference-aware"]
    )
    ax.set_ylabel(
        "Rank (1 = best)"
    )
    ax.grid(
        axis="x",
        visible=False,
    )

    return fig, data


def render_native_point_errorbar(
    *,
    native_summary: pd.DataFrame,
    metric: str,
    xlabel: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    models = (
        "GaussianMF",
        "g-emb",
        "ng-emb",
    )

    data = native_summary.loc[
        (
            native_summary["metric"]
            ==
            metric
        )
        &
        (
            native_summary["model"].isin(
                models
            )
        )
    ].copy()

    data["_order"] = data["model"].map(
        {
            model: index
            for index, model in enumerate(
                models
            )
        }
    )

    data.sort_values(
        "_order",
        inplace=True,
        kind="stable",
    )
    data.drop(
        columns="_order",
        inplace=True,
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 3.8)
    )

    y = np.arange(
        len(data)
    )

    ax.errorbar(
        data["mean"],
        y,
        xerr=data["sd"],
        fmt="o",
        capsize=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        data["model"]
    )
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    add_errorbar_note(ax)

    return fig, data


def make_captions() -> dict[str, str]:
    return {
        "F01": (
            "Primary full-catalog NDCG@10 on the frozen MovieLens 1M test set. "
            "Points show the mean across seeds 2026, 2027, and 2028; error bars show "
            "the sample standard deviation across those three frozen seeds."
        ),
        "F02": (
            "Primary NDCG at K = 5, 10, and 20 under the frozen full-catalog test "
            "evaluation. Error bars show sample standard deviation across three frozen seeds."
        ),
        "F03": (
            "Primary Recall at K = 5, 10, and 20 under the frozen full-catalog test "
            "evaluation. Error bars show sample standard deviation across three frozen seeds."
        ),
        "F04": (
            "Primary mean ranking-metric profile at K = 10 for all thirteen models. "
            "Values are descriptive means across the three frozen seeds."
        ),
        "F05": (
            "NDCG@10 under the primary all-observed relevance definition and the separate "
            "rating >= 4 preference-aware sensitivity analysis. Error bars show sample "
            "standard deviation across three frozen seeds. The two evaluations use different "
            "relevance definitions and are compared descriptively."
        ),
        "F06": (
            "Distribution across 5,851 eligible users of per-user preference-aware mean "
            "NDCG@10, where rating >= 4 defines relevance. The sensitivity analysis is "
            "secondary and does not replace the primary results."
        ),
        "F07": (
            "Validation NDCG@10 for g-emb and ng-emb across raw, square-root, and mean "
            "context normalization in the prespecified Gaussian EF-EMB stability ablation. "
            "Error bars show sample standard deviation across three frozen seeds; no test "
            "metrics are used in this ablation."
        ),
        "F08": (
            "Validation RMSE for g-emb and ng-emb across context-normalization choices in "
            "the prespecified Gaussian EF-EMB stability ablation. The y-axis is logarithmic. "
            "Error bars show sample standard deviation across three frozen seeds."
        ),
        "F09": (
            "99th percentile of positive-training context-vector norm across context "
            "normalizations for g-emb and ng-emb. Points summarize the three stability seeds; "
            "the y-axis is logarithmic."
        ),
        "F10": (
            "Maximum pre-clipping gradient norm observed over training epochs in the "
            "Gaussian EF-EMB stability ablation. Points summarize the three stability seeds; "
            "the y-axis is logarithmic."
        ),
        "F11": (
            "Distribution of explicit ratings in the frozen MovieLens 1M dataset used for "
            "the publication benchmark."
        ),
        "F12": (
            "Distribution of MovieLens 1M interaction-history length by user. The x-axis is "
            "logarithmic to expose the heavy upper tail."
        ),
        "F13": (
            "Empirical cumulative distribution of MovieLens 1M user interaction-history "
            "length. The x-axis is logarithmic."
        ),
        "F14": (
            "Interaction counts in the frozen canonical MovieLens 1M train, validation, and "
            "test partitions."
        ),
        "F15": (
            "Distribution of relevant test items per eligible user in the rating >= 4 "
            "preference-aware sensitivity analysis."
        ),
        "F16": (
            "Rating distribution in the frozen MovieLens 1M test partition. Ratings 4 and 5 "
            "define relevant items in the secondary preference-aware sensitivity analysis."
        ),
        "S01": (
            "Primary test NDCG@10 for each individual frozen seed, shown without additional "
            "statistical inference."
        ),
        "S02": (
            "Preference-aware test NDCG@10 for each individual frozen seed in the secondary "
            "rating >= 4 sensitivity analysis."
        ),
        "S03": (
            "Descriptive model-rank shift from the primary all-observed evaluation to the "
            "separate preference-aware sensitivity analysis."
        ),
        "S04": (
            "Final-test RMSE for GaussianMF, g-emb, and ng-emb. Points show means across "
            "three frozen seeds and error bars show sample standard deviation."
        ),
        "S05": (
            "Final-test MAE for GaussianMF, g-emb, and ng-emb. Points show means across "
            "three frozen seeds and error bars show sample standard deviation."
        ),
        "S06": (
            "Maximum absolute validation prediction magnitude across context normalizations "
            "in the Gaussian EF-EMB stability ablation. The y-axis is logarithmic."
        ),
        "S07": (
            "Maximum positive-training context-vector norm across context normalizations in "
            "the Gaussian EF-EMB stability ablation. The y-axis is logarithmic."
        ),
        "S08": (
            "Distribution of legal full-catalog candidate-set sizes across eligible users in "
            "the preference-aware sensitivity analysis."
        ),
        "S09": (
            "Primary Precision at K = 5, 10, and 20. Error bars show sample standard "
            "deviation across three frozen seeds."
        ),
        "S10": (
            "Primary MAP at K = 5, 10, and 20. Error bars show sample standard deviation "
            "across three frozen seeds."
        ),
        "S11": (
            "Primary MRR at K = 5, 10, and 20. Error bars show sample standard deviation "
            "across three frozen seeds."
        ),
        "S12": (
            "Primary Hit Rate at K = 5, 10, and 20. Error bars show sample standard "
            "deviation across three frozen seeds."
        ),
    }


def generate_all_figures(
    *,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    primary_summary = pd.read_csv(
        AGG_ROOT / "primary_summary.csv"
    )

    primary_seed = pd.read_csv(
        AGG_ROOT / "primary_seed_metrics.csv"
    )

    native_summary = pd.read_csv(
        AGG_ROOT / "primary_native_summary.csv"
    )

    preference_summary = pd.read_csv(
        AGG_ROOT / "preference_summary.csv"
    )

    preference_seed = pd.read_csv(
        AGG_ROOT / "preference_seed_metrics.csv"
    )

    preference_user = pd.read_csv(
        AGG_ROOT / "preference_user_model_summary.csv"
    )

    comparison = pd.read_csv(
        AGG_ROOT / "primary_vs_preference_ndcg10.csv"
    )

    rank_summary = pd.read_csv(
        AGG_ROOT / "model_rank_summary.csv"
    )

    stability_summary = pd.read_csv(
        AGG_ROOT / "stability_summary.csv"
    )

    diagnostics = pd.read_csv(
        AGG_ROOT / "stability_diagnostics_flat.csv"
    )

    ratings = pd.read_csv(
        RAW_RATINGS
    )

    test = pd.read_csv(
        TEST_SPLIT
    )

    figure_records: list[dict[str, Any]] = []
    captions = make_captions()

    # F01
    fig, data = render_horizontal_point_errorbar(
        frame=primary_summary,
        mean_column="ndcg_at_10_mean",
        sd_column="ndcg_at_10_sd",
        rank_column="primary_rank",
        xlabel="NDCG@10",
    )
    figure_records.append(
        finalize_figure(
            figure_id="F01",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F02
    fig, data = render_k_lines(
        frame=primary_summary,
        metric_base="ndcg",
        rank_column="primary_rank",
        ylabel="NDCG",
    )
    figure_records.append(
        finalize_figure(
            figure_id="F02",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F03
    fig, data = render_k_lines(
        frame=primary_summary,
        metric_base="recall",
        rank_column="primary_rank",
        ylabel="Recall",
    )
    figure_records.append(
        finalize_figure(
            figure_id="F03",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F04
    fig, data = render_heatmap(
        frame=primary_summary
    )
    figure_records.append(
        finalize_figure(
            figure_id="F04",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F05
    fig, data = render_primary_vs_preference(
        comparison
    )
    figure_records.append(
        finalize_figure(
            figure_id="F05",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F06
    fig, data = render_violin_box(
        preference_user,
        preference_summary,
    )
    figure_records.append(
        finalize_figure(
            figure_id="F06",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F07
    fig, data = render_grouped_stability(
        frame=stability_summary,
        value_column="ndcg_at_10_mean",
        sd_column="ndcg_at_10_sd",
        ylabel="Validation NDCG@10",
        log_y=False,
    )
    figure_records.append(
        finalize_figure(
            figure_id="F07",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F08
    fig, data = render_grouped_stability(
        frame=stability_summary,
        value_column="rmse_mean",
        sd_column="rmse_sd",
        ylabel="Validation RMSE",
        log_y=True,
    )
    figure_records.append(
        finalize_figure(
            figure_id="F08",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F09
    fig, data = render_grouped_diagnostic(
        diagnostic_frame=diagnostics,
        value_column="positive_training_context_norm.p99",
        ylabel="P99 context-vector norm",
        log_y=True,
    )
    figure_records.append(
        finalize_figure(
            figure_id="F09",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F10
    fig, data = render_grouped_diagnostic(
        diagnostic_frame=diagnostics,
        value_column=(
            "training_history_extrema."
            "max_max_gradient_norm_before_clipping_over_epochs"
        ),
        ylabel="Max pre-clipping gradient norm",
        log_y=True,
    )
    figure_records.append(
        finalize_figure(
            figure_id="F10",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F11
    rating_counts = (
        ratings["rating"]
        .value_counts()
        .sort_index()
    )

    fig = render_bar_counts(
        labels=rating_counts.index.tolist(),
        counts=rating_counts.values.tolist(),
        xlabel="Rating",
        ylabel="Interactions",
    )

    data = (
        rating_counts
        .rename("count")
        .reset_index()
        .rename(
            columns={"index": "rating"}
        )
    )

    figure_records.append(
        finalize_figure(
            figure_id="F11",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # F12
    history = (
        ratings.groupby("userId")
        .size()
        .rename("history_length")
        .reset_index()
    )

    fig = render_histogram(
        history["history_length"].to_numpy(
            dtype=float
        ),
        xlabel="Interactions per user",
        log_x=True,
    )

    figure_records.append(
        finalize_figure(
            figure_id="F12",
            fig=fig,
            data=history,
            output_root=output_root,
        )
    )

    # F13
    fig, ecdf_data = render_ecdf(
        history["history_length"].to_numpy(
            dtype=float
        ),
        xlabel="Interactions per user",
        log_x=True,
    )

    figure_records.append(
        finalize_figure(
            figure_id="F13",
            fig=fig,
            data=ecdf_data,
            output_root=output_root,
        )
    )

    # F14
    split_counts = pd.DataFrame(
        {
            "split": [
                "train",
                "validation",
                "test",
            ],
            "interactions": [
                sum(
                    1
                    for _ in open(
                        TRAIN_SPLIT,
                        "r",
                        encoding="utf-8",
                    )
                ) - 1,
                sum(
                    1
                    for _ in open(
                        VALIDATION_SPLIT,
                        "r",
                        encoding="utf-8",
                    )
                ) - 1,
                sum(
                    1
                    for _ in open(
                        TEST_SPLIT,
                        "r",
                        encoding="utf-8",
                    )
                ) - 1,
            ],
        }
    )

    fig = render_bar_counts(
        labels=split_counts["split"],
        counts=split_counts["interactions"],
        xlabel="Split",
        ylabel="Interactions",
    )

    figure_records.append(
        finalize_figure(
            figure_id="F14",
            fig=fig,
            data=split_counts,
            output_root=output_root,
        )
    )

    # F15 / S08 deduplicate preference user metadata.
    preference_users = (
        preference_user[
            [
                "userId",
                "n_relevant",
                "n_candidates",
            ]
        ]
        .drop_duplicates()
        .sort_values("userId")
        .reset_index(drop=True)
    )

    require(
        len(preference_users) == 5851,
        (
            "Preference user metadata does not contain exactly "
            "5,851 unique eligible users"
        ),
    )

    fig = render_histogram(
        preference_users[
            "n_relevant"
        ].to_numpy(
            dtype=float
        ),
        xlabel="Relevant test items per eligible user",
        bins=np.arange(
            0.5,
            float(
                preference_users[
                    "n_relevant"
                ].max()
            ) + 1.5,
            1.0,
        ),
    )

    figure_records.append(
        finalize_figure(
            figure_id="F15",
            fig=fig,
            data=preference_users[
                [
                    "userId",
                    "n_relevant",
                ]
            ],
            output_root=output_root,
        )
    )

    # F16
    test_rating_counts = (
        test["rating"]
        .value_counts()
        .sort_index()
    )

    fig = render_bar_counts(
        labels=test_rating_counts.index.tolist(),
        counts=test_rating_counts.values.tolist(),
        xlabel="Test rating",
        ylabel="Interactions",
        annotation="Preference relevance: rating >= 4",
    )

    data = (
        test_rating_counts
        .rename("count")
        .reset_index()
        .rename(
            columns={"index": "rating"}
        )
    )
    data["preference_relevant"] = (
        data["rating"] >= 4
    )

    figure_records.append(
        finalize_figure(
            figure_id="F16",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S01
    fig, data = render_seed_dotplot(
        seed_frame=primary_seed,
        summary_frame=primary_summary,
        rank_column="primary_rank",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S01",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S02
    fig, data = render_seed_dotplot(
        seed_frame=preference_seed,
        summary_frame=preference_summary,
        rank_column="preference_rank",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S02",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S03
    fig, data = render_rank_slopegraph(
        rank_summary
    )
    figure_records.append(
        finalize_figure(
            figure_id="S03",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S04
    fig, data = render_native_point_errorbar(
        native_summary=native_summary,
        metric="rmse",
        xlabel="RMSE",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S04",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S05
    fig, data = render_native_point_errorbar(
        native_summary=native_summary,
        metric="mae",
        xlabel="MAE",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S05",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S06
    fig, data = render_grouped_diagnostic(
        diagnostic_frame=diagnostics,
        value_column="validation_max_abs_prediction",
        ylabel="Max |validation prediction|",
        log_y=True,
    )
    figure_records.append(
        finalize_figure(
            figure_id="S06",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S07
    fig, data = render_grouped_diagnostic(
        diagnostic_frame=diagnostics,
        value_column="positive_training_context_norm.max",
        ylabel="Max context-vector norm",
        log_y=True,
    )
    figure_records.append(
        finalize_figure(
            figure_id="S07",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S08
    fig = render_histogram(
        preference_users[
            "n_candidates"
        ].to_numpy(
            dtype=float
        ),
        xlabel="Candidate-set size",
        bins=30,
    )

    figure_records.append(
        finalize_figure(
            figure_id="S08",
            fig=fig,
            data=preference_users[
                [
                    "userId",
                    "n_candidates",
                ]
            ],
            output_root=output_root,
        )
    )

    # S09
    fig, data = render_k_lines(
        frame=primary_summary,
        metric_base="precision",
        rank_column="primary_rank",
        ylabel="Precision",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S09",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S10
    fig, data = render_k_lines(
        frame=primary_summary,
        metric_base="map",
        rank_column="primary_rank",
        ylabel="MAP",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S10",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S11
    fig, data = render_k_lines(
        frame=primary_summary,
        metric_base="mrr",
        rank_column="primary_rank",
        ylabel="MRR",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S11",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    # S12
    fig, data = render_k_lines(
        frame=primary_summary,
        metric_base="hit_rate",
        rank_column="primary_rank",
        ylabel="Hit Rate",
    )
    figure_records.append(
        finalize_figure(
            figure_id="S12",
            fig=fig,
            data=data,
            output_root=output_root,
        )
    )

    return figure_records, captions


def finalize_figure(
    *,
    figure_id: str,
    fig: plt.Figure,
    data: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    require(
        figure_id in ALL_FIGURE_IDS,
        f"Unknown figure ID: {figure_id}",
    )

    image_paths = save_figure(
        fig,
        figure_id=figure_id,
        output_root=output_root,
    )

    data_path = save_plot_data(
        data,
        figure_id=figure_id,
        output_root=output_root,
    )

    return {
        "figure_id": figure_id,
        "category": (
            "core"
            if figure_id.startswith("F")
            else "supplementary"
        ),
        "filename": FIGURE_FILENAMES[
            figure_id
        ],
        "title": FIGURE_TITLES[
            figure_id
        ],
        "role": FIGURE_ROLES[
            figure_id
        ],
        "data_csv": str(
            data_path.relative_to(
                output_root
            )
        ).replace("\\", "/"),
        "pdf": str(
            image_paths[0].relative_to(
                output_root
            )
        ).replace("\\", "/"),
        "svg": str(
            image_paths[1].relative_to(
                output_root
            )
        ).replace("\\", "/"),
        "png": str(
            image_paths[2].relative_to(
                output_root
            )
        ).replace("\\", "/"),
    }


def write_caption_file(
    *,
    captions: Mapping[str, str],
    output_root: Path,
) -> Path:
    lines = [
        "# Paper 2 — Step 61 Figure Captions",
        "",
        "These captions are factual draft captions generated from the frozen "
        "Step 61 figure contract. Final journal-specific wording may be "
        "shortened later without changing the underlying results.",
        "",
    ]

    for figure_id in ALL_FIGURE_IDS:
        lines.extend(
            [
                (
                    f"## {figure_id} — "
                    f"{FIGURE_TITLES[figure_id]}"
                ),
                "",
                captions[figure_id],
                "",
            ]
        )

    path = (
        output_root
        / "figure_captions.md"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


def write_figure_index(
    *,
    records: Sequence[Mapping[str, Any]],
    captions: Mapping[str, str],
    output_root: Path,
) -> Path:
    rows: list[dict[str, Any]] = []

    for record in records:
        row = dict(record)
        row["caption"] = captions[
            str(record["figure_id"])
        ]
        rows.append(row)

    frame = pd.DataFrame(rows)

    path = (
        output_root
        / "figure_index.csv"
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    return path


def output_file_inventory(
    output_root: Path,
) -> list[Path]:
    return sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.name != "figure_manifest.json"
    )


def build_manifest(
    *,
    contract: Mapping[str, Any],
    aggregation_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}

    for path in output_file_inventory(
        output_root
    ):
        relative = str(
            path.relative_to(
                output_root
            )
        ).replace("\\", "/")

        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": int(
                path.stat().st_size
            ),
        }

    raw_source_hashes = {
        str(
            path.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"): sha256_file(path)
        for path in (
            RAW_RATINGS,
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
            TEST_SPLIT,
        )
    }

    return {
        "schema_version": "1.0",
        "freeze_id": "paper2-step61-figures-v1",
        "figure_contract_sha256": (
            EXPECTED_FIGURE_CONTRACT_SHA256
        ),
        "statistics_contract_sha256": (
            EXPECTED_STATISTICS_CONTRACT_SHA256
        ),
        "figure_generator_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "aggregation_freeze_id": aggregation_manifest[
            "freeze_id"
        ],
        "aggregation_manifest_sha256": sha256_file(
            AGG_MANIFEST_PATH
        ),
        "target_journal_known": False,
        "figure_policy": (
            "comprehensive publication-grade library; "
            "final inclusion decided later"
        ),
        "core_figure_count": len(CORE_IDS),
        "supplementary_figure_count": len(
            SUPPLEMENT_IDS
        ),
        "total_figure_count": len(
            ALL_FIGURE_IDS
        ),
        "image_formats_per_figure": [
            "pdf",
            "svg",
            "png",
        ],
        "png_dpi": PNG_DPI,
        "plot_data_csv_per_figure": True,
        "statistical_display": {
            "central_tendency": "mean",
            "uncertainty": (
                "sample standard deviation across "
                "three frozen seeds"
            ),
            "confidence_intervals": False,
            "p_values": False,
            "significance_markers": False,
        },
        "source_raw_file_hashes": (
            raw_source_hashes
        ),
        "figure_records": list(records),
        "files": files,
    }


def execute_generation(
    *,
    contract: Mapping[str, Any],
    aggregation_manifest: Mapping[str, Any],
) -> Path:
    require(
        not OUTPUT_ROOT.exists(),
        (
            f"Output directory already exists: {OUTPUT_ROOT}. "
            "Refusing to overwrite frozen figure artifacts."
        ),
    )

    staging = (
        OUTPUT_ROOT.parent
        / f".{OUTPUT_ROOT.name}.tmp"
    )

    require(
        not staging.exists(),
        (
            f"Staging directory already exists: {staging}. "
            "Inspect it before any manual cleanup."
        ),
    )

    try:
        staging.mkdir(
            parents=False,
            exist_ok=False,
        )

        (staging / "core").mkdir()
        (staging / "supplementary").mkdir()
        (staging / "figure_data").mkdir()

        configure_matplotlib()

        records, captions = (
            generate_all_figures(
                output_root=staging,
            )
        )

        require(
            len(records) == 28,
            (
                f"Expected 28 generated figure records, "
                f"found {len(records)}"
            ),
        )

        require(
            {
                str(record["figure_id"])
                for record in records
            }
            ==
            set(ALL_FIGURE_IDS),
            "Generated figure-ID inventory mismatch",
        )

        write_caption_file(
            captions=captions,
            output_root=staging,
        )

        write_figure_index(
            records=records,
            captions=captions,
            output_root=staging,
        )

        manifest = build_manifest(
            contract=contract,
            aggregation_manifest=(
                aggregation_manifest
            ),
            records=records,
            output_root=staging,
        )

        (
            staging
            / "figure_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        staging.replace(
            OUTPUT_ROOT
        )

    except Exception:
        if staging.exists():
            shutil.rmtree(staging)

        raise

    return OUTPUT_ROOT


def print_preflight(
    *,
    contract: Mapping[str, Any],
    aggregation_manifest: Mapping[str, Any],
) -> None:
    print("=" * 100)
    print("STEP 61C FIGURE GENERATION PREFLIGHT: PASS")
    print("=" * 100)
    print(
        "figure_contract_sha256:",
        EXPECTED_FIGURE_CONTRACT_SHA256,
    )
    print(
        "statistics_contract_sha256:",
        EXPECTED_STATISTICS_CONTRACT_SHA256,
    )
    print(
        "aggregation freeze:",
        aggregation_manifest["freeze_id"],
    )
    print(
        "core figure specifications:",
        len(CORE_IDS),
    )
    print(
        "supplementary figure specifications:",
        len(SUPPLEMENT_IDS),
    )
    print(
        "total figure specifications:",
        len(ALL_FIGURE_IDS),
    )
    print(
        "formats per figure:",
        "pdf, svg, png",
    )
    print(
        "plot-data CSV per figure:",
        "YES",
    )
    print(
        "target journal fixed:",
        "NO",
    )
    print()
    print("NO MODEL WAS FIT.")
    print("NO SCORE WAS RECOMPUTED.")
    print("NO STATISTICAL TEST WAS ADDED.")
    print("NO STEP 60 ARTIFACT WAS MODIFIED.")
    print("=" * 100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen Step 61 publication-figure "
            "library from Step 60 aggregate artifacts."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Write Step 61 figure artifacts. Without this "
            "flag, perform read-only preflight only."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    contract = validate_contract()
    validate_statistics_contract()
    aggregation_manifest = (
        validate_aggregation_manifest()
    )
    validate_raw_data()

    print_preflight(
        contract=contract,
        aggregation_manifest=(
            aggregation_manifest
        ),
    )

    if not args.execute:
        return 0

    output_root = execute_generation(
        contract=contract,
        aggregation_manifest=(
            aggregation_manifest
        ),
    )

    manifest = json.loads(
        (
            output_root
            / "figure_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    print()
    print("=" * 100)
    print("STEP 61C FIGURE GENERATION COMPLETE")
    print("=" * 100)
    print(
        "Output directory:",
        output_root,
    )
    print(
        "Core figures:",
        manifest["core_figure_count"],
    )
    print(
        "Supplementary figures:",
        manifest[
            "supplementary_figure_count"
        ],
    )
    print(
        "Total figures:",
        manifest[
            "total_figure_count"
        ],
    )
    print(
        "Image files:",
        manifest[
            "total_figure_count"
        ] * 3,
    )
    print(
        "Figure-data CSVs:",
        manifest[
            "total_figure_count"
        ],
    )
    print(
        "Manifest-tracked non-manifest files:",
        len(
            manifest[
                "files"
            ]
        ),
    )
    print()
    print("STEP 61C COMPLETE")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
