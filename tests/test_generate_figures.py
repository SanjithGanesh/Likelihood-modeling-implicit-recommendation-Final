from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from scripts import generate_figures as gf


def test_figure_contract_hash_is_frozen():
    assert (
        gf.EXPECTED_FIGURE_CONTRACT_SHA256
        ==
        "0ea1f1a945bbf3fd2fdcd171497eab064f0942235b08d3375db78551c6fd49d9"
    )


def test_figure_inventory_is_exact():
    assert len(gf.CORE_IDS) == 16
    assert len(gf.SUPPLEMENT_IDS) == 12
    assert len(gf.ALL_FIGURE_IDS) == 28

    assert set(gf.FIGURE_FILENAMES) == set(
        gf.ALL_FIGURE_IDS
    )

    assert set(gf.FIGURE_TITLES) == set(
        gf.ALL_FIGURE_IDS
    )

    assert set(gf.FIGURE_ROLES) == set(
        gf.ALL_FIGURE_IDS
    )


def test_model_order_uses_rank_column():
    frame = pd.DataFrame(
        {
            "model": ["B", "A", "C"],
            "primary_rank": [2, 1, 3],
        }
    )

    assert gf.model_order(
        frame,
        "primary_rank",
    ) == ["A", "B", "C"]


def test_aggregate_diagnostic_metric_uses_sample_sd():
    frame = pd.DataFrame(
        {
            "model_name": [
                "g-emb",
                "g-emb",
                "g-emb",
            ],
            "context_normalization": [
                "raw",
                "raw",
                "raw",
            ],
            "metric": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    result = gf.aggregate_diagnostic_metric(
        frame,
        value_column="metric",
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["mean"] == pytest.approx(2.0)
    assert row["sd"] == pytest.approx(1.0)
    assert row["min"] == pytest.approx(1.0)
    assert row["max"] == pytest.approx(3.0)
    assert row["n_seeds"] == 3


def test_ecdf_is_monotone():
    fig, data = gf.render_ecdf(
        np.array(
            [5, 1, 3, 3],
            dtype=float,
        ),
        xlabel="x",
        log_x=False,
    )

    try:
        assert data[
            "history_length"
        ].tolist() == [
            1.0,
            3.0,
            3.0,
            5.0,
        ]

        assert np.all(
            np.diff(
                data["ecdf"]
            ) >= 0
        )

        assert data[
            "ecdf"
        ].iloc[-1] == pytest.approx(1.0)

    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_primary_vs_preference_plot_preserves_rank_order():
    frame = pd.DataFrame(
        {
            "model": [
                "B",
                "A",
            ],
            "primary_rank": [
                2,
                1,
            ],
            "preference_rank": [
                1,
                2,
            ],
            "primary_ndcg_at_10_mean": [
                0.2,
                0.3,
            ],
            "primary_ndcg_at_10_sd": [
                0.01,
                0.02,
            ],
            "preference_ndcg_at_10_mean": [
                0.25,
                0.21,
            ],
            "preference_ndcg_at_10_sd": [
                0.02,
                0.01,
            ],
            "absolute_difference": [
                0.05,
                -0.09,
            ],
        }
    )

    fig, data = (
        gf.render_primary_vs_preference(
            frame
        )
    )

    try:
        assert data["model"].tolist() == [
            "A",
            "B",
        ]

    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_k_line_plot_emits_expected_data_rows():
    frame = pd.DataFrame(
        {
            "model": [
                "A",
                "B",
            ],
            "primary_rank": [
                1,
                2,
            ],
            "ndcg_at_5_mean": [
                0.5,
                0.4,
            ],
            "ndcg_at_5_sd": [
                0.01,
                0.02,
            ],
            "ndcg_at_10_mean": [
                0.45,
                0.35,
            ],
            "ndcg_at_10_sd": [
                0.01,
                0.02,
            ],
            "ndcg_at_20_mean": [
                0.40,
                0.30,
            ],
            "ndcg_at_20_sd": [
                0.01,
                0.02,
            ],
        }
    )

    fig, data = gf.render_k_lines(
        frame=frame,
        metric_base="ndcg",
        rank_column="primary_rank",
        ylabel="NDCG",
    )

    try:
        assert len(data) == 6

        assert set(
            data["k"]
        ) == {
            5,
            10,
            20,
        }

        assert set(
            data["model"]
        ) == {
            "A",
            "B",
        }

    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_caption_inventory_is_exact():
    captions = gf.make_captions()

    assert set(captions) == set(
        gf.ALL_FIGURE_IDS
    )

    for caption in captions.values():
        assert isinstance(
            caption,
            str,
        )
        assert len(
            caption.strip()
        ) > 20


def test_render_histogram_log_x_accepts_positive_values():
    fig = gf.render_histogram(
        np.array(
            [1, 2, 5, 10, 20],
            dtype=float,
        ),
        xlabel="x",
        log_x=True,
    )

    try:
        assert (
            fig.axes[0].get_xscale()
            ==
            "log"
        )

    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_rank_slopegraph_returns_input_rows():
    frame = pd.DataFrame(
        {
            "model": [
                "A",
                "B",
            ],
            "primary_rank": [
                1,
                2,
            ],
            "preference_rank": [
                2,
                1,
            ],
            "primary_ndcg_at_10_mean": [
                0.5,
                0.4,
            ],
            "preference_ndcg_at_10_mean": [
                0.45,
                0.46,
            ],
            "rank_shift": [
                1,
                -1,
            ],
        }
    )

    fig, data = gf.render_rank_slopegraph(
        frame
    )

    try:
        assert len(data) == 2
        assert data["model"].tolist() == [
            "A",
            "B",
        ]

    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
