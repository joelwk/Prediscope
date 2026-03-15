from __future__ import annotations

import sqlite3

from calibration import (
    build_counterfactual_flags,
    compute_adaptive_thresholds,
    compute_category_edge_adjustments,
)


def test_build_counterfactual_flags_with_edge() -> None:
    flags = build_counterfactual_flags(0.051)
    assert flags["would_trade_at_03"] is True
    assert flags["would_trade_at_04"] is True
    assert flags["would_trade_at_05"] is True
    assert flags["would_trade_at_06"] is False


def test_build_counterfactual_flags_without_edge() -> None:
    flags = build_counterfactual_flags(None)
    assert all(value is None for value in flags.values())


def test_compute_adaptive_thresholds_with_sufficient_samples() -> None:
    samples = []
    for i in range(30):
        samples.append(
            {
                "edge_market": 0.03 + (i * 0.001),
                "evidence_quality": 0.6,
                "analysis_duration_ms": 42000 + i * 100,
                "orderbook_spread_abs": 0.06 + (i * 0.001),
            }
        )
    recommendation = compute_adaptive_thresholds(
        samples=samples,
        current_edge_threshold=0.05,
        current_spread_cutoff=0.08,
        current_workers=2,
        min_samples=20,
    )
    assert recommendation["insufficient_edge_data"] is False
    assert recommendation["insufficient_spread_data"] is False
    assert 0.03 <= recommendation["recommended_min_market_edge_for_trade"] <= 0.07
    assert 0.06 <= recommendation["recommended_orderbook_spread_cutoff"] <= 0.12
    assert recommendation["recommended_analysis_max_workers"] >= 2


def test_compute_adaptive_thresholds_with_insufficient_samples() -> None:
    samples = [{"edge_market": 0.02, "evidence_quality": 0.8}]
    recommendation = compute_adaptive_thresholds(
        samples=samples,
        current_edge_threshold=0.05,
        current_spread_cutoff=0.08,
        current_workers=3,
        min_samples=20,
    )
    assert recommendation["insufficient_edge_data"] is True
    assert recommendation["recommended_min_market_edge_for_trade"] == 0.05
    assert recommendation["recommended_orderbook_spread_cutoff"] == 0.08
    assert recommendation["recommended_analysis_max_workers"] == 3


def test_compute_category_edge_adjustments_recommends_underperformers(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE trade_outcomes (
                market_id TEXT PRIMARY KEY,
                sport_subcategory TEXT,
                entry_price REAL,
                won INTEGER
            )
            """
        )
        rows = []
        for idx in range(12):
            rows.append((f"es-{idx}", "esports", 0.70, 1 if idx < 3 else 0))
        for idx in range(12):
            rows.append((f"hk-{idx}", "hockey", 0.55, 1 if idx < 7 else 0))
        conn.executemany(
            "INSERT INTO trade_outcomes (market_id, sport_subcategory, entry_price, won) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    recs = compute_category_edge_adjustments(str(db_path), min_samples=10)
    assert recs.get("esports") == 0.1
    assert "hockey" not in recs
