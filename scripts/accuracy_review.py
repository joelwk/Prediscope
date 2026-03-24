from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config import Settings, load_settings
except ModuleNotFoundError:  # pragma: no cover - CLI fallback
    Settings = None
    load_settings = None

_DEFAULT_MARKET_IDS = ("20983", "20910")
_FALLBACK_DEFAULTS: dict[str, str] = {
    "DRY_RUN": "True",
    "EXECUTE_ONCHAIN": "False",
    "AUTO_APPROVE_USDC": "False",
    "MIN_CONFIDENCE": "0.55",
    "MIN_EDGE": "0.05",
    "LOW_PRICE_MIN_EDGE": "0.12",
    "MIN_LIQUIDITY_USDC": "100.0",
    "MAX_SPORTS_CONFIDENCE": "0.8",
    "SCORE_GATE_MODE": "active",
    "SCORE_GATE_THRESHOLD": "0.1",
    "SCORE_MAX_EVIDENCE_QUALITY_SPORTS": "0.7",
    "BAYESIAN_ENABLED": "False",
    "BAYESIAN_APPLY_TO_SPORTS": "False",
    "KELLY_SIZING_ENABLED": "False",
    "SPORTS_MIN_MODEL_CONFIDENCE": "0.66",
    "SPORTS_REQUIRE_EXTERNAL_EDGE": "True",
    "SPORTS_MIN_EXTERNAL_EDGE": "0.04",
    "REANALYSIS_COOLDOWN_HOURS": "6",
    "URGENT_REANALYSIS_COOLDOWN_HOURS": "1",
}


@dataclass(frozen=True)
class ArtifactPaths:
    db_path: Path
    predictbot_log_path: Path
    trades_log_path: Path


def _resolve_artifacts(db: str, predictbot_log: str, trades_log: str) -> ArtifactPaths:
    paths = ArtifactPaths(
        db_path=Path(db),
        predictbot_log_path=Path(predictbot_log),
        trades_log_path=Path(trades_log),
    )
    missing = [str(path) for path in paths.__dict__.values() if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing runtime artifacts: {joined}")
    return paths


def _query_rows(conn: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
    return conn.execute(query, params).fetchall()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _select_existing_columns(conn: sqlite3.Connection, table_name: str, requested: tuple[str, ...]) -> str:
    available = _table_columns(conn, table_name)
    selected = [name for name in requested if name in available]
    if not selected:
        return "market_id"
    return ", ".join(selected)


def _load_market_snapshot(conn: sqlite3.Connection, market_id: str) -> dict[str, Any]:
    outcomes_columns = _select_existing_columns(
        conn,
        "trade_outcomes",
        (
            "market_id",
            "predicted_outcome",
            "sport_subcategory",
            "entry_price",
            "implied_prob",
            "confidence",
            "model_confidence",
            "bayesian_posterior",
            "final_score",
            "amount_usdc",
            "shares",
            "resolved_winning_outcome",
            "won",
            "pnl_estimate",
            "resolved_at",
            "resolution_state",
        ),
    )
    trade_outcomes = _query_rows(
        conn,
        """
        SELECT {columns}
        FROM trade_outcomes
        WHERE market_id = ?
        """.format(columns=outcomes_columns),
        (market_id,),
    )
    event_columns = _select_existing_columns(
        conn,
        "trade_outcome_events",
        (
            "market_id",
            "order_id",
            "predicted_outcome",
            "confidence",
            "model_confidence",
            "bayesian_posterior",
            "final_score",
            "edge_market",
            "edge_external",
            "evidence_quality",
            "amount_usdc",
            "shares",
            "timestamp",
            "analysis_id",
        ),
    )
    trade_events = _query_rows(
        conn,
        """
        SELECT {columns}
        FROM trade_outcome_events
        WHERE market_id = ?
        ORDER BY timestamp ASC
        """.format(columns=event_columns),
        (market_id,),
    )
    analyses = _query_rows(
        conn,
        """
        SELECT id, confidence, outcome, is_refined, refinement_reason, timestamp, reasoning
        FROM analyses
        WHERE market_id = ?
        ORDER BY timestamp ASC
        """,
        (market_id,),
    )
    return {
        "trade_outcomes": [dict(row) for row in trade_outcomes],
        "trade_outcome_events": [dict(row) for row in trade_events],
        "analyses": [dict(row) for row in analyses],
    }


def _iter_json_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _market_log_events(
    log_rows: list[dict[str, Any]],
    market_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in log_rows:
        data = row.get("data")
        if isinstance(data, dict) and str(data.get("market_id")) == market_id:
            result.append(row)
            continue
        message = str(row.get("message", ""))
        if f"[{market_id}]" in message or f"market={market_id}" in message:
            result.append(row)
    return result


def _print_config_diff() -> None:
    watched = (
        "DRY_RUN",
        "EXECUTE_ONCHAIN",
        "AUTO_APPROVE_USDC",
        "MIN_CONFIDENCE",
        "MIN_EDGE",
        "LOW_PRICE_MIN_EDGE",
        "MIN_LIQUIDITY_USDC",
        "MAX_SPORTS_CONFIDENCE",
        "SCORE_GATE_MODE",
        "SCORE_GATE_THRESHOLD",
        "SCORE_MAX_EVIDENCE_QUALITY_SPORTS",
        "BAYESIAN_ENABLED",
        "BAYESIAN_APPLY_TO_SPORTS",
        "KELLY_SIZING_ENABLED",
        "SPORTS_MIN_MODEL_CONFIDENCE",
        "SPORTS_REQUIRE_EXTERNAL_EDGE",
        "SPORTS_MIN_EXTERNAL_EDGE",
        "REANALYSIS_COOLDOWN_HOURS",
        "URGENT_REANALYSIS_COOLDOWN_HOURS",
    )
    print("\n=== Active config vs defaults ===")
    if load_settings is None or Settings is None:
        env_values = _parse_env_file(Path(".env"))
        for key in watched:
            active = env_values.get(key, "<unset>")
            default = _FALLBACK_DEFAULTS.get(key, "<unknown>")
            marker = "*" if active != "<unset>" and active != default else "-"
            print(f"{marker} {key}: active={active} default={default}")
        return
    settings = load_settings()
    for key in watched:
        active = getattr(settings, key)
        default = getattr(Settings, key)
        changed = active != default
        marker = "*" if changed else "-"
        print(f"{marker} {key}: active={active} default={default}")


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def run(db: str, predictbot_log: str, trades_log: str, market_ids: tuple[str, ...]) -> None:
    artifacts = _resolve_artifacts(db, predictbot_log, trades_log)
    _print_config_diff()

    conn = sqlite3.connect(str(artifacts.db_path))
    conn.row_factory = sqlite3.Row
    try:
        predict_rows = _iter_json_log(artifacts.predictbot_log_path)
        trades_rows = _iter_json_log(artifacts.trades_log_path)

        for market_id in market_ids:
            print(f"\n=== Market {market_id} ===")
            snapshot = _load_market_snapshot(conn, market_id)
            outcomes = snapshot["trade_outcomes"]
            events = snapshot["trade_outcome_events"]
            analyses = snapshot["analyses"]

            print(f"trade_outcomes rows: {len(outcomes)}")
            if outcomes:
                print(json.dumps(outcomes[0], ensure_ascii=True))
            print(f"trade_outcome_events rows: {len(events)}")
            for event in events:
                print(json.dumps(event, ensure_ascii=True))
            print(f"analyses rows: {len(analyses)}")
            for analysis in analyses[-6:]:
                trimmed = dict(analysis)
                reasoning = str(trimmed.get("reasoning") or "")
                trimmed["reasoning"] = reasoning[:240]
                print(json.dumps(trimmed, ensure_ascii=True))

            predict_events = _market_log_events(predict_rows, market_id)
            trade_events = _market_log_events(trades_rows, market_id)
            print(f"predictbot.log events: {len(predict_events)}")
            for row in predict_events[-12:]:
                print(
                    json.dumps(
                        {
                            "timestamp": row.get("timestamp"),
                            "level": row.get("level"),
                            "message": row.get("message"),
                        },
                        ensure_ascii=True,
                    )
                )
            print(f"trades.log events: {len(trade_events)}")
            for row in trade_events[-8:]:
                print(
                    json.dumps(
                        {
                            "timestamp": row.get("timestamp"),
                            "message": row.get("message"),
                        },
                        ensure_ascii=True,
                    )
                )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review recent losing trade windows.")
    parser.add_argument("--db", default="data/market_state.db")
    parser.add_argument("--predictbot-log", default="logs/predictbot.log")
    parser.add_argument("--trades-log", default="logs/trades.log")
    parser.add_argument(
        "--market-id",
        action="append",
        default=[],
        help="Market ID to inspect. Pass multiple times for multiple markets.",
    )
    args = parser.parse_args()
    selected_market_ids = tuple(args.market_id) if args.market_id else _DEFAULT_MARKET_IDS
    run(
        db=args.db,
        predictbot_log=args.predictbot_log,
        trades_log=args.trades_log,
        market_ids=selected_market_ids,
    )


if __name__ == "__main__":
    main()
