import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from config import Settings
from main import (
    _bayesian_min_updates_for_market,
    _best_orderbook_sell_price,
    _build_reasoning_hash,
    _cap_effective_confidence_for_market,
    _calculate_bet,
    _compute_next_wakeup_seconds,
    _edge_threshold_for_market,
    _effective_position_override_threshold,
    _filter_markets,
    _is_sports_market,
    _log_settings_summary,
    _score_decision_for_market,
    _should_adjust_position,
    _should_apply_bayesian_to_market,
    _sports_guardrail_block_reason,
    _usdc_from_wei,
)
from models import Market, MarketOutcome, MarketState, Position, TradeDecision


class DummyStateManager:
    def __init__(self, mapping: dict[str, MarketState | None]) -> None:
        self.mapping = mapping

    def get_market_state(self, market_id: str) -> MarketState | None:
        return self.mapping.get(market_id)


class TestMainUtils(unittest.TestCase):
    def test_filter_markets(self) -> None:
        markets = [
            Market(id="1", question="Q1", liquidity_usdc=50, category="sports"),
            Market(id="2", question="Q2", liquidity_usdc=150, category="sports"),
            Market(id="3", question="Q3", liquidity_usdc=200, category="politics"),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=("sports",),
            blocklist=("politics",),
        )
        self.assertEqual([m.id for m in filtered], ["2"])

    def test_filter_markets_by_close_date(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=6)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=3)),
            Market(id="3", question="Q3", close_time=now + timedelta(days=10)),
            Market(id="4", question="Q4", close_time=None),
        ]
        # Filter: only markets closing between 1 and 7 days from now
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            min_close_days=1,
            max_close_days=7,
        )
        # Market 1 closes too soon (<1 day), Market 3 closes too far (>7 days)
        # Market 4 has no close_time, so it passes (no filter applied)
        self.assertEqual([m.id for m in filtered], ["2", "4"])

    def test_filter_markets_max_close_days_only(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=12)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=5)),
        ]
        # Only set max_close_days (markets closing within 3 days)
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            max_close_days=3,
        )
        self.assertEqual([m.id for m in filtered], ["1"])

    def test_calculate_bet(self) -> None:
        self.assertEqual(_calculate_bet(100, 0.5), 50)
        self.assertEqual(_calculate_bet(100, -1), 0)
        self.assertEqual(_calculate_bet(100, 2), 100)

    def test_filter_markets_populates_skip_counters(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="open", question="Open market", category="sports", liquidity_usdc=200, close_time=now + timedelta(days=2)),
            Market(id="low", question="Low liquidity", category="sports", liquidity_usdc=10, close_time=now + timedelta(days=2)),
            Market(id="blocked", question="Blocked category", category="politics", liquidity_usdc=200, close_time=now + timedelta(days=2)),
            Market(id="soon", question="Closing soon", category="sports", liquidity_usdc=200, close_time=now + timedelta(hours=4)),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=("politics",),
            min_close_days=1,
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["open"])
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["skipped_liquidity"], 1)
        self.assertEqual(stats["skipped_blocklist"], 1)
        self.assertEqual(stats["skipped_close_too_soon"], 1)

    def test_best_orderbook_sell_price(self) -> None:
        orderbook = {
            "sells": [
                {"optionIndex": 0, "price": 0.62},
                {"optionIndex": 1, "price": 0.44},
                {"optionIndex": 0, "price": 0.60},
            ]
        }
        self.assertAlmostEqual(_best_orderbook_sell_price(orderbook, 0) or 0.0, 0.60)
        self.assertAlmostEqual(_best_orderbook_sell_price(orderbook, 1) or 0.0, 0.44)
        self.assertIsNone(_best_orderbook_sell_price(orderbook, 2))

    def test_log_settings_summary_includes_phase1_flags(self) -> None:
        settings = Settings(
            BAYESIAN_ENABLED=False,
            LMSR_ENABLED=False,
            KELLY_SIZING_ENABLED=True,
            KELLY_FRACTION_DEFAULT=0.2,
            KELLY_FRACTION_SHORT_HORIZON_HOURS=1,
            KELLY_FRACTION_SHORT_HORIZON=0.1,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        with patch("main.logger.info") as info_mock:
            _log_settings_summary(settings)

        self.assertTrue(info_mock.called)
        summary_data = {}
        strict_hint_data = {}
        for call in info_mock.call_args_list:
            data = call.kwargs.get("data") or {}
            if "dry_run" in data:
                summary_data = data
            if "effective_min_bet_pct" in data:
                strict_hint_data = data
        data = summary_data
        self.assertEqual(data.get("bayesian_enabled"), False)
        self.assertEqual(data.get("lmsr_enabled"), False)
        self.assertEqual(data.get("kelly_sizing_enabled"), True)
        self.assertEqual(data.get("kelly_fraction_default"), 0.2)
        self.assertEqual(data.get("kelly_fraction_short_horizon_hours"), 1)
        self.assertEqual(data.get("kelly_fraction_short_horizon"), 0.1)
        self.assertEqual(data.get("kelly_min_bet_policy"), "fallback_edge_scaling")
        self.assertGreater(strict_hint_data.get("effective_min_bet_pct", 0.0), 0.0)

    def test_compute_next_wakeup_seconds_uses_action_aware_cooldown(self) -> None:
        now = datetime.now(timezone.utc)
        market = Market(
            id="m-cooldown",
            question="Cooldown test",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            close_time=now + timedelta(days=2),
        )
        state = MarketState(
            market_id="m-cooldown",
            last_analysis=now - timedelta(minutes=20),
            analysis_count=1,
            last_confidence=0.55,
            confidence_trend=[0.55],
            last_terminal_outcome="no_trade_recommended",
        )
        state_manager = DummyStateManager({"m-cooldown": state})
        settings = Settings(
            REANALYSIS_COOLDOWN_HOURS=6,
            URGENT_REANALYSIS_DAYS_BEFORE_CLOSE=1,
            URGENT_REANALYSIS_COOLDOWN_HOURS=1,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        wakeup_seconds = _compute_next_wakeup_seconds(
            [market],
            state_manager,
            settings,
            now=now,
        )
        self.assertEqual(wakeup_seconds, 1)

    def test_cap_effective_confidence_for_market_respects_category_caps(self) -> None:
        settings = Settings(
            MAX_SPORTS_CONFIDENCE=0.80,
            MAX_ESPORTS_CONFIDENCE=0.75,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        sports_market = Market(id="s1", question="NBA: A vs B", category="sports")
        esports_market = Market(id="e1", question="Esports: A vs B", category="esports")
        politics_market = Market(id="p1", question="Election", category="politics")

        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, sports_market, settings),
            0.80,
        )
        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, esports_market, settings),
            0.75,
        )
        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, politics_market, settings),
            0.99,
        )

    def test_should_apply_bayesian_to_market_respects_sports_toggle(self) -> None:
        sports_market = Market(id="s2", question="NHL: A vs B", category="sports")
        politics_market = Market(id="p2", question="Election", category="politics")
        sports_blocked = Settings(
            BAYESIAN_ENABLED=True,
            BAYESIAN_APPLY_TO_SPORTS=False,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        self.assertFalse(_should_apply_bayesian_to_market(sports_market, sports_blocked))
        self.assertTrue(_should_apply_bayesian_to_market(politics_market, sports_blocked))

        sports_allowed = Settings(
            BAYESIAN_ENABLED=True,
            BAYESIAN_APPLY_TO_SPORTS=True,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        self.assertTrue(_should_apply_bayesian_to_market(sports_market, sports_allowed))

    def test_score_decision_for_market_caps_sports_evidence_quality(self) -> None:
        settings = Settings(
            SCORE_MAX_EVIDENCE_QUALITY_SPORTS=0.65,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        sports_market = Market(id="s3", question="NBA: A vs B", category="sports")
        decision = TradeDecision(
            should_trade=True,
            outcome="A",
            confidence=0.7,
            bet_size_pct=0.5,
            reasoning="test",
            evidence_quality=0.95,
        )
        scored = _score_decision_for_market(decision, sports_market, settings)
        self.assertEqual(scored.evidence_quality, 0.65)

    def test_sports_guardrail_block_reason_requires_model_confidence_and_external_edge(self) -> None:
        settings = Settings(
            SPORTS_MIN_MODEL_CONFIDENCE=0.66,
            SPORTS_REQUIRE_EXTERNAL_EDGE=True,
            SPORTS_MIN_EXTERNAL_EDGE=0.04,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        sports_market = Market(id="s4", question="NHL: A vs B", category="sports")
        self.assertEqual(
            _sports_guardrail_block_reason(
                sports_market,
                settings,
                model_confidence=0.63,
                edge_external=0.06,
            ),
            "sports_model_confidence_below_min",
        )
        self.assertEqual(
            _sports_guardrail_block_reason(
                sports_market,
                settings,
                model_confidence=0.70,
                edge_external=0.02,
            ),
            "sports_external_edge_below_min",
        )
        self.assertIsNone(
            _sports_guardrail_block_reason(
                sports_market,
                settings,
                model_confidence=0.70,
                edge_external=0.06,
            )
        )

    def test_is_sports_market_true_for_sports_and_esports(self) -> None:
        self.assertTrue(_is_sports_market(Market(id="s5", question="NBA game", category="sports")))
        self.assertTrue(_is_sports_market(Market(id="e5", question="Valorant match", category="esports")))
        self.assertFalse(_is_sports_market(Market(id="p5", question="Election", category="politics")))

    def test_edge_threshold_applies_fallback_and_coinflip_guards(self) -> None:
        settings = Settings(
            MIN_EDGE=0.05,
            LOW_PRICE_MIN_EDGE=0.08,
            LOW_PRICE_THRESHOLD=0.50,
            COINFLIP_PRICE_LOWER=0.45,
            COINFLIP_PRICE_UPPER=0.55,
            FALLBACK_EDGE_MIN_EDGE=0.08,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        self.assertEqual(_edge_threshold_for_market(0.60, settings, "computed"), 0.05)
        self.assertEqual(_edge_threshold_for_market(0.52, settings, "computed"), 0.08)
        self.assertEqual(_edge_threshold_for_market(0.60, settings, "fallback"), 0.08)

    def test_should_adjust_position_uses_bankroll_relative_cap(self) -> None:
        settings = Settings(
            MAX_POSITION_PER_MARKET_USDC=200.0,
            MAX_POSITION_PCT_OF_BANKROLL=0.15,
            POSITION_CAP_FLOOR_TO_MIN_BET=True,
            MIN_BET_USDC=5.0,
            MAX_BET_USDC=50.0,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.70,
            bet_size_pct=1.0,
            reasoning="test",
        )
        existing_position = Position(
            market_id="m-bankroll",
            outcome="YES",
            total_amount_usdc=2.5,
            avg_confidence=0.60,
            trade_count=1,
            first_trade=datetime.now(timezone.utc),
            last_trade=datetime.now(timezone.utc),
        )
        allowed, bet_pct, reason = _should_adjust_position(
            decision=decision,
            market=Market(id="m-bankroll", question="Q", category="sports"),
            existing_position=existing_position,
            state=None,
            settings=settings,
            cycle_bankroll=20.0,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "confidence_increase_threshold_met")
        self.assertAlmostEqual(bet_pct, 0.05, places=4)

    def test_bayesian_min_updates_uses_short_horizon_override(self) -> None:
        now = datetime.now(timezone.utc)
        near_market = Market(
            id="m-near",
            question="Near close",
            close_time=now + timedelta(hours=3),
        )
        far_market = Market(
            id="m-far",
            question="Far close",
            close_time=now + timedelta(hours=48),
        )
        settings = Settings(
            BAYESIAN_MIN_UPDATES_FOR_TRADE=2,
            BAYESIAN_SHORT_HORIZON_HOURS=24,
            BAYESIAN_MIN_UPDATES_SHORT_HORIZON=1,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        self.assertEqual(_bayesian_min_updates_for_market(near_market, settings), 1)
        self.assertEqual(_bayesian_min_updates_for_market(far_market, settings), 2)

    def test_build_reasoning_hash_ignores_validated_prefix_variation(self) -> None:
        decision_a = TradeDecision(
            should_trade=False,
            outcome="Yes",
            confidence=0.70,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=1.00 gate=allow reason=ok edge_market=0.041 "
                "edge_source=computed] Core thesis unchanged"
            ),
        )
        decision_b = TradeDecision(
            should_trade=False,
            outcome="Yes",
            confidence=0.70,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=0.95 gate=allow reason=ok edge_market=0.038 "
                "edge_source=computed] Core thesis unchanged"
            ),
        )
        self.assertEqual(_build_reasoning_hash(decision_a), _build_reasoning_hash(decision_b))

    def test_effective_position_override_threshold_not_capped_by_category(self) -> None:
        settings = Settings(
            HIGH_CONFIDENCE_POSITION_OVERRIDE=0.85,
            MAX_SPORTS_CONFIDENCE=0.80,
            XAI_API_KEY="xai-key",
            WALLET_PRIVATE_KEY="wallet-key",
        )
        sports_market = Market(id="s2", question="NBA: A vs B", category="sports")
        threshold = _effective_position_override_threshold(sports_market, settings)
        self.assertEqual(threshold, 0.85)
        self.assertFalse(0.80 >= threshold)

    def test_usdc_from_wei_conversion(self) -> None:
        self.assertEqual(_usdc_from_wei(2_500_000, 6), 2.5)


if __name__ == "__main__":
    unittest.main()
