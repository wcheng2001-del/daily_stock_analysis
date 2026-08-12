# -*- coding: utf-8 -*-
"""Regression tests for US-stock risk-scope prompt separation."""

from src.agent.agents.intel_agent import IntelAgent
from src.agent.agents.risk_agent import RiskAgent
from src.agent.protocols import AgentContext
from src.analyzer import GeminiAnalyzer
from src.market_context import detect_market, get_market_guidelines


def _minimal_context(code: str = "AAPL") -> dict:
    return {
        "code": code,
        "stock_name": "Apple Inc.",
        "date": "2026-08-12",
        "today": {
            "close": 230.0,
            "open": 228.0,
            "high": 232.0,
            "low": 226.0,
            "pct_chg": 1.2,
            "volume": 1000000,
            "amount": 230000000,
            "ma5": 225.0,
            "ma10": 220.0,
            "ma20": 215.0,
        },
        "ma_status": "多头排列",
    }


def test_us_market_guidelines_exclude_a_share_only_scope() -> None:
    assert detect_market("AAPL") == "us"

    guidelines = get_market_guidelines("AAPL")

    assert "美股" in guidelines
    assert "不要套用 A 股" in guidelines
    assert "游资" in guidelines
    assert "解禁/减持专项" in guidelines
    assert "SEC filings" in guidelines


def test_stock_analyzer_us_prompt_does_not_penalize_a_share_missing_data() -> None:
    prompt = GeminiAnalyzer()._format_prompt(
        _minimal_context("AVGO"),
        "Broadcom Inc.",
        news_context="2026-08-12: Broadcom analyst target price raised.",
        report_language="zh",
    )

    assert "美股分析约束" in prompt
    assert "不要套用 A 股政策专项、游资/龙虎榜、解禁/减持专项、主力资金流" in prompt
    assert "不得作为美股利空或买入否决依据" in prompt
    assert "财报/guidance" in prompt
    assert "SEC" in prompt


def test_stock_analyzer_cn_prompt_keeps_a_share_risk_scope() -> None:
    prompt = GeminiAnalyzer()._format_prompt(
        _minimal_context("600519"),
        "贵州茅台",
        news_context="2026-08-12: 公司公告。",
        report_language="zh",
    )

    assert "风险警报**：减持、处罚、利空" in prompt
    assert "美股分析约束" not in prompt


def test_agent_prompts_are_market_aware_for_us_stocks() -> None:
    ctx = AgentContext(stock_code="NVDA", stock_name="NVIDIA Corporation")

    risk_prompt = RiskAgent().system_prompt(ctx)
    intel_prompt = IntelAgent().system_prompt(ctx)
    intel_user_message = IntelAgent().build_user_message(ctx)

    for text in (risk_prompt, intel_prompt):
        assert "US Stock" in text
        assert "Do NOT apply China A-share-only" in text or "Skip A-share-only" in text
        assert "Missing A-share-only fields" in text

    assert "Do not call get_capital_flow" in intel_user_message
    assert "capital_flow_signal='not_available'" in intel_user_message
