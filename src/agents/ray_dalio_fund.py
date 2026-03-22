from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    detect_fund_type,
    get_etf_profile,
    get_etf_holdings,
    get_etf_sector,
    get_etf_country,
    get_mutual_fund_profile,
    get_mutual_fund_holdings,
    get_fund_price_history,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm


class RayDalioFundSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ray_dalio_fund_agent(state: AgentState, agent_id: str = "ray_dalio_fund_agent"):
    """
    Evaluates ETFs and mutual funds using Ray Dalio's All Weather / Risk Parity principles.
    Scores funds on cost efficiency, geographic spread, sector spread, AUM liquidity, and
    price momentum. Passes the quantitative analysis to an LLM for a final signal.
    """
    data = state["data"]
    tickers = data["tickers"]
    dalio_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Detecting fund type")
        fund_type = detect_fund_type(ticker)

        if fund_type == "EQUITY":
            dalio_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": f"{ticker} is an equity, not a fund. Ray Dalio Fund agent only evaluates ETFs and mutual funds.",
            }
            progress.update_status(agent_id, ticker, "Done", analysis=dalio_analysis[ticker]["reasoning"])
            continue

        # Fetch fund data based on type
        if fund_type == "ETF":
            progress.update_status(agent_id, ticker, "Fetching ETF profile")
            profile = get_etf_profile(ticker)

            progress.update_status(agent_id, ticker, "Fetching ETF holdings")
            holdings = get_etf_holdings(ticker)

            progress.update_status(agent_id, ticker, "Fetching ETF sector exposures")
            sector_data = get_etf_sector(ticker)

            progress.update_status(agent_id, ticker, "Fetching ETF country exposures")
            country_data = get_etf_country(ticker)
        else:
            # MUTUALFUND
            progress.update_status(agent_id, ticker, "Fetching mutual fund profile")
            profile = get_mutual_fund_profile(ticker)

            progress.update_status(agent_id, ticker, "Fetching mutual fund holdings")
            holdings = get_mutual_fund_holdings(ticker)

            # Mutual fund sector/country may come from profile; set empty dicts as fallback
            sector_data = {}
            country_data = {}

        progress.update_status(agent_id, ticker, "Fetching fund price history")
        price_history = get_fund_price_history(ticker)

        # Run scoring dimensions
        progress.update_status(agent_id, ticker, "Scoring cost efficiency")
        cost_analysis = analyze_cost_efficiency(profile, fund_type)

        progress.update_status(agent_id, ticker, "Scoring geographic spread")
        geo_analysis = analyze_geographic_spread(profile, country_data, fund_type)

        progress.update_status(agent_id, ticker, "Scoring sector spread")
        sector_analysis = analyze_sector_spread(profile, sector_data, fund_type)

        progress.update_status(agent_id, ticker, "Scoring AUM liquidity")
        aum_analysis = analyze_aum_liquidity(profile, fund_type)

        progress.update_status(agent_id, ticker, "Scoring price momentum")
        momentum_analysis = analyze_price_momentum(price_history)

        progress.update_status(agent_id, ticker, "Scoring holdings concentration")
        concentration_analysis = analyze_holdings_concentration(holdings, fund_type)

        # Aggregate scores (only dimensions that returned a score)
        dimension_scores = [
            cost_analysis["score"],
            geo_analysis["score"],
            sector_analysis["score"],
            aum_analysis["score"],
            momentum_analysis["score"],
            concentration_analysis["score"],
        ]
        valid_scores = [s for s in dimension_scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 5.0

        if avg_score >= 7:
            quant_signal = "bullish"
        elif avg_score >= 4:
            quant_signal = "neutral"
        else:
            quant_signal = "bearish"

        ticker_analysis = {
            "fund_type": fund_type,
            "quant_signal": quant_signal,
            "avg_score": round(avg_score, 2),
            "cost_efficiency": cost_analysis,
            "geographic_spread": geo_analysis,
            "sector_spread": sector_analysis,
            "aum_liquidity": aum_analysis,
            "price_momentum": momentum_analysis,
            "holdings_concentration": concentration_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Ray Dalio analysis")
        dalio_output = generate_dalio_output(
            ticker=ticker,
            analysis_data=ticker_analysis,
            state=state,
            agent_id=agent_id,
        )

        dalio_analysis[ticker] = {
            "signal": dalio_output.signal,
            "confidence": dalio_output.confidence,
            "reasoning": dalio_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=dalio_output.reasoning)

    message = HumanMessage(content=json.dumps(dalio_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(dalio_analysis, "Ray Dalio Fund Agent")

    state["data"]["analyst_signals"][agent_id] = dalio_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
    }


# ---------------------------------------------------------------------------
# Scoring helpers — each returns {"score": 0-10 or None, "details": str}
# ---------------------------------------------------------------------------

def _extract_profile_dict(profile: dict, fund_type: str) -> dict:
    """Normalize profile response — ETF uses 'profile' key, MF may be flat."""
    if not profile:
        return {}
    # ETF profile from Finnhub is nested under 'profile'
    if "profile" in profile and isinstance(profile["profile"], dict):
        return profile["profile"]
    return profile


def analyze_cost_efficiency(profile: dict, fund_type: str) -> dict:
    """Score expense ratio. Lower is better per Dalio risk-parity: want low-cost passive exposure."""
    p = _extract_profile_dict(profile, fund_type)
    # Try multiple field name variants
    expense_ratio = p.get("expenseRatio") or p.get("expense_ratio") or p.get("ter")
    if expense_ratio is None:
        return {"score": 5, "details": "Expense ratio data not available; using neutral score."}

    try:
        er = float(expense_ratio)
    except (TypeError, ValueError):
        return {"score": 5, "details": f"Could not parse expense ratio value: {expense_ratio}"}

    # er is expected as a decimal (e.g., 0.0009 = 0.09%) or percent (e.g., 0.09)
    # Finnhub typically returns as decimal fraction; if > 1 assume it's already in percent
    if er > 1:
        er = er / 100

    if er < 0.001:
        score = 10
        flag = "excellent (<0.1%)"
    elif er < 0.003:
        score = 7
        flag = "good (<0.3%)"
    elif er < 0.005:
        score = 4
        flag = "moderate (<0.5%)"
    else:
        score = 1
        flag = f"costly ({er*100:.2f}% — above 0.5% threshold)"

    return {"score": score, "details": f"Expense ratio {er*100:.3f}% — {flag}.", "expense_ratio_pct": round(er * 100, 4)}


def analyze_geographic_spread(profile: dict, country_data: dict, fund_type: str) -> dict:
    """
    Score country diversification. Dalio's All Weather thrives on uncorrelated geographies.
    Penalizes single-country concentration above 60%.
    """
    # Try to get country exposure list
    exposures = []
    if country_data:
        exposures = country_data.get("countryExposure") or country_data.get("data") or []

    # Mutual funds may embed country data inside profile
    if not exposures:
        p = _extract_profile_dict(profile, fund_type)
        exposures = p.get("countryExposure") or p.get("country_exposure") or []

    if not exposures or not isinstance(exposures, list):
        return {"score": 5, "details": "Country exposure data not available; using neutral score."}

    # Build country weight map; field names vary: 'exposure', 'weight', 'percent'
    weights = {}
    for item in exposures:
        if not isinstance(item, dict):
            continue
        country = item.get("country") or item.get("name") or "Unknown"
        weight = item.get("exposure") or item.get("weight") or item.get("percent") or 0
        try:
            weights[country] = float(weight)
        except (TypeError, ValueError):
            pass

    if not weights:
        return {"score": 5, "details": "Could not parse country exposure data; using neutral score."}

    top_country = max(weights, key=lambda c: weights[c])
    top_weight = weights[top_country]
    num_countries = len(weights)

    if top_weight > 0.60:
        score = 2
        flag = f"heavy concentration in {top_country} ({top_weight*100:.1f}%) — single-country risk"
    elif top_weight > 0.40:
        score = 5
        flag = f"{top_country} dominant ({top_weight*100:.1f}%) but not extreme"
    elif num_countries >= 5 and top_weight <= 0.35:
        score = 10
        flag = f"well-diversified across {num_countries} countries (top: {top_country} {top_weight*100:.1f}%)"
    else:
        score = 7
        flag = f"{num_countries} countries, top {top_country} at {top_weight*100:.1f}%"

    return {"score": score, "details": f"Geographic spread: {flag}.", "top_country": top_country, "top_weight_pct": round(top_weight * 100, 1), "num_countries": num_countries}


def analyze_sector_spread(profile: dict, sector_data: dict, fund_type: str) -> dict:
    """
    Score sector diversification. Dalio avoids sector concentration risk.
    Penalizes any single sector above 40%.
    """
    exposures = []
    if sector_data:
        exposures = sector_data.get("sectorExposure") or sector_data.get("data") or []

    if not exposures:
        p = _extract_profile_dict(profile, fund_type)
        exposures = p.get("sectorExposure") or p.get("sector_exposure") or []

    if not exposures or not isinstance(exposures, list):
        return {"score": 5, "details": "Sector exposure data not available; using neutral score."}

    weights = {}
    for item in exposures:
        if not isinstance(item, dict):
            continue
        sector = item.get("sector") or item.get("name") or "Unknown"
        weight = item.get("exposure") or item.get("weight") or item.get("percent") or 0
        try:
            weights[sector] = float(weight)
        except (TypeError, ValueError):
            pass

    if not weights:
        return {"score": 5, "details": "Could not parse sector exposure data; using neutral score."}

    top_sector = max(weights, key=lambda s: weights[s])
    top_weight = weights[top_sector]
    num_sectors = len(weights)

    if top_weight > 0.40:
        score = 3
        flag = f"concentrated in {top_sector} ({top_weight*100:.1f}%) — above 40% threshold"
    elif top_weight > 0.25:
        score = 6
        flag = f"{top_sector} leads at {top_weight*100:.1f}% across {num_sectors} sectors"
    else:
        score = 10
        flag = f"well-distributed — no sector above 25% (top: {top_sector} {top_weight*100:.1f}%)"

    return {"score": score, "details": f"Sector spread: {flag}.", "top_sector": top_sector, "top_weight_pct": round(top_weight * 100, 1), "num_sectors": num_sectors}


def analyze_aum_liquidity(profile: dict, fund_type: str) -> dict:
    """
    Score AUM (total net assets). Dalio's All Weather favors large, liquid funds.
    Penalizes funds below $100M as illiquid.
    """
    p = _extract_profile_dict(profile, fund_type)
    # Try multiple field name variants; Finnhub ETF profile uses 'TNA' (total net assets)
    aum = p.get("TNA") or p.get("totalAssets") or p.get("totalNetAssets") or p.get("aum") or p.get("netAssets")
    if aum is None:
        return {"score": 5, "details": "AUM data not available; using neutral score."}

    try:
        aum_val = float(aum)
    except (TypeError, ValueError):
        return {"score": 5, "details": f"Could not parse AUM value: {aum}"}

    B = 1_000_000_000
    M = 1_000_000

    if aum_val >= 10 * B:
        score = 10
        flag = f"${aum_val/B:.1f}B — highly liquid"
    elif aum_val >= 1 * B:
        score = 7
        flag = f"${aum_val/B:.1f}B — adequately liquid"
    elif aum_val >= 100 * M:
        score = 4
        flag = f"${aum_val/M:.0f}M — acceptable but watch liquidity"
    else:
        score = 1
        flag = f"${aum_val/M:.1f}M — below $100M threshold, liquidity risk"

    return {"score": score, "details": f"AUM: {flag}.", "aum_usd": aum_val}


def analyze_price_momentum(price_history: list) -> dict:
    """
    Score 1-year price momentum using Tiingo data.
    Positive return = 5, negative = 2. No data = 3 (slight negative since unknown).
    """
    if not price_history or len(price_history) < 2:
        return {"score": 3, "details": "Price history unavailable (TIINGO_API_KEY not set or no data); using below-neutral score."}

    try:
        # Tiingo returns oldest-first; fields: adjClose or close
        first = price_history[0]
        last = price_history[-1]
        start_price = float(first.get("adjClose") or first.get("close") or 0)
        end_price = float(last.get("adjClose") or last.get("close") or 0)
        if start_price <= 0:
            return {"score": 3, "details": "Could not compute return — invalid start price."}
        one_yr_return = (end_price - start_price) / start_price
    except (TypeError, ValueError, IndexError) as e:
        return {"score": 3, "details": f"Price momentum calculation error: {e}"}

    if one_yr_return >= 0:
        score = 5
        flag = f"positive ({one_yr_return*100:.1f}%)"
    else:
        score = 2
        flag = f"negative ({one_yr_return*100:.1f}%)"

    return {"score": score, "details": f"1-year price return: {flag}.", "one_yr_return_pct": round(one_yr_return * 100, 2)}


def analyze_holdings_concentration(holdings: dict, fund_type: str) -> dict:
    """
    Score top-10 holdings concentration. Dalio prefers broad diversification.
    Flag if top 10 holdings represent more than 50% of the fund.
    """
    if not holdings:
        return {"score": 5, "details": "Holdings data not available; using neutral score."}

    # Holdings list may be under 'holdings' key or 'data'
    holdings_list = holdings.get("holdings") or holdings.get("data") or []
    if not isinstance(holdings_list, list) or not holdings_list:
        return {"score": 5, "details": "Could not parse holdings data; using neutral score."}

    # Extract weight for each holding; field names vary
    weights = []
    for item in holdings_list:
        if not isinstance(item, dict):
            continue
        w = item.get("percent") or item.get("weight") or item.get("portfolioPercent") or 0
        try:
            weights.append(float(w))
        except (TypeError, ValueError):
            pass

    if not weights:
        return {"score": 5, "details": "Could not extract holding weights; using neutral score."}

    weights_sorted = sorted(weights, reverse=True)
    top10_sum = sum(weights_sorted[:10])

    # Weights may be in 0-100 range or 0-1 range
    if top10_sum > 1:
        top10_pct = top10_sum
    else:
        top10_pct = top10_sum * 100

    if top10_pct > 50:
        score = 3
        flag = f"top-10 holdings = {top10_pct:.1f}% — above 50% concentration threshold"
    elif top10_pct > 35:
        score = 6
        flag = f"top-10 holdings = {top10_pct:.1f}% — moderate concentration"
    else:
        score = 9
        flag = f"top-10 holdings = {top10_pct:.1f}% — well diversified"

    return {"score": score, "details": f"Holdings concentration: {flag}.", "top10_weight_pct": round(top10_pct, 1)}


# ---------------------------------------------------------------------------
# LLM output generation
# ---------------------------------------------------------------------------

def generate_dalio_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> RayDalioFundSignal:
    """Generate a Ray Dalio All Weather signal via LLM."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Ray Dalio AI agent applying All Weather / Risk Parity principles to fund selection.

            Core philosophy:
            1. Balance risk across all economic environments: growth, recession, inflation, deflation.
            2. Favor broad diversification across asset classes, geographies, and sectors — not concentration in a single bet.
            3. Prefer low-cost, liquid funds. High expense ratios silently erode long-term compounding.
            4. Large AUM signals institutional trust and liquidity — small funds carry hidden risks.
            5. No single country or sector should dominate. Correlation kills diversification.
            6. Price momentum is a secondary signal — Dalio does not chase short-term performance.

            In your reasoning:
            - Quote the specific scores and data points provided.
            - Call out any dimension that scored poorly (score < 4) as a key risk.
            - Praise dimensions that scored well (score >= 8) as All Weather strengths.
            - Note any dimensions where data was unavailable and how that affects conviction.
            - Conclude with a clear, principled recommendation in Dalio's measured, risk-conscious tone.

            Return your final recommendation (signal: bullish, neutral, or bearish) with a 0-100 confidence and thorough reasoning.
            """,
        ),
        (
            "human",
            """Based on the following All Weather quantitative analysis, generate a Ray Dalio fund investment signal.

            Fund: {ticker}
            Analysis:
            {analysis_data}

            Return strictly valid JSON:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
        ),
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "analysis_data": json.dumps(analysis_data, indent=2),
    })

    def create_default_signal():
        return RayDalioFundSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error generating analysis, defaulting to neutral.",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=RayDalioFundSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
