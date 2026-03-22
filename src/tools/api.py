import datetime
import logging
import os
import urllib3

import finnhub
import pandas as pd
import urllib3.exceptions
import yfinance as yf
from curl_cffi import requests as curl_requests

# Suppress SSL warnings from corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

logger = logging.getLogger(__name__)

# Global cache instance
_cache = get_cache()

# curl_cffi session — handles corporate SSL proxy environments by disabling cert verification
_yf_session: curl_requests.Session | None = None


def _get_yf_session() -> curl_requests.Session:
    global _yf_session
    if _yf_session is None:
        _yf_session = curl_requests.Session(impersonate="chrome", verify=False)
    return _yf_session


# Lazy Finnhub client
_finnhub_client: finnhub.Client | None = None


def _get_finnhub_client() -> finnhub.Client:
    global _finnhub_client
    if _finnhub_client is None:
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        _finnhub_client = finnhub.Client(api_key=api_key)
        # Disable SSL verification for corporate proxy environments
        _finnhub_client._session.verify = False
    return _finnhub_client


# ---------------------------------------------------------------------------
# Mapping from our canonical line-item names to yfinance statement labels
# ---------------------------------------------------------------------------
_LINE_ITEM_MAP: dict[str, str] = {
    "revenue": "Total Revenue",
    "net_income": "Net Income",
    "operating_income": "Operating Income",
    "gross_profit": "Gross Profit",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "free_cash_flow": "Free Cash Flow",
    "capital_expenditure": "Capital Expenditure",
    "depreciation_and_amortization": "Reconciled Depreciation",
    "operating_expense": "Operating Expense",
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities Net Minority Interest",
    "current_assets": "Current Assets",
    "current_liabilities": "Current Liabilities",
    "cash_and_equivalents": "Cash And Cash Equivalents",
    "total_debt": "Total Debt",
    "shareholders_equity": "Stockholders Equity",
    "outstanding_shares": "Ordinary Shares Number",
    "earnings_per_share": "Basic EPS",
    "research_and_development": "Research And Development",
    "dividends_and_other_cash_distributions": "Common Stock Dividend",
    "issuance_or_purchase_of_equity_shares": "Issuance Of Capital Stock",
}


def _safe_float(value) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    """Convert a value to int, returning None on failure."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_stmt_value(df: pd.DataFrame, label: str, col) -> float | None:
    """Extract a value from a yfinance statement DataFrame by label and column."""
    if df is None or df.empty:
        return None
    for idx in df.index:
        if str(idx).strip() == label:
            try:
                val = df.loc[idx, col]
                return _safe_float(val)
            except (KeyError, TypeError):
                return None
    return None


def _compute_growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch daily OHLCV prices via yfinance."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached_data]

    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, multi_level_index=False, session=_get_yf_session())
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", ticker, e)
        return []

    if df is None or df.empty:
        return []

    prices: list[Price] = []
    for date, row in df.iterrows():
        prices.append(
            Price(
                open=float(row["Open"]),
                close=float(row["Close"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=int(row["Volume"]),
                time=date.strftime("%Y-%m-%dT%H:%M:%S"),
            )
        )

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)


# ---------------------------------------------------------------------------
# Financial Metrics
# ---------------------------------------------------------------------------

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics via yfinance."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached_data]

    try:
        yf_ticker = yf.Ticker(ticker, session=_get_yf_session())
        info = yf_ticker.info or {}
        # Annual statements; columns are period-end Timestamps
        income_annual = yf_ticker.financials          # rows=items, cols=dates
        balance_annual = yf_ticker.balance_sheet
        cashflow_annual = yf_ticker.cashflow
    except Exception as e:
        logger.warning("yfinance metrics fetch failed for %s: %s", ticker, e)
        return []

    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # Collect valid period columns (those on or before end_date)
    def valid_cols(df):
        if df is None or df.empty:
            return []
        cols = []
        for c in df.columns:
            try:
                c_dt = pd.Timestamp(c).to_pydatetime().replace(tzinfo=None)
                if c_dt <= end_dt:
                    cols.append(c)
            except Exception:
                pass
        return sorted(cols, reverse=True)[:limit]

    income_cols = valid_cols(income_annual)
    balance_cols = valid_cols(balance_annual)
    cashflow_cols = valid_cols(cashflow_annual)

    # We build one FinancialMetrics entry per period. Use the most recent available column.
    # If no statement columns exist before end_date, fall back to info-only (current snapshot).
    all_periods = sorted(set(income_cols + balance_cols + cashflow_cols), reverse=True)[:limit]

    if not all_periods:
        # No historical statements before end_date — return a current-snapshot entry
        all_periods = [None]

    metrics_list: list[FinancialMetrics] = []

    for col in all_periods:
        report_period = col.strftime("%Y-%m-%d") if col is not None else end_date

        def g_inc(label):
            return _get_stmt_value(income_annual, label, col) if col is not None else None

        def g_bal(label):
            return _get_stmt_value(balance_annual, label, col) if col is not None else None

        def g_cf(label):
            return _get_stmt_value(cashflow_annual, label, col) if col is not None else None

        # ---- Raw statement values ----
        revenue = g_inc("Total Revenue")
        gross_profit = g_inc("Gross Profit")
        operating_income = g_inc("Operating Income")
        net_income = g_inc("Net Income")
        ebit = g_inc("EBIT")
        ebitda = g_inc("EBITDA")
        interest_expense = g_inc("Interest Expense")
        total_assets = g_bal("Total Assets")
        total_liabilities = g_bal("Total Liabilities Net Minority Interest")
        shareholders_equity = g_bal("Stockholders Equity")
        total_debt = g_bal("Total Debt")
        current_assets = g_bal("Current Assets")
        current_liabilities = g_bal("Current Liabilities")
        cash = g_bal("Cash And Cash Equivalents")
        shares = g_bal("Ordinary Shares Number") or _safe_float(info.get("sharesOutstanding"))
        free_cash_flow = g_cf("Free Cash Flow")

        # ---- Compute ratios from statements ----
        gross_margin = _safe_float(gross_profit / revenue) if gross_profit and revenue else _safe_float(info.get("grossMargins"))
        operating_margin = _safe_float(operating_income / revenue) if operating_income and revenue else _safe_float(info.get("operatingMargins"))
        net_margin = _safe_float(net_income / revenue) if net_income and revenue else _safe_float(info.get("profitMargins"))
        roe = _safe_float(net_income / shareholders_equity) if net_income and shareholders_equity else _safe_float(info.get("returnOnEquity"))
        roa = _safe_float(net_income / total_assets) if net_income and total_assets else _safe_float(info.get("returnOnAssets"))
        debt_to_equity_raw = _safe_float(info.get("debtToEquity"))
        debt_to_equity = (debt_to_equity_raw / 100) if debt_to_equity_raw is not None else (_safe_float(total_debt / shareholders_equity) if total_debt and shareholders_equity else None)
        debt_to_assets = _safe_float(total_debt / total_assets) if total_debt and total_assets else None
        interest_coverage = _safe_float(ebit / abs(interest_expense)) if ebit and interest_expense and interest_expense != 0 else None
        current_ratio_val = _safe_float(current_assets / current_liabilities) if current_assets and current_liabilities else _safe_float(info.get("currentRatio"))
        quick_ratio_val = _safe_float(info.get("quickRatio"))
        cash_ratio_val = _safe_float(cash / current_liabilities) if cash and current_liabilities else None

        # Market-cap based ratios — always from info (point-in-time unavailable for free)
        market_cap = _safe_float(info.get("marketCap"))
        enterprise_value = _safe_float(info.get("enterpriseValue"))
        pe_ratio = _safe_float(info.get("trailingPE"))
        pb_ratio = _safe_float(info.get("priceToBook"))
        ps_ratio = _safe_float(info.get("priceToSalesTrailing12Months"))
        ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
        ev_revenue = _safe_float(info.get("enterpriseToRevenue"))
        ev_ebit = _safe_float(enterprise_value / ebit) if enterprise_value and ebit else None
        peg_ratio = _safe_float(info.get("pegRatio"))
        fcf_yield = _safe_float(free_cash_flow / market_cap) if free_cash_flow and market_cap else None

        # Per-share metrics
        eps = _safe_float(info.get("trailingEps"))
        bvps = _safe_float(info.get("bookValue"))
        fcf_per_share = _safe_float(free_cash_flow / shares) if free_cash_flow and shares else None

        # ROIC: (net_income - dividends) / (total_debt + equity)
        invested_capital = (total_debt or 0) + (shareholders_equity or 0)
        roic = _safe_float(net_income / invested_capital) if net_income and invested_capital else None

        # Asset & inventory turnover
        asset_turnover = _safe_float(revenue / total_assets) if revenue and total_assets else None

        # Growth rates — need prior period column
        prior_idx = all_periods.index(col) + 1 if col in all_periods else None
        prior_col = all_periods[prior_idx] if prior_idx is not None and prior_idx < len(all_periods) else None

        def g_inc_prior(label):
            return _get_stmt_value(income_annual, label, prior_col) if prior_col is not None else None

        def g_cf_prior(label):
            return _get_stmt_value(cashflow_annual, label, prior_col) if prior_col is not None else None

        def g_bal_prior(label):
            return _get_stmt_value(balance_annual, label, prior_col) if prior_col is not None else None

        rev_prior = g_inc_prior("Total Revenue")
        ni_prior = g_inc_prior("Net Income")
        bv_prior = g_bal_prior("Stockholders Equity")
        fcf_prior = g_cf_prior("Free Cash Flow")
        oi_prior = g_inc_prior("Operating Income")
        ebitda_prior = g_inc_prior("EBITDA")
        eps_prior = _safe_float(info.get("forwardEps"))  # fallback only

        revenue_growth = _compute_growth(revenue, rev_prior) or _safe_float(info.get("revenueGrowth"))
        earnings_growth = _compute_growth(net_income, ni_prior) or _safe_float(info.get("earningsGrowth"))
        bv_growth = _compute_growth(shareholders_equity, bv_prior)
        fcf_growth = _compute_growth(free_cash_flow, fcf_prior)
        oi_growth = _compute_growth(operating_income, oi_prior)
        ebitda_growth = _compute_growth(ebitda, ebitda_prior)
        eps_growth = _safe_float(info.get("earningsGrowth"))

        # OCF ratio
        operating_cf = g_cf("Operating Cash Flow") if col is not None else None
        ocf_ratio = _safe_float(operating_cf / current_liabilities) if operating_cf and current_liabilities else None

        # Receivables / inventory turnover (best-effort)
        receivables = g_bal("Accounts Receivable") if col is not None else None
        inventory = g_bal("Inventory") if col is not None else None
        receivables_turnover = _safe_float(revenue / receivables) if revenue and receivables else None
        inventory_turnover = _safe_float(revenue / inventory) if revenue and inventory else None
        days_sales = _safe_float(365 / receivables_turnover) if receivables_turnover else None
        working_capital = (current_assets or 0) - (current_liabilities or 0) if current_assets and current_liabilities else None
        wc_turnover = _safe_float(revenue / working_capital) if revenue and working_capital else None
        operating_cycle = _safe_float((days_sales or 0) + (365 / inventory_turnover if inventory_turnover else 0)) if days_sales else None

        payout_ratio = _safe_float(info.get("payoutRatio"))

        m = FinancialMetrics(
            ticker=ticker,
            report_period=report_period,
            period=period,
            currency=info.get("currency", "USD"),
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            price_to_earnings_ratio=pe_ratio,
            price_to_book_ratio=pb_ratio,
            price_to_sales_ratio=ps_ratio,
            enterprise_value_to_ebitda_ratio=ev_ebitda,
            enterprise_value_to_revenue_ratio=ev_revenue,
            ev_to_ebit=ev_ebit,
            free_cash_flow_yield=fcf_yield,
            peg_ratio=peg_ratio,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            return_on_equity=roe,
            return_on_assets=roa,
            return_on_invested_capital=roic,
            asset_turnover=asset_turnover,
            inventory_turnover=inventory_turnover,
            receivables_turnover=receivables_turnover,
            days_sales_outstanding=days_sales,
            operating_cycle=operating_cycle,
            working_capital_turnover=wc_turnover,
            current_ratio=current_ratio_val,
            quick_ratio=quick_ratio_val,
            cash_ratio=cash_ratio_val,
            operating_cash_flow_ratio=ocf_ratio,
            debt_to_equity=debt_to_equity,
            debt_to_assets=debt_to_assets,
            interest_coverage=interest_coverage,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            book_value_growth=bv_growth,
            earnings_per_share_growth=eps_growth,
            free_cash_flow_growth=fcf_growth,
            operating_income_growth=oi_growth,
            ebitda_growth=ebitda_growth,
            payout_ratio=payout_ratio,
            earnings_per_share=eps,
            book_value_per_share=bvps,
            free_cash_flow_per_share=fcf_per_share,
        )
        metrics_list.append(m)

    if metrics_list:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics_list])
    return metrics_list


# ---------------------------------------------------------------------------
# Line Items
# ---------------------------------------------------------------------------

def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch financial line items via yfinance statements."""
    try:
        yf_ticker = yf.Ticker(ticker, session=_get_yf_session())
        if period in ("annual", "ttm"):
            income = yf_ticker.financials
            balance = yf_ticker.balance_sheet
            cashflow = yf_ticker.cashflow
        else:
            income = yf_ticker.quarterly_financials
            balance = yf_ticker.quarterly_balance_sheet
            cashflow = yf_ticker.quarterly_cashflow
    except Exception as e:
        logger.warning("yfinance line items fetch failed for %s: %s", ticker, e)
        return []

    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # Merge all statements into one wide dict keyed by period Timestamp
    period_data: dict = {}
    for stmt in [income, balance, cashflow]:
        if stmt is None or stmt.empty:
            continue
        for col in stmt.columns:
            try:
                col_dt = pd.Timestamp(col).to_pydatetime().replace(tzinfo=None)
            except Exception:
                continue
            if col_dt > end_dt:
                continue
            key = col_dt
            if key not in period_data:
                period_data[key] = {}
            for idx in stmt.index:
                val = stmt.loc[idx, col]
                if not (isinstance(val, float) and pd.isna(val)):
                    period_data[key][str(idx).strip()] = val

    if not period_data:
        return []

    sorted_periods = sorted(period_data.keys(), reverse=True)[:limit]

    results: list[LineItem] = []
    for period_dt in sorted_periods:
        data = period_data[period_dt]
        extra_fields: dict = {}
        for item_name in line_items:
            yf_label = _LINE_ITEM_MAP.get(item_name)
            value = None
            if yf_label and yf_label in data:
                value = _safe_float(data[yf_label])
            elif item_name in data:
                value = _safe_float(data[item_name])
            # Special computed: book_value_per_share
            if item_name == "book_value_per_share" and value is None:
                equity = _safe_float(data.get("Stockholders Equity"))
                shares = _safe_float(data.get("Ordinary Shares Number"))
                if equity and shares:
                    value = equity / shares
            # Special computed: ebitda fallback
            if item_name == "ebitda" and value is None:
                ebit = _safe_float(data.get("EBIT"))
                da = _safe_float(data.get("Reconciled Depreciation"))
                if ebit and da:
                    value = ebit + da
            extra_fields[item_name] = value

        li = LineItem(ticker=ticker, report_period=period_dt.strftime("%Y-%m-%d"), period=period, currency="USD", **extra_fields)
        results.append(li)

    return results


# ---------------------------------------------------------------------------
# Insider Trades
# ---------------------------------------------------------------------------

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades via Finnhub."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**t) for t in cached_data]

    # Finnhub requires a from-date; default to 1 year before end_date
    if start_date is None:
        start_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=365)
        from_date = start_dt.strftime("%Y-%m-%d")
    else:
        from_date = start_date

    try:
        client = _get_finnhub_client()
        response = client.stock_insider_transactions(ticker, from_date, end_date)
        raw_trades = response.get("data", []) if isinstance(response, dict) else []
    except Exception as e:
        logger.warning("Finnhub insider trades failed for %s: %s", ticker, e)
        return []

    if not raw_trades:
        return []

    trades: list[InsiderTrade] = []
    for item in raw_trades[:limit]:
        change = item.get("change", 0) or 0
        price = item.get("transactionPrice") or 0.0
        share = item.get("share", 0) or 0
        trades.append(
            InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=item.get("name"),
                title=None,
                is_board_director=None,
                transaction_date=item.get("transactionDate"),
                transaction_shares=float(change),
                transaction_price_per_share=float(price),
                transaction_value=float(change) * float(price),
                shares_owned_before_transaction=float(share - change),
                shares_owned_after_transaction=float(share),
                security_title=None,
                filing_date=item.get("filingDate", end_date),
            )
        )

    if trades:
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
    return trades


# ---------------------------------------------------------------------------
# Company News
# ---------------------------------------------------------------------------

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news via Finnhub."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**n) for n in cached_data]

    # Finnhub requires a from-date; default to 90 days before end_date
    if start_date is None:
        start_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=90)
        from_date = start_dt.strftime("%Y-%m-%d")
    else:
        from_date = start_date

    try:
        client = _get_finnhub_client()
        raw_news = client.company_news(ticker, _from=from_date, to=end_date)
    except Exception as e:
        logger.warning("Finnhub company news failed for %s: %s", ticker, e)
        return []

    if not raw_news:
        return []

    news_list: list[CompanyNews] = []
    for item in raw_news[:limit]:
        ts = item.get("datetime", 0)
        try:
            date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            date_str = end_date
        news_list.append(
            CompanyNews(
                ticker=ticker,
                title=item.get("headline", ""),
                author=None,
                source=item.get("source", ""),
                date=date_str,
                url=item.get("url", ""),
                sentiment=None,
            )
        )

    if news_list:
        _cache.set_company_news(cache_key, [n.model_dump() for n in news_list])
    return news_list


# ---------------------------------------------------------------------------
# Market Cap
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ETF / Mutual Fund Data (Finnhub + Tiingo + yfinance)
# ---------------------------------------------------------------------------

def detect_fund_type(symbol: str) -> str:
    """Return 'ETF', 'MUTUALFUND', or 'EQUITY' based on yfinance quoteType."""
    cache_key = f"fund_type_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached.get("type", "EQUITY")
    try:
        info = yf.Ticker(symbol, session=_get_yf_session()).info or {}
        fund_type = (info.get("quoteType") or "EQUITY").upper()
        if fund_type not in ("ETF", "MUTUALFUND"):
            fund_type = "EQUITY"
    except Exception:
        fund_type = "EQUITY"
    _cache.set_fund_data(cache_key, {"type": fund_type})
    return fund_type


def get_etf_profile(symbol: str) -> dict:
    """Fetch ETF profile via Finnhub (etfs_profile)."""
    cache_key = f"etf_profile_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached
    try:
        result = _get_finnhub_client().etfs_profile(symbol=symbol)
        data = result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("Finnhub ETF profile failed for %s: %s", symbol, e)
        return {}
    if data:
        _cache.set_fund_data(cache_key, data)
    return data


def get_etf_holdings(symbol: str) -> dict:
    """Fetch ETF holdings via Finnhub (etfs_holdings)."""
    cache_key = f"etf_holdings_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached
    try:
        result = _get_finnhub_client().etfs_holdings(symbol=symbol)
        data = result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("Finnhub ETF holdings failed for %s: %s", symbol, e)
        return {}
    if data:
        _cache.set_fund_data(cache_key, data)
    return data


def get_etf_sector(symbol: str) -> dict:
    """Fetch ETF sector exposures via Finnhub (etfs_sector_exp)."""
    cache_key = f"etf_sector_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached
    try:
        result = _get_finnhub_client().etfs_sector_exp(symbol=symbol)
        data = result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("Finnhub ETF sector failed for %s: %s", symbol, e)
        return {}
    if data:
        _cache.set_fund_data(cache_key, data)
    return data


def get_etf_country(symbol: str) -> dict:
    """Fetch ETF country exposures via Finnhub (etfs_country_exp)."""
    cache_key = f"etf_country_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached
    try:
        result = _get_finnhub_client().etfs_country_exp(symbol=symbol)
        data = result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("Finnhub ETF country failed for %s: %s", symbol, e)
        return {}
    if data:
        _cache.set_fund_data(cache_key, data)
    return data


def get_mutual_fund_profile(symbol: str) -> dict:
    """Fetch mutual fund profile via Finnhub (mutual_fund_profile)."""
    cache_key = f"mf_profile_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached
    try:
        result = _get_finnhub_client().mutual_fund_profile(symbol=symbol)
        data = result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("Finnhub mutual fund profile failed for %s: %s", symbol, e)
        return {}
    if data:
        _cache.set_fund_data(cache_key, data)
    return data


def get_mutual_fund_holdings(symbol: str) -> dict:
    """Fetch mutual fund holdings via Finnhub (mutual_fund_holdings)."""
    cache_key = f"mf_holdings_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached
    try:
        result = _get_finnhub_client().mutual_fund_holdings(symbol=symbol)
        data = result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("Finnhub mutual fund holdings failed for %s: %s", symbol, e)
        return {}
    if data:
        _cache.set_fund_data(cache_key, data)
    return data


def get_fund_price_history(symbol: str) -> list:
    """Fetch 1-year daily price history from Tiingo. Returns [] if TIINGO_API_KEY is not set."""
    tiingo_key = os.environ.get("TIINGO_API_KEY", "")
    if not tiingo_key:
        return []
    cache_key = f"fund_price_history_{symbol}"
    if cached := _cache.get_fund_data(cache_key):
        return cached
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    try:
        resp = _get_yf_session().get(url, params={"startDate": start_date, "token": tiingo_key})
        data = resp.json() if resp.status_code == 200 else []
        if not isinstance(data, list):
            data = []
    except Exception as e:
        logger.warning("Tiingo price history failed for %s: %s", symbol, e)
        return []
    if data:
        _cache.set_fund_data(cache_key, data)
    return data


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap via yfinance."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        info = yf.Ticker(ticker, session=_get_yf_session()).info or {}
    except Exception as e:
        logger.warning("yfinance market cap fetch failed for %s: %s", ticker, e)
        return None

    if end_date == today:
        return _safe_float(info.get("marketCap"))

    # Historical approximation: price on end_date × shares outstanding
    shares = _safe_float(info.get("sharesOutstanding"))
    if not shares:
        return _safe_float(info.get("marketCap"))

    prices = get_prices(ticker, end_date, end_date)
    if not prices:
        # Try fetching a small window around end_date
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        start_approx = (end_dt - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        prices = get_prices(ticker, start_approx, end_date)

    if not prices:
        return _safe_float(info.get("marketCap"))

    close_price = prices[-1].close
    return close_price * shares
