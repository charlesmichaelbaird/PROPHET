"""BTC market data helpers for the Streamlit dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st

COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINBASE_SPOT_PRICE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"


def _moving_average(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = []
    rolling_sum = 0.0
    for idx, value in enumerate(values):
        rolling_sum += value
        if idx >= window:
            rolling_sum -= values[idx - window]
        if idx + 1 >= window:
            result.append(round(rolling_sum / window, 2))
        else:
            result.append(None)
    return result


@st.cache_data(ttl=60 * 60)
def fetch_btc_history(limit: int = 365) -> list[dict[str, Any]]:
    """Fetch daily BTC-USD close prices from CoinGecko market chart data."""
    response = requests.get(
        COINGECKO_MARKET_CHART_URL,
        params={"vs_currency": "usd", "days": max(limit, 120), "interval": "daily"},
        timeout=15,
    )
    response.raise_for_status()
    raw_prices = response.json().get("prices", [])

    rows: list[dict[str, Any]] = []
    closes: list[float] = []
    for point in raw_prices[-limit:]:
        close_ts = int(point[0])
        close_price = float(point[1])
        closes.append(close_price)
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(close_ts / 1000, tz=timezone.utc),
                "close": close_price,
            }
        )

    ma_10 = _moving_average(closes, 10)
    ma_30 = _moving_average(closes, 30)
    ma_100 = _moving_average(closes, 100)

    for idx, row in enumerate(rows):
        row["ma_10"] = ma_10[idx]
        row["ma_30"] = ma_30[idx]
        row["ma_100"] = ma_100[idx]

    return rows


def fetch_spot_btc_price() -> tuple[float, datetime]:
    """Fetch live-ish BTC-USD spot price from Coinbase."""
    response = requests.get(COINBASE_SPOT_PRICE_URL, timeout=10)
    response.raise_for_status()
    payload = response.json()
    amount = float(payload["data"]["amount"])
    return amount, datetime.now(timezone.utc)
