from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock, patch
from urllib.parse import urlparse
import unittest

from frontend.btc_data import COINBASE_SPOT_PRICE_URL, fetch_spot_btc_price


class TestBtcSpotLinkQuality(unittest.TestCase):
    def test_spot_url_is_https_and_expected_coinbase_endpoint(self) -> None:
        parsed = urlparse(COINBASE_SPOT_PRICE_URL)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "api.coinbase.com")
        self.assertEqual(parsed.path, "/v2/prices/BTC-USD/spot")

    @patch("frontend.btc_data.requests.get")
    def test_fetch_spot_price_calls_link_without_errors(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": {"amount": "12345.67"}}
        mock_get.return_value = mock_response

        amount, updated_at = fetch_spot_btc_price()

        mock_get.assert_called_once_with(COINBASE_SPOT_PRICE_URL, timeout=10)
        mock_response.raise_for_status.assert_called_once_with()
        self.assertEqual(amount, 12345.67)
        self.assertIsInstance(updated_at, datetime)


if __name__ == "__main__":
    unittest.main()
