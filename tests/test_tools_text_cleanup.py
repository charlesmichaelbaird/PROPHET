from __future__ import annotations

import unittest

from mcp_server.tools import extract_article_text


class TestArticleTextCleanup(unittest.TestCase):
    def test_strips_ap_photo_caption_prefix_and_keeps_following_text(self) -> None:
        html = """
        <html><body>
          <p>A person walks near a building (AP Photo/Jane Doe) The city announced new policy changes today.</p>
          <p>Officials said implementation begins next month.</p>
        </body></html>
        """

        text = extract_article_text(html)

        self.assertNotIn("AP Photo/Jane Doe", text)
        self.assertTrue(text.startswith("The city announced new policy changes today."))
        self.assertIn("Officials said implementation begins next month.", text)

    def test_removes_caption_only_paragraph(self) -> None:
        html = """
        <html><body>
          <p>A firefighter sprays water on a home (AP Photo/John Smith)</p>
          <p>Residents were evacuated safely.</p>
        </body></html>
        """

        text = extract_article_text(html)

        self.assertEqual(text, "Residents were evacuated safely.")


if __name__ == "__main__":
    unittest.main()
