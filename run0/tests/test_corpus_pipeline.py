from __future__ import annotations

import unittest

from deepvoice_run0.corpus_pipeline import _fma_license_allowed, _instrumental_row


class CorpusPipelineTests(unittest.TestCase):
    def test_license_filter_requires_an_explicit_adaptable_cc_license(self) -> None:
        self.assertFalse(_fma_license_allowed("Music by Example Artist"))
        self.assertFalse(_fma_license_allowed("https://creativecommons.org/licenses/by-nc-nd/4.0/"))
        self.assertTrue(_fma_license_allowed("https://creativecommons.org/licenses/by-nc/4.0/"))

    def test_instrumental_filter_requires_license_and_no_vocals(self) -> None:
        valid = {
            "attribution": "Artist: Example\nLicense: https://creativecommons.org/licenses/by/4.0/",
            "description": "Vocal content: none",
            "genre_tags": [],
        }
        self.assertTrue(_instrumental_row(valid))
        self.assertFalse(_instrumental_row(valid | {"description": "Vocal content: present"}))
        self.assertFalse(_instrumental_row(valid | {"attribution": "Music by Example"}))


if __name__ == "__main__":
    unittest.main()
