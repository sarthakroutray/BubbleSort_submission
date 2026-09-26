"""Unit tests locking the §3.4 data traps and §5.2 normalization behaviour."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.cleaning import text_cleaner as tc


class TestNormalization(unittest.TestCase):
    def test_zero_for_o(self):
        self.assertIn("foods", tc.normalize("Hitech F0ods Bakery").split())
        self.assertEqual(tc.normalize("F0ods"), "foods")

    def test_leading_zeros(self):
        self.assertEqual(tc.normalize("002050"), "2050")
        self.assertEqual(tc.numbers("0000000305/1A"), ["305", "1"])

    def test_ordinals(self):
        self.assertEqual(tc.normalize("2Nd Floor"), "2 floor")
        self.assertEqual(tc.address_words("2Nd Floor"), ["2", "fl"])
        self.assertEqual(tc.normalize("158ND"), "158")
        self.assertEqual(tc.normalize("1st Street"), "1 street")
        self.assertEqual(tc.address_words("1st Street"), ["1", "st"])

    def test_letter_digit_split(self):
        self.assertEqual(tc.normalize("Shop4"), "shop 4")
        self.assertEqual(tc.normalize("158ND"), "158")

    def test_word_numerals(self):
        self.assertEqual(tc.normalize("One Two Three"), "1 2 3")

    def test_ampersand_fold(self):
        self.assertEqual(tc.normalize("AT&T"), "at and t")

    def test_quotes_and_punct_stripped(self):
        self.assertEqual(tc.normalize('"""ehpad Club SAS"'), "ehpad club sas")


class TestTransliteration(unittest.TestCase):
    def test_devanagari(self):
        raw = tc.normalize("तिरुपति फाइनेंस")
        self.assertTrue(raw.isascii() and raw, raw)
        # the whole point: transliteration + skeleton collapses script drift
        self.assertEqual(tc.skeleton_text("तिरुपति फाइनेंस"),
                         tc.skeleton_text("Tirupati Finance"))

    def test_accented(self):
        raw, _ = tc.name_views("Société Générale")
        self.assertEqual(raw, "societe generale")

    def test_ascii_fast_path_is_identity_on_ascii(self):
        self.assertEqual(tc.transliterate("Plain Ascii 123"), "Plain Ascii 123")


class TestSkeleton(unittest.TestCase):
    def test_tirupati_variants_collapse(self):
        self.assertEqual(tc.skeleton("tirupti"), tc.skeleton("tirupati"))
        self.assertEqual(tc.skeleton("tirupti"), "trpt")

    def test_documented_substitutions(self):
        self.assertEqual(tc.skeleton("pharma"), "frn")
        self.assertEqual(tc.skeleton("bright"), "brt")

    def test_skeleton_text(self):
        self.assertEqual(tc.skeleton_text("Tirupati Finance"),
                         f"{tc.skeleton('tirupati')} {tc.skeleton('finance')}")


class TestNameViews(unittest.TestCase):
    def test_legal_suffix_core(self):
        raw, core = tc.name_views("Tata Consultancy Services Ltd")
        self.assertEqual(raw, "tata consultancy services ltd")
        self.assertEqual(core, "tata consultancy services")

    def test_repeated_suffix_strip(self):
        _, core = tc.name_views("Acme Pvt Ltd")
        self.assertEqual(core, "acme")

    def test_core_never_empty(self):
        _, core = tc.name_views("Ltd")
        self.assertTrue(core)

    def test_france_legal_form(self):
        _, core = tc.name_views("Marina Ecole France Sarl")
        self.assertEqual(core, "marina ecole france")


class TestAddress(unittest.TestCase):
    def test_street_folds(self):
        words = tc.address_words("175 Boulevard du Président Franklin Roosevelt")
        self.assertIn("blvd", words)

    def test_number_word_pairs(self):
        self.assertEqual(tc.number_word_pairs("203 26th Street"), ["203_26", "26_st"])

    def test_france_abbreviations(self):
        words = tc.address_words("26 R DNEIS CORDONNIER")
        self.assertIn("r", words)


class TestCsvTraps(unittest.TestCase):
    """§3.4: literal N/A + embedded \"\"\" quoting must survive ingestion."""

    def _read(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.tsv"
            p.write_text(text, encoding="utf-8")
            it = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=250_000, quoting=csv.QUOTE_MINIMAL)
            df = next(iter(it))
            it.close()
            return df

    def test_literal_na_not_nan(self):
        df = self._read("entity_id\tbusiness_address\nS1-1\tN/A\n")
        self.assertEqual(df.loc[0, "business_address"], "N/A")

    def test_triple_quote_field_parses(self):
        df = self._read('entity_id\tbusiness_name\nS1-2\t"""ehpad Club SAS"\n')
        self.assertIn("ehpad", df.loc[0, "business_name"])
        self.assertEqual(df.loc[0, "business_name"], '"ehpad Club SAS')

    def test_comma_in_quoted_field(self):
        df = self._read(
            'entity_id\tbusiness_address\nS1-3\t"110, MG Road, Pune"\n')
        self.assertEqual(df.loc[0, "business_address"], "110, MG Road, Pune")


if __name__ == "__main__":
    unittest.main()
