"""Tests for the one-to-one resolver, per-entity F0.5 helper, and v2 features."""

import unittest

import numpy as np

from src.matching import pair_features as pf
from src.matching import selection as sel


class TestResolveOneToOne(unittest.TestCase):
    def test_contested_claim_goes_to_higher_prob(self):
        claims = [("a", "x", 0.9), ("b", "x", 0.85), ("b", "y", 0.8), ("c", "z", 0.7)]
        self.assertEqual(sel.resolve_one_to_one(claims),
                         {("a", "x"), ("b", "y"), ("c", "z")})

    def test_uncontested_passes_through(self):
        self.assertEqual(sel.resolve_one_to_one([("a", "x", 0.9)]), {("a", "x")})

    def test_tie_is_deterministic(self):
        claims = [("a", "x", 0.9), ("b", "x", 0.9)]
        self.assertEqual(sel.resolve_one_to_one(claims), {("a", "x")})

    def test_every_s1_keeps_a_claim_when_possible(self):
        claims = [("a", "x", 0.9), ("b", "x", 0.85), ("b", "y", 0.6)]
        self.assertEqual(sel.resolve_one_to_one(claims), {("a", "x"), ("b", "y")})


class TestOneToOneEval(unittest.TestCase):
    def _matrices(self):
        # E1: c1 (p=.9, true). E2: c1 (p=.85, FALSE — c1 belongs to E1), c2 (p=.8, true)
        P = np.array([[0.9, sel.ABSENT], [0.85, 0.8]], dtype=np.float32)
        T = np.array([[True, False], [False, True]])
        cmat = np.array([["c1", ""], ["c1", "c2"]], dtype=object)
        true_total = np.array([1, 1])
        return P, T, true_total, cmat

    def test_dedup_improves_f05(self):
        P, T, true_total, cmat = self._matrices()
        params = dict(tau_singleton=0.68, tau_match=0.60, gamma_rel=0.70)
        f_plain, _, _ = sel.evaluate_selection(P, T, true_total, **params)
        f_dedup, n_claims, n_kept = sel.apply_one_to_one_f05(
            P, T, true_total, cmat, ["E1", "E2"], **params)
        # plain: E2 keeps its FP claim on c1 -> (1.0 + 0.5556) / 2
        self.assertAlmostEqual(f_plain, (1.0 + 0.5556) / 2, places=3)
        # dedup: E2's contested claim dropped, c2 survives -> both perfect
        self.assertAlmostEqual(f_dedup, 1.0, places=6)
        self.assertEqual((n_claims, n_kept), (3, 2))

    def test_per_entity_f05_matches_evaluate(self):
        P, T, true_total, _ = self._matrices()
        params = dict(tau_singleton=0.68, tau_match=0.60, gamma_rel=0.70)
        f_e, _, _, _, _ = sel.per_entity_f05(P, T, true_total, **params)
        f_mean, _, _ = sel.evaluate_selection(P, T, true_total, **params)
        self.assertAlmostEqual(float(f_e.mean()), f_mean, places=6)


class TestSelectMatches(unittest.TestCase):
    def test_return_probs(self):
        cands = [{"entity_id": "x", "prob": 0.9}, {"entity_id": "y", "prob": 0.65}]
        out = sel.select_matches(cands, tau_singleton=0.68, tau_match=0.60,
                                 gamma_rel=0.70, return_probs=True)
        self.assertEqual(out, [("x", 0.9), ("y", 0.65)])


class TestV2Features(unittest.TestCase):
    JOB = ("Alpha Zylker", "12 Main St 94016",
           "Alpha Zylker", "12 Main St 94016",
           (0.8, 1, 1.0, 0.0, 0.0, "S2-1"))

    def test_feature_count_and_names(self):
        self.assertEqual(len(pf.FEATURE_NAMES), 42)
        self.assertEqual(pf.FEATURE_NAMES[-4:],
                         ["skel_concat_ratio", "long_num_exact",
                          "addr_housenum_exact", "name_idf_soft_jaccard"])

    def test_identical_pair_flags(self):
        pf.set_idf({}, 1.0)
        a = pf.pair_vec(self.JOB)
        self.assertEqual(a[39], 1.0)  # shared 5-digit number
        self.assertEqual(a[40], 1.0)  # house number 12 == 12
        self.assertAlmostEqual(a[41], 1.0, places=6)

    def test_idf_weighting_rare_tokens(self):
        # shared token "alpha" is rare (high idf) -> soft jaccard above plain jaccard
        job = ("Alpha Zylker", "", "Alpha Group", "",
               (0.8, 1, 1.0, 0.0, 0.0, "S2-1"))
        pf.set_idf({}, 1.0)
        plain = pf.pair_vec(job)[41]
        pf.set_idf({"alpha": 15.0, "group": 2.0}, 15.0)
        weighted = pf.pair_vec(job)[41]
        self.assertAlmostEqual(plain, 1 / 3, places=6)
        self.assertAlmostEqual(weighted, 15.0 / 32.0, places=6)
        self.assertGreater(weighted, plain)

    def test_unweighted_fallback_equals_jaccard(self):
        pf.set_idf({}, 1.0)
        job = ("Alpha Zylker", "", "Alpha Group", "", (0.8, 1, 1.0, 0.0, 0.0, "S2-1"))
        a = pf.pair_vec(job)
        self.assertAlmostEqual(a[41], a[17], places=6)  # == core_token_jaccard


if __name__ == "__main__":
    unittest.main()
