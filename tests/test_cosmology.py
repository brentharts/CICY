"""
Tests for pyCICY.theories.cosmology and cosmobase: the registry entries for cosmological singularities.

Organised around the ledger, like test_theories.py:

  [1] registry     the four entries are registered, are not CICY compactifications, and describe themselves
  [2] exact        exponents in exact rational arithmetic, K by three routes, the BKL orbit, K(0) of the pancake
  [3] absent       a finite-curvature vacuum pancake does not exist (NoSuchTheory), and it is the *flat* (1, 0, 0)
  [4] declined     no cosmology is compared with data: NeedsFit, with what is missing named
  [5] horizons     the entries report the horizon statements the modules prove

Run with:  python3 tests/test_cosmology.py
"""
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyCICY import theories as T
from pyCICY import kasner as qk


# ---------------------------------------------------------------- [1] registry

KEYS = ["kasner-vacuum", "mixmaster-bkl", "pancake-bigbang", "flrw-toys"]


@pytest.mark.parametrize("key", KEYS)
def test_registered_and_not_a_compactification(key):
    cls = T.get(key)
    assert issubclass(cls, T.CosmologyTheory) and not issubclass(cls, T.Theory)
    text = cls().describe()
    assert "exact:" in text and "declined:" in text


def test_the_base_class_declines_every_physical_prediction():
    for key in KEYS:
        with pytest.raises(T.NeedsFit) as e:
            T.get(key)().physical_prediction()
        assert len(e.value.missing) >= 2


def test_exact_and_declined_lists_are_not_empty():
    for key in KEYS:
        th = T.get(key)()
        assert len(th.exact_content()) >= 4 and len(th.declined()) >= 2


# ---------------------------------------------------------------- [2] exact

def test_kasner_exponents_are_exact_for_rational_u():
    k = T.KasnerVacuum(2)
    p = k.exponents()
    assert p == (Fraction(-2, 7), Fraction(3, 7), Fraction(6, 7))
    assert sum(p) == 1 and sum(x * x for x in p) == 1
    assert k.curvature() == Fraction(576, 343)
    assert k.singularity_type() == "cigar"


def test_kasner_curvature_three_routes_agree():
    v = T.KasnerVacuum(Fraction(3, 2)).verify_curvature(einsteinpy=qk.has_einsteinpy())
    assert v["closed"] == v["sympy"] and v["sympy_ricci_zero"]
    if qk.has_einsteinpy():
        assert v["einsteinpy"] == v["closed"] and v["einsteinpy_ricci_zero"]


def test_flat_kasner_is_not_a_curvature_singularity():
    assert T.KasnerVacuum(10 ** 9).curvature() == pytest.approx(0.0, abs=1e-6)


def test_bkl_orbit_is_exact_and_terminates_for_rational_u():
    m = T.MixmasterBKL("37/10")
    ep = m.epochs(40)
    assert [str(e["u"]) for e in ep[:4]] == ["37/10", "27/10", "17/10", "10/7"]
    assert all(isinstance(e["u"], Fraction) for e in ep)
    assert str(ep[-1]["u"]) == "1" and len(ep) < 40                    # rational orbit reaches u = 1
    for e in ep:                                                       # K t^4 = 16 u^2 (1+u)^2 / (1+u+u^2)^3
        u = e["u"]
        assert e["K_coefficient"] == 16 * u * u * (1 + u) ** 2 / (1 + u + u * u) ** 3


def test_pancake_curvature_at_zero_matches_the_series_and_has_a0_dependence():
    a = T.PancakeBigBang(a0=1.0, b0=3.0, f0=2.0, m=0.5)
    b = T.PancakeBigBang(a0=1.2, b0=3.0, f0=2.0, m=0.5)
    assert a.curvature() == pytest.approx(16 * (0.25 * 0.25 * 4 - 2 / 9.0) ** 2 + 80 / 81.0, rel=1e-13)
    assert b.curvature() != pytest.approx(a.curvature(), rel=1e-3)     # K(0) depends on the opening speed a0
    assert a.singularity_type() == "pancake"
    c = a.series()
    assert c["a2"] == pytest.approx(2 * 1.0 / (3 * 9.0), rel=1e-14)


def test_flrw_models_and_their_laws():
    f = T.FLRWToys()
    assert f.second_law("lcdm") and f.second_law("toy") and f.second_law("tsallis") and f.second_law("pade")
    assert f.bounce_curvature(rho_c=10.0, h0=1.0) == 2700.0
    a = np.geomspace(0.1, 1.0, 20)
    assert np.allclose(f.model("toy", a_t=1e-9)(a), f.model("lcdm")(a), rtol=1e-12)    # a_t -> 0 recovers LCDM
    assert np.allclose(f.model("ccbh", eps=0.0)(a), f.model("lcdm")(a), rtol=1e-6)     # eps = 0 is LCDM
    with pytest.raises(KeyError):
        f.model("nonsense")


def test_the_bounce_violates_the_second_law_just_after_the_bounce():
    """H' > 0 there (q < -1): an exact statement about the toy, not a defect of the criterion."""
    assert not T.FLRWToys().second_law("bounce")
    assert T.FLRWToys().singularity_type("bounce") == "none"


# ---------------------------------------------------------------- [3] absent

def test_a_finite_curvature_vacuum_pancake_does_not_exist():
    with pytest.raises(T.NoSuchTheory) as e:
        T.PancakeBigBang().vacuum_pancake()
    assert "flat" in str(e.value)
    assert qk.is_flat_kasner((1.0, 0.0, 0.0)) and qk.singularity_type((1.0, 0.0, 0.0)) == "pancake"


# ---------------------------------------------------------------- [4] declined

def test_fitting_a_toy_is_declined_with_what_is_missing():
    with pytest.raises(T.NeedsFit) as e:
        T.FLRWToys().fit_to_data()
    assert any("likelihood" in m for m in e.value.missing)


# ---------------------------------------------------------------- [5] horizons

def test_horizon_statements():
    assert T.KasnerVacuum(2).horizon()["types"] == ("finite", "finite", "finite")
    h = T.PancakeBigBang().horizon()
    assert h["fibre_log_coefficient"] == pytest.approx(2.0)
    assert 4.5 < h["base_contact_time"] < 5.8
    m = T.MixmasterBKL().horizon(n_epochs=20)
    lh = np.array(m["log10_h"])
    assert np.all(np.isfinite(lh[-1])) and np.allclose(lh[9], lh[-1], atol=1e-6)    # saturates: finite horizon


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
