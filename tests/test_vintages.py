"""Tests for the single-source-of-truth dataset vintage profiles."""

import pytest

from microplex_us.vintages import (
    MP_2024,
    DatasetProfile,
    Release,
    get_profile,
)


def _release(**overrides) -> Release:
    base = dict(release=2024, income_year=2024)
    base.update(overrides)
    return Release(**base)


def _profile(**overrides) -> DatasetProfile:
    base = dict(
        name="test",
        model_year=2024,
        cps_asec=Release(release=2025, income_year=2024),
        puf=Release(release=2015, income_year=2015, age_to=2024, factors="soi"),
        acs=Release(release=2024, income_year=2024),
        sipp=Release(release=2023, income_year=2023, age_to=2024, factors="g"),
        scf=Release(release=2022, income_year=2022, age_to=2024, factors="g"),
    )
    base.update(overrides)
    return DatasetProfile(**base)


# --- Release ---------------------------------------------------------------


def test_release_effective_year_native_vs_aged():
    assert Release(release=2025, income_year=2024).effective_year == 2024
    assert (
        Release(release=2015, income_year=2015, age_to=2024, factors="soi").effective_year
        == 2024
    )


def test_release_age_to_requires_factors():
    with pytest.raises(ValueError, match="factors"):
        Release(release=2015, income_year=2015, age_to=2024)


def test_release_factors_requires_age_to():
    with pytest.raises(ValueError, match="age_to"):
        Release(release=2015, income_year=2015, factors="soi")


def test_release_cannot_age_backward():
    with pytest.raises(ValueError, match="ages backward"):
        Release(release=2015, income_year=2024, age_to=2022, factors="soi")


# --- DatasetProfile coherence ---------------------------------------------


def test_mp_2024_is_coherent_and_has_no_gaps():
    assert MP_2024.model_year == 2024
    assert MP_2024.incoherent_sources() == {}
    assert MP_2024.declared_gaps() == {}
    # Every source's dollars land on the model year.
    for name, release in MP_2024.sources().items():
        assert release.effective_year == 2024, name


def test_mp_2024_cps_spine_is_native_2024_income():
    # ASEC survey year 2025 -> income year 2024; the spine is native, not aged.
    assert MP_2024.cps_asec.release == 2025
    assert MP_2024.cps_asec.income_year == 2024
    assert MP_2024.cps_asec.age_to is None


def test_incoherent_profile_raises():
    # A donor stuck at 2022 with no aging and no acknowledged gap is incoherent.
    with pytest.raises(ValueError, match="incoherent"):
        _profile(scf=Release(release=2022, income_year=2022))


def test_declared_gap_passes_coherence_but_is_surfaced():
    profile = _profile(
        scf=Release(release=2022, income_year=2022, gap_reason="aging not wired yet")
    )
    assert profile.incoherent_sources() == {}
    assert profile.declared_gaps() == {"scf": "aging not wired yet"}


def test_aged_source_reaching_wrong_year_is_incoherent():
    with pytest.raises(ValueError, match="incoherent"):
        _profile(sipp=Release(release=2023, income_year=2023, age_to=2023, factors="g"))


# --- registry --------------------------------------------------------------


def test_get_profile_returns_known_and_raises_unknown():
    assert get_profile("mp_2024") is MP_2024
    with pytest.raises(KeyError, match="Unknown dataset profile"):
        get_profile("mp_1999")
