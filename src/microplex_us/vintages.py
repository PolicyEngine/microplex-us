"""Single source of truth for the source vintages a built dataset uses.

A :class:`DatasetProfile` declares, in ONE place, the model year a dataset
represents and the exact source *release* feeding each input, plus how that
release's dollars reach the model year (used natively, or aged with a
component-specific factor family).

Build code reads vintages from a profile instead of per-call literal defaults,
so a stale year cannot hide in a function signature, a CLI default, or a
forgotten shell flag: the value is defined once and the safe path is the only
path. (The motivating bug: ``cps_source_year`` defaulted to 2023 -- income year
2022 -- while every production build overrode it to 2025; the stale literal sat
in three signatures for who knows how long because nothing failed.)

The coherence checks here are the spec the build must satisfy: every source must
reach ``model_year`` -- either it is native to that year (``income_year ==
model_year``) or it declares an ``age_to == model_year`` aging step. A source
that does not yet reach the model year must declare a ``gap_reason`` so the gap
is explicit rather than silent. A future build-time gate verifies a produced
artifact against the active profile; this module guarantees the *profile itself*
is internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Release:
    """One source release and how its dollars reach a model year.

    Attributes:
        release: the survey/file release actually loaded (e.g. CPS ASEC 2025).
        income_year: the calendar/income year that release represents. CPS ASEC
            survey year ``Y`` covers income year ``Y - 1`` (ASEC 2025 -> 2024);
            most other sources have ``release == income_year``.
        age_to: when set, dollar variables are aged from ``income_year`` to this
            year using ``factors``; when ``None`` the release is used on its
            native basis.
        factors: label of the component-specific growth-factor family used when
            aging (e.g. ``"soi"``). Required iff ``age_to`` is set; the build
            binds the label to the actual factor implementation.
        gap_reason: explicit, temporary acknowledgement that this source does not
            yet reach the model year (e.g. aging not wired). Lets a profile stay
            honest about a known gap without silently passing coherence.
    """

    release: int
    income_year: int
    age_to: int | None = None
    factors: str | None = None
    gap_reason: str | None = None

    def __post_init__(self) -> None:
        if self.age_to is not None and self.factors is None:
            raise ValueError(
                f"Release(release={self.release}) sets age_to={self.age_to} but "
                "no `factors` family to age with."
            )
        if self.age_to is None and self.factors is not None:
            raise ValueError(
                f"Release(release={self.release}) sets factors={self.factors!r} "
                "but no `age_to` year to age toward."
            )
        if self.age_to is not None and self.age_to < self.income_year:
            raise ValueError(
                f"Release(release={self.release}) ages backward: age_to="
                f"{self.age_to} < income_year={self.income_year}."
            )

    @property
    def effective_year(self) -> int:
        """The model year this release's dollars land on after any aging."""
        return self.age_to if self.age_to is not None else self.income_year


@dataclass(frozen=True)
class DatasetProfile:
    """The complete vintage definition for one built dataset.

    ``model_year`` is the year the dataset represents; every source must reach it
    (or declare a ``gap_reason``).
    """

    name: str
    model_year: int
    cps_asec: Release
    puf: Release
    acs: Release
    sipp: Release
    scf: Release

    def sources(self) -> dict[str, Release]:
        return {
            "cps_asec": self.cps_asec,
            "puf": self.puf,
            "acs": self.acs,
            "sipp": self.sipp,
            "scf": self.scf,
        }

    def incoherent_sources(self) -> dict[str, str]:
        """Map each source that fails to reach ``model_year`` (and has not
        declared a ``gap_reason``) to a human-readable explanation."""
        problems: dict[str, str] = {}
        for name, release in self.sources().items():
            if release.gap_reason is not None:
                continue
            if release.effective_year != self.model_year:
                problems[name] = (
                    f"reaches {release.effective_year} (release {release.release}, "
                    f"income {release.income_year}, age_to {release.age_to}); "
                    f"model_year is {self.model_year}"
                )
        return problems

    def declared_gaps(self) -> dict[str, str]:
        """Map each source with a declared (acknowledged) basis gap to its reason."""
        return {
            name: release.gap_reason
            for name, release in self.sources().items()
            if release.gap_reason is not None
        }

    def __post_init__(self) -> None:
        problems = self.incoherent_sources()
        if problems:
            detail = "; ".join(f"{name}: {why}" for name, why in problems.items())
            raise ValueError(
                f"DatasetProfile {self.name!r} is incoherent: every source must "
                f"reach model_year {self.model_year} or declare a gap_reason. {detail}"
            )


# The current Microplex eCPS-replacement target: a 2024 base dataset that
# replaces ``enhanced_cps_2024``. Source releases match what the production build
# loads today; the aging declarations are the spec the build satisfies (PUF ages
# via SOI factors; SIPP/SCF aging to 2024 landed in #185; ACS donor is now the
# native-2024 release).
MP_2024 = DatasetProfile(
    name="mp_2024",
    model_year=2024,
    # CPS ASEC survey year 2025 == income/calendar year 2024: native 2024 spine.
    cps_asec=Release(release=2025, income_year=2024),
    # Public-use PUF base is 2015 (latest released); aged to 2024 via SOI factors.
    puf=Release(release=2015, income_year=2015, age_to=2024, factors="soi"),
    # ACS donor at the native-2024 release.
    acs=Release(release=2024, income_year=2024),
    # SIPP/SCF donors aged from their latest releases to 2024.
    sipp=Release(release=2023, income_year=2023, age_to=2024, factors="pe_growfactors"),
    scf=Release(release=2022, income_year=2022, age_to=2024, factors="pe_growfactors"),
)


PROFILES: dict[str, DatasetProfile] = {MP_2024.name: MP_2024}


def get_profile(name: str) -> DatasetProfile:
    """Return the named dataset profile, or raise with the known names."""
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown dataset profile {name!r}; known profiles: {known}")
