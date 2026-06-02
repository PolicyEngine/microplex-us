import zipfile

import pandas as pd
import polars as pl

from microplex_us.data_sources.cps import load_cps_asec
from microplex_us.data_sources.cps_age import randomize_cps_topcoded_age_80_84
from microplex_us.data_sources.cps_mappings import map_age


def test_randomize_cps_topcoded_age_80_84_is_deterministic():
    frame = pl.DataFrame({"age": [79, *([80] * 20), 85]})

    first = randomize_cps_topcoded_age_80_84(frame)
    second = randomize_cps_topcoded_age_80_84(frame)
    ages = first["age"].to_list()

    assert ages == second["age"].to_list()
    assert ages[0] == 79
    assert ages[-1] == 85
    assert set(ages[1:-1]).issubset({80, 81, 82, 83, 84})
    assert len(set(ages[1:-1])) > 1


def test_map_age_spreads_cps_80_to_80_84():
    frame = pl.DataFrame({"A_AGE": [79, *([80] * 20), 85]})

    result = map_age(frame)
    ages = result["age"].to_list()

    assert ages[0] == 79
    assert ages[-1] == 85
    assert set(ages[1:-1]).issubset({80, 81, 82, 83, 84})
    assert len(set(ages[1:-1])) > 1


def test_load_cps_asec_spreads_topcoded_age_80(tmp_path):
    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1] * 22,
            "A_LINENO": list(range(1, 23)),
            "A_AGE": [79, *([80] * 20), 85],
            "A_FNLWGT": [100] * 22,
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    ages = dataset.persons.sort("person_number")["age"].to_list()

    assert ages[0] == 79
    assert ages[-1] == 85
    assert set(ages[1:-1]).issubset({80, 81, 82, 83, 84})
    assert len(set(ages[1:-1])) > 1
