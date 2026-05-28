# ACA PTC Multiplier Source Choice

This records the first Microplex-US reconstruction of
`policyengine-us-data`'s `aca_ptc_multipliers_2022_2024.csv` from Arch
publisher-source consumer facts.

## Recipe

Inputs:

- KFF full-year average marketplace effectuated enrollment, 2022 and 2024
- CMS 2022 OEP state-level average monthly APTC
- CMS 2024 OEP state-level average monthly APTC
- CMS full-year 2022 effectuated-enrollment workbook average monthly APTC

Source selection:

- `enroll_2022` and `enroll_2024`: KFF full-year effectuated enrollment
- `aptc_2024`: CMS 2024 OEP average monthly APTC
- `aptc_2022`: CMS 2022 OEP average monthly APTC where published, with CMS
  full-year 2022 average monthly APTC as fallback

Derived columns:

- `vol_mult = enroll_2024 / enroll_2022`
- `val_mult = aptc_2024 / aptc_2022`
- PE's state `tax_unit_count` factor uses `vol_mult`
- PE's state `aca_ptc` amount factor uses `vol_mult * val_mult`

## Reproduction

Build the five Arch source-package suites, then run:

```bash
uv run microplex-us-build-aca-ptc-multipliers \
  /tmp/mp-aca-ptc-arch-sources/kff-2022/consumer_facts.jsonl \
  /tmp/mp-aca-ptc-arch-sources/kff-2024/consumer_facts.jsonl \
  /tmp/mp-aca-ptc-arch-sources/cms-oep-2022/consumer_facts.jsonl \
  /tmp/mp-aca-ptc-arch-sources/cms-oep-2024/consumer_facts.jsonl \
  /tmp/mp-aca-ptc-arch-sources/cms-effectuated-2022/consumer_facts.jsonl \
  --out /tmp/mp-aca-ptc-arch-sources/aca_ptc_multipliers_2022_2024.csv
```

The 2026-05-12 run wrote 51 rows. Compared with PE's incumbent
`policyengine_us_data/storage/aca_ptc_multipliers_2022_2024.csv`:

- state set matches
- `enroll_2022` matches for all 51 states
- `enroll_2024` matches for all 51 states
- `vol_mult` matches for all 51 states
- `aptc_2024` matches for all 51 states
- `aptc_2022` differs for 22 states
- `val_mult` differs for the same 22 states

## PE Incumbent Provenance Trace

The local `policyengine-us-data` history does not contain a generator for the
incumbent CSV. `git log --follow` shows the file first appearing at its current
path in `8d2c49fa15a515e2379d1b4b5e2c1856a1d4ebe9` on 2026-02-11:
`Add hierarchical uprating notebook, fix verification, move ACA PTC
multipliers`. The commit adds
`policyengine_us_data/storage/aca_ptc_multipliers_2022_2024.csv` directly, plus
notebooks which document that ACA PTC factors are loaded from the CSV and
described as CMS/KFF enrollment data. Those notebooks do not show row-level
source derivation.

Spot checks against the raw CMS 2022 OEP state-level source support the
Microplex-US source choice for the mismatching states where OEP publishes a
number. For example, current Arch-selected OEP values are New Jersey `489`, New
Mexico `460`, and Virginia `506`, matching the CMS OEP
`APTC_Cnsmr_Avg_APTC` column. The PE incumbent has `504`, `534`, and `407` for
those states, respectively. Nevada remains the explicit fallback case because
the CMS 2022 OEP state-level file reports no Nevada average monthly APTC fact;
Microplex-US uses the CMS full-year effectuated-enrollment value `429.75`.

## Reconciliation Queue

States not listed matched PE's incumbent CSV exactly. For listed states, the
Microplex-US value is the Arch publisher-source value selected by the recipe
above. Nevada is the known CMS full-year fallback case because the CMS 2022 OEP
state-level source package has no Nevada average monthly APTC fact.

| State | PE aptc_2022 | Microplex-US aptc_2022 | PE val_mult | Microplex-US val_mult |
| --- | ---: | ---: | ---: | ---: |
| Nevada | 435 | 429.75 | 1.006896551724138 | 1.019197207678883 |
| New Jersey | 504 | 489 | 1.0337301587301588 | 1.065439672801636 |
| New Mexico | 534 | 460 | 1.0318352059925093 | 1.1978260869565218 |
| New York | 364 | 363 | 1.25 | 1.2534435261707988 |
| North Carolina | 583 | 579 | 0.9571183533447685 | 0.9637305699481865 |
| North Dakota | 436 | 452 | 0.9931192660550459 | 0.9579646017699115 |
| Ohio | 479 | 437 | 1.0396659707724425 | 1.139588100686499 |
| Oklahoma | 577 | 558 | 0.9965337954939342 | 1.0304659498207884 |
| Oregon | 503 | 489 | 1.0417495029821073 | 1.0715746421267893 |
| Pennsylvania | 523 | 501 | 1.0133843212237095 | 1.0578842315369261 |
| Rhode Island | 427 | 403 | 1.063231850117096 | 1.1265508684863523 |
| South Carolina | 566 | 512 | 0.9770318021201413 | 1.080078125 |
| South Dakota | 649 | 640 | 0.9414483821263482 | 0.9546875 |
| Tennessee | 572 | 543 | 1.013986013986014 | 1.0681399631675874 |
| Texas | 539 | 502 | 0.9944341372912802 | 1.0677290836653386 |
| Utah | 385 | 370 | 1.0935064935064935 | 1.1378378378378378 |
| Vermont | 620 | 566 | 1.132258064516129 | 1.2402826855123674 |
| Virginia | 407 | 506 | 0.995085995085995 | 0.8003952569169961 |
| Washington | 438 | 437 | 1.0342465753424657 | 1.036613272311213 |
| West Virginia | 1057 | 1002 | 0.97918637653737 | 1.032934131736527 |
| Wisconsin | 562 | 530 | 1.0177935943060499 | 1.079245283018868 |
| Wyoming | 873 | 812 | 0.9885452462772051 | 1.062807881773399 |

Open reconciliation decision:

- Treat the Microplex-US output as the publisher-source reconstruction.
- Treat PE byte parity as a separate legacy-compatibility target. Do not add
  overrides unless a row-level legacy source or intentional source-choice table
  is supplied.
