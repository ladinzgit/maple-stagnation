# EDA Summary

This directory contains exploratory checks for the active 270-290 main-character sample.

## Current Sample

The current collection is filtered in `scripts/collect_main_characters.py`.

- checkpoint months: `2025-06`, `2025-12`, `2026-05`
- active condition: access observed in all three checkpoint months
- final sample: 2,000 characters
- level bands: 665 / 665 / 670 for `270-279`, `280-285`, `286-290`
- class groups: 400 each

## Current Feature Profile

Run:

```bash
python eda/profile_active_features.py
```

Outputs:

- `eda/ACTIVE_FEATURE_PROFILE.md`
- `eda/active_feature_summary.csv`
- `eda/figures/active_h1_feature_distributions.png`
- `eda/figures/active_support_feature_distributions.png`

The profiler filters feature tables to the current `data/main_characters.csv` OCID set before summarizing.

## Current H1 Features

The final H1 feature aliases are:

| Alias | Concrete feature |
|---|---|
| `cumexp_avg` | `log1p_avg_monthly_delta_cumexp` |
| `union_avg` | `avg_monthly_delta_union_level`, clipped `[0, p99]` |
| `access_ratio` | monthly access ratio |

Support/diagnostic features include level, combat power, authentic symbol, HEXA core growth, HEXA fragment growth, access activity, and character age.

## Notes

- `avg_monthly_delta_combat_power` is heavy-tailed and can be negative because monthly stat snapshots are noisy; use p01-p99 winsorization before modeling when included.
- Nonnegative growth deltas are clipped at lower 0 and upper p99 in the H1 feature selection workflow.
- Access features are still informative because collection requires activity only at three checkpoint months, not continuous activity across all months.
