"""Profile feature distributions for the current active collection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EDA = ROOT / "eda"
FIGURES = EDA / "figures"

MAIN = DATA / "main_characters.csv"
FEATURES = DATA / "features_monthly.csv"
HEXA = DATA / "hexa_fragments.csv"
RAW = DATA / "monthly_snapshots_raw.csv"

H1_FEATURES = {
    "cumexp_avg": "log1p_avg_monthly_delta_cumexp",
    "union_avg": "avg_monthly_delta_union_level",
    "access_ratio": "access_ratio",
}

SUPPORT_FEATURES = [
    "avg_monthly_delta_level",
    "avg_monthly_delta_combat_power",
    "avg_monthly_delta_authentic_symbol",
    "avg_monthly_delta_hexa",
    "avg_monthly_delta_hexa_frag",
    "access_active_months",
    "access_active_weeks",
    "num_valid_months",
    "character_age_months",
]


def load_current_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(MAIN, encoding="utf-8-sig")
    features = pd.read_csv(FEATURES, encoding="utf-8-sig")
    hexa = pd.read_csv(HEXA, encoding="utf-8-sig")
    raw = pd.read_csv(RAW, encoding="utf-8-sig")

    for df in [main, features, hexa, raw]:
        df["ocid"] = df["ocid"].astype(str)

    current_ocids = set(main["ocid"])
    features = features[features["ocid"].isin(current_ocids)].copy()
    hexa = hexa[hexa["ocid"].isin(current_ocids)].copy()
    raw = raw[raw["ocid"].isin(current_ocids)].copy()
    raw = raw.drop_duplicates(["ocid", "year_month"], keep="last")

    df = main.merge(features, on="ocid", how="left", suffixes=("_main", ""))
    df = df.merge(
        hexa[["ocid", "hexa_fragments_total", "avg_monthly_delta_hexa_frag",
              "recent3_delta_hexa_frag", "recent6_delta_hexa_frag"]],
        on="ocid",
        how="left",
    )
    df["level_band"] = pd.cut(
        df["level_main"],
        bins=[269, 279, 285, 290],
        labels=["270-279", "280-285", "286-290"],
        include_lowest=True,
    )
    return main, df, raw, hexa


def summarize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "feature": col,
            "non_null": int(s.notna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "zero_pct": float(s.fillna(np.nan).eq(0).mean() * 100),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, index: bool = True) -> str:
    table = df.copy()
    if index:
        table = table.reset_index()
    headers = [str(c) for c in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_h1_distributions(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cols = list(H1_FEATURES.values())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes[0], cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(s, bins=40, color="#4e79a7", alpha=0.85)
        ax.set_title(col)
        ax.grid(alpha=0.2)
    for ax, col in zip(axes[1], cols):
        data = [
            pd.to_numeric(df.loc[df["level_band"].eq(band), col], errors="coerce").dropna()
            for band in ["270-279", "280-285", "286-290"]
        ]
        ax.boxplot(data, tick_labels=["270-279", "280-285", "286-290"], showfliers=False)
        ax.set_title(f"{col} by level band")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "active_h1_feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_support_distributions(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cols = [
        "avg_monthly_delta_level",
        "avg_monthly_delta_combat_power",
        "avg_monthly_delta_authentic_symbol",
        "avg_monthly_delta_hexa",
        "access_ratio",
        "character_age_months",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if col == "avg_monthly_delta_combat_power":
            lower, upper = s.quantile([0.01, 0.99])
            s = s.clip(lower, upper)
            title = f"{col} clipped 1-99%"
        else:
            title = col
        ax.hist(s, bins=40, color="#59a14f", alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "active_support_feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(main: pd.DataFrame, df: pd.DataFrame, raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    level_counts = main.assign(
        level_band=pd.cut(
            main["level"],
            bins=[269, 279, 285, 290],
            labels=["270-279", "280-285", "286-290"],
            include_lowest=True,
        )
    )["level_band"].value_counts().sort_index()

    group_counts = main["class_group"].value_counts().reindex(["전사", "마법사", "궁수", "도적", "해적"])
    checkpoint = raw[raw["year_month"].isin(["2025-06", "2025-12", "2026-05"])]
    checkpoint_rate = (
        checkpoint.groupby("year_month")["access_flag"]
        .mean()
        .reindex(["2025-06", "2025-12", "2026-05"])
    )

    h1_rows = summary[summary["feature"].isin(H1_FEATURES.values())]
    support_rows = summary[summary["feature"].isin(SUPPORT_FEATURES)]

    lines = [
        "# Active Feature Profile",
        "",
        "## Sample",
        "",
        f"- main sample: {len(main):,}",
        f"- feature rows matched to current sample: {len(df):,}",
        f"- monthly raw rows matched to current sample: {len(raw):,}",
        "- active collection filter: access observed in all checkpoint months `2025-06`, `2025-12`, `2026-05`",
        "",
        "### Level Bands",
        "",
        markdown_table(level_counts.to_frame("n")),
        "",
        "### Class Groups",
        "",
        markdown_table(group_counts.to_frame("n")),
        "",
        "### Checkpoint Access Rate In Current Sample",
        "",
        markdown_table((checkpoint_rate * 100).round(2).to_frame("access_rate_pct")),
        "",
        "## H1 Feature Summary",
        "",
        markdown_table(h1_rows.round(3), index=False),
        "",
        "## Support Feature Summary",
        "",
        markdown_table(support_rows.round(3), index=False),
        "",
        "## Generated Figures",
        "",
        "- `eda/figures/active_h1_feature_distributions.png`",
        "- `eda/figures/active_support_feature_distributions.png`",
        "",
        "## Notes",
        "",
        "- `features_monthly.csv` and `hexa_fragments.csv` can contain old OCIDs because collectors append and deduplicate rather than pruning removed samples. This profile filters all analysis to the current `main_characters.csv` OCID set.",
        "- `avg_monthly_delta_combat_power` is heavy-tailed and can be negative because monthly stat snapshots are noisy; downstream analysis should keep the existing winsorization/clipping step.",
        "- H1 currently uses `log1p_avg_monthly_delta_cumexp`, `avg_monthly_delta_union_level`, and `access_ratio`.",
    ]
    (EDA / "ACTIVE_FEATURE_PROFILE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    main_df, df, raw, _ = load_current_sample()
    columns = list(H1_FEATURES.values()) + SUPPORT_FEATURES
    summary = summarize_numeric(df, columns)
    summary.to_csv(EDA / "active_feature_summary.csv", index=False, encoding="utf-8-sig")
    plot_h1_distributions(df)
    plot_support_distributions(df)
    write_report(main_df, df, raw, summary)
    print(f"sample_n={len(main_df)}")
    print(f"matched_feature_rows={len(df)}")
    print(f"raw_rows={len(raw)}")
    print(f"wrote={EDA / 'ACTIVE_FEATURE_PROFILE.md'}")


if __name__ == "__main__":
    main()
