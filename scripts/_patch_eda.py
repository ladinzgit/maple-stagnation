"""_patch_eda.py — EDA notebook patcher for v2 data + hexa_fragments"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

NB_PATH = "eda/eda.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def make_md(text, cell_id):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "source": src, "metadata": {"id": cell_id}}


def make_code(text, cell_id):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code", "source": src,
        "metadata": {"id": cell_id},
        "execution_count": None, "outputs": [],
    }


def set_source(cell, text):
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


# ─── 1. Cell [03]: add hexa_fragments load ───────────────────────────────────
old3 = "".join(nb["cells"][3]["source"])
if "hexa_fragments" not in old3:
    addition = (
        "\n# 헥사 조각 데이터 로드\n"
        "hf_raw = pd.read_csv('../data/hexa_fragments.csv', encoding='utf-8-sig')\n"
        "print(f'hexa_fragments: {hf_raw.shape}')\n"
        "print(f'hexa_fragments_total nulls: {hf_raw[\"hexa_fragments_total\"].isna().sum()}')"
    )
    set_source(nb["cells"][3], old3 + addition)
    print("[1] Cell 03 updated — hexa_fragments load added")
else:
    print("[1] Cell 03 already has hexa_fragments — skipped")


# ─── 2. Insert Sec 7.5 after cell [30] ───────────────────────────────────────
already = any("Sec 7.5" in "".join(c["source"]) for c in nb["cells"])
if not already:
    sec75_md = make_md(
        "## Sec 7.5. 솔 에르다 조각 소비량 분석 (가설 외 탐색)\n"
        "\n"
        "`hexa_fragments.csv` 기반. HEXA 코어 레벨합(`hexa_level_sum`)은 features_monthly에 포함되나,\n"
        "**조각 누적 소비량** 은 레벨 비례가 아닌 실제 투자 행동을 반영.\n"
        "파킹 유저는 보스 파밍 목적이므로 HEXA 투자를 하지 않는다는 가설을 탐색.\n"
        "→ `delta_hexa_frag`가 `delta_hexa`와 r > 0.7이면 중복 → 클러스터링 피처 제외 근거.",
        "sec75_md",
    )

    sec75_code1_src = (
        "# hexa_fragments 병합 + 기초 통계\n"
        "df_hf = df_clean.merge(\n"
        "    hf_raw[['ocid', 'hexa_fragments_total', 'avg_monthly_delta_hexa_frag',\n"
        "            'recent3_delta_hexa_frag', 'recent6_delta_hexa_frag']],\n"
        "    on='ocid', how='left'\n"
        ")\n"
        "print(f'df_hf: {df_hf.shape}')\n"
        "print(f'hexa_fragments_total nulls: {df_hf[\"hexa_fragments_total\"].isna().sum()}')\n"
        "print()\n"
        "print(df_hf[['hexa_fragments_total', 'avg_monthly_delta_hexa_frag',\n"
        "             'recent3_delta_hexa_frag', 'recent6_delta_hexa_frag']]\n"
        "      .describe(percentiles=[.25, .5, .75, .9, .95]).round(1))"
    )
    sec75_code1 = make_code(sec75_code1_src, "sec75_code1")

    sec75_code2_src = (
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "\n"
        "# 왼쪽: hexa_fragments_total 분포\n"
        "vals = df_hf['hexa_fragments_total'].dropna()\n"
        "axes[0].hist(vals, bins=50, color='#4e79a7', edgecolor='white', linewidth=0.3)\n"
        "axes[0].set_title('HEXA 조각 누적 소비량 분포', fontsize=11)\n"
        "axes[0].set_xlabel('hexa_fragments_total')\n"
        "axes[0].set_ylabel('빈도')\n"
        "axes[0].axvline(vals.median(), color='red', linestyle='--',\n"
        "               label=f'중앙값 {vals.median():.0f}')\n"
        "axes[0].legend()\n"
        "\n"
        "# 가운데: avg_monthly_delta_hexa_frag — 파킹 프록시 비교\n"
        "parked_p = (\n"
        "    (df_hf['avg_monthly_delta_level'] == 0) &\n"
        "    (df_hf['avg_monthly_delta_combat_power'] == 0) &\n"
        "    (df_hf['avg_monthly_delta_union_level'] == 0)\n"
        ")\n"
        "df_hf['parked_proxy'] = parked_p\n"
        "for label, mask, color in [\n"
        "    ('주차 프록시', parked_p, '#e15759'),\n"
        "    ('정상 성장', ~parked_p, '#4e79a7'),\n"
        "]:\n"
        "    axes[1].hist(\n"
        "        df_hf.loc[mask, 'avg_monthly_delta_hexa_frag'].dropna(),\n"
        "        bins=40, alpha=0.6, label=label, color=color, density=True\n"
        "    )\n"
        "axes[1].set_title('월평균 조각 소비 변화량\\n(파킹 프록시 vs 정상)', fontsize=11)\n"
        "axes[1].set_xlabel('avg_monthly_delta_hexa_frag')\n"
        "axes[1].set_ylabel('밀도')\n"
        "axes[1].legend()\n"
        "\n"
        "# 오른쪽: hexa_level_sum vs hexa_fragments_total 상관\n"
        "x = df_hf['hexa_level_sum'].dropna()\n"
        "y = df_hf.loc[x.index, 'hexa_fragments_total']\n"
        "axes[2].scatter(x, y, alpha=0.3, s=10, color='#76b7b2')\n"
        "rho_lv, pval_lv = spearmanr(x, y)\n"
        "axes[2].set_title(\n"
        "    f'HEXA 레벨합 vs 조각 소비\\nSpearman r={rho_lv:.3f} (p={pval_lv:.1e})',\n"
        "    fontsize=11\n"
        ")\n"
        "axes[2].set_xlabel('hexa_level_sum')\n"
        "axes[2].set_ylabel('hexa_fragments_total')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "# 파킹 프록시별 비교\n"
        "print('[파킹 프록시별 헥사 조각 소비 비교]')\n"
        "print(df_hf.groupby('parked_proxy')\n"
        "      [['hexa_fragments_total', 'avg_monthly_delta_hexa_frag']]\n"
        "      .agg(['median', 'mean']).round(1))\n"
        "\n"
        "# delta_hexa_frag vs delta_hexa 중복 검사\n"
        "rho2, p2 = spearmanr(\n"
        "    df_hf['avg_monthly_delta_hexa_frag'].fillna(0),\n"
        "    df_hf['avg_monthly_delta_hexa'].fillna(0)\n"
        ")\n"
        "print(f'\\nSpearman r(delta_hexa_frag, delta_hexa): {rho2:.3f} (p={p2:.1e})')\n"
        "print('  r > 0.7 → 중복 → 클러스터링 피처에서 hexa_frag 제외 근거')"
    )
    sec75_code2 = make_code(sec75_code2_src, "sec75_code2")

    nb["cells"] = nb["cells"][:31] + [sec75_md, sec75_code1, sec75_code2] + nb["cells"][31:]
    print("[2] Sec 7.5 inserted (3 cells) after cell 30")
    cluster_feat_idx = 41 + 3  # shifted by 3
else:
    print("[2] Sec 7.5 already present — skipped")
    cluster_feat_idx = 41


# ─── 3. Fix CLUSTER_FEATURES ─────────────────────────────────────────────────
cell_src = "".join(nb["cells"][cluster_feat_idx]["source"])
if "arcane_stagnant" in cell_src and "CLUSTER_FEATURES_A" not in cell_src:
    new_src = (
        "# ── 클러스터링 피처 셋 (H1) ────────────────────────────────────────────────\n"
        "# arcane_stagnant(이진)은 StandardScaler 후 K-Means 거리 왜곡 → 제외.\n"
        "# 연속형 avg_monthly_delta_arcane_symbol(≥0 클램프) 사용.\n"
        "\n"
        "# Feature Set A (primary) — 12개월 avg_monthly_delta\n"
        "CLUSTER_FEATURES_A = [\n"
        "    'avg_monthly_delta_level',             # 핵심 정체 신호\n"
        "    'delta_cp_winsor',                     # P5-P95 winsorize\n"
        "    'avg_monthly_delta_union_level',       # 핵심 정체 신호\n"
        "    'avg_monthly_delta_authentic_symbol',  # 280+ 성장 신호\n"
        "    'avg_monthly_delta_arcane_symbol',     # 연속형 (arcane_stagnant 대체)\n"
        "]\n"
        "\n"
        "# Feature Set A' (age-debiased) — recent6 기울기, 렌 코호트 편향 제거\n"
        "CLUSTER_FEATURES_Ap = [\n"
        "    'recent6_delta_level',\n"
        "    'recent6_delta_combat_power',\n"
        "    'recent6_delta_union_level',\n"
        "    'recent6_delta_authentic_symbol',\n"
        "    'recent6_delta_arcane_symbol',\n"
        "]\n"
        "\n"
        "CLUSTER_FEATURES = CLUSTER_FEATURES_A  # H1에서 A vs A' 비교 후 확정\n"
        "\n"
        "print('[Feature Set A (primary)]')\n"
        "for f in CLUSTER_FEATURES_A:\n"
        "    print(f'  - {f}')\n"
        "print(f'총 {len(CLUSTER_FEATURES_A)}개 | StandardScaler → K-Means / DBSCAN')\n"
        "\n"
        "print(\"\\n[Feature Set A' (recent6, age-debiased)]\")\n"
        "for f in CLUSTER_FEATURES_Ap:\n"
        "    print(f'  - {f}')\n"
        "\n"
        "print('\\n[결측 점검 — Feature Set A]')\n"
        "print(df_final[CLUSTER_FEATURES_A].isnull().sum().to_string())"
    )
    set_source(nb["cells"][cluster_feat_idx], new_src)
    print(f"[3] Cell [{cluster_feat_idx}] CLUSTER_FEATURES fixed: arcane_stagnant → avg_monthly_delta_arcane_symbol")
else:
    print(f"[3] Cell [{cluster_feat_idx}] — already updated or arcane_stagnant not found, skipped")


# ─── Save ────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] {NB_PATH} saved. Total cells: {len(nb['cells'])}")
