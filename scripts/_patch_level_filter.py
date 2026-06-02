"""Patch level filter 260-285 → 270-290 in eda/eda.ipynb and h1_clustering/h1_clustering.ipynb."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

PATCHES = {
    "eda/eda.ipynb": {
        "f367cd68": {
            "old": ["LEVEL_BIN_ORDER = ['260-269', '270-279', '280-285']",
                    "BIN_PALETTE = {'260-269': '#4e79a7', '270-279': '#f28e2b', '280-285': '#e15759'}"],
            "new": ["LEVEL_BIN_ORDER = ['270-279', '280-285', '286-290']",
                    "BIN_PALETTE = {'270-279': '#4e79a7', '280-285': '#f28e2b', '286-290': '#e15759'}"],
        },
        "cb42ab08": {
            "old": ["# 레벨 구간 (수집 시 사용한 동일 경계: 260-269 / 270-279 / 280-285)",
                    "df['level_bin'] = pd.cut(df['level'], bins=[259, 269, 279, 285], labels=LEVEL_BIN_ORDER)"],
            "new": ["# 레벨 구간 (수집 시 사용한 동일 경계: 270-279 / 280-285 / 286-290)",
                    "df['level_bin'] = pd.cut(df['level'], bins=[269, 279, 285, 290], labels=LEVEL_BIN_ORDER)"],
        },
        "547ea1db": {
            "old": ["# 프로젝트 범위(260≤level≤285)로 한정 후 핵심 delta(level/cp/union) NaN 행 listwise 제거",
                    "df_clean = df[df['level'].between(260, 285)].dropna(subset=DELTA_COLS[:3]).copy()"],
            "new": ["# 수집 범위(270≤level≤290)로 한정 후 핵심 delta(level/cp/union) NaN 행 listwise 제거",
                    "df_clean = df[df['level'].between(270, 290)].dropna(subset=DELTA_COLS[:3]).copy()"],
        },
        "0f0fe477": {
            "old": ["      2) 프로젝트 범위(260≤level≤285)로 필터 — 수집 의도 외 행 제거",
                    "      3) level_bin 생성 (260-269 / 270-279 / 280-285)",
                    "    df = df[df['level'].between(260, 285)].copy()",
                    "    df['level_bin'] = pd.cut(df['level'], bins=[259, 269, 279, 285],",
                    "                              labels=['260-269', '270-279', '280-285'])"],
            "new": ["      2) 수집 범위(270≤level≤290)로 필터",
                    "      3) level_bin 생성 (270-279 / 280-285 / 286-290)",
                    "    df = df[df['level'].between(270, 290)].copy()",
                    "    df['level_bin'] = pd.cut(df['level'], bins=[269, 279, 285, 290],",
                    "                              labels=['270-279', '280-285', '286-290'])"],
        },
    },
    "h1_clustering/h1_clustering.ipynb": {
        "md-h1-header": {
            "old": ["**데이터:** `data/features_monthly.csv` → level 260–285 필터 후 ~1,337명  "],
            "new": ["**데이터:** `data/features_monthly.csv` → level 270–290 필터 후 ~2,000명  "],
        },
        "sec-setup": {
            "old": ['df = df[df["level"].between(260, 285)].copy()',
                    'df["level_bin"] = pd.cut(df["level"], bins=[259, 269, 279, 285],',
                    '                          labels=["260-269", "270-279", "280-285"])',
                    'BAND_LABELS  = ["260-269", "270-279", "280-285"]',
                    'BAND_PALETTE = {"260-269": "#4e79a7", "270-279": "#f28e2b", "280-285": "#e15759"}'],
            "new": ['df = df[df["level"].between(270, 290)].copy()',
                    'df["level_bin"] = pd.cut(df["level"], bins=[269, 279, 285, 290],',
                    '                          labels=["270-279", "280-285", "286-290"])',
                    'BAND_LABELS  = ["270-279", "280-285", "286-290"]',
                    'BAND_PALETTE = {"270-279": "#4e79a7", "280-285": "#f28e2b", "286-290": "#e15759"}'],
        },
    },
}


def patch_cell_source(source_lines, old_strs, new_strs):
    """Replace old_strs[i] with new_strs[i] in source_lines (line-by-line)."""
    assert len(old_strs) == len(new_strs)
    src = "".join(source_lines)
    changed = False
    for old, new in zip(old_strs, new_strs):
        if old in src:
            src = src.replace(old, new, 1)
            changed = True
        else:
            print("  WARNING: not found: " + repr(old)[:80])
    if not changed:
        return source_lines, False
    # Rebuild as list of lines ending with \n
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n") and src.endswith("\n"):
        lines[-1] += "\n"
    return lines, True


for nb_path, cell_patches in PATCHES.items():
    print(f"\n=== {nb_path} ===")
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Build id → cell index map
    id_map = {}
    for i, cell in enumerate(nb["cells"]):
        cid = cell.get("id") or cell.get("metadata", {}).get("id")
        if cid:
            id_map[cid] = i

    for cell_id, patch in cell_patches.items():
        if cell_id not in id_map:
            print(f"  Cell {cell_id!r} NOT FOUND")
            continue
        idx = id_map[cell_id]
        cell = nb["cells"][idx]
        new_src, changed = patch_cell_source(cell["source"], patch["old"], patch["new"])
        if changed:
            cell["source"] = new_src
            print(f"  Cell {cell_id!r} (idx={idx}): patched")
        else:
            print(f"  Cell {cell_id!r} (idx={idx}): no change (already patched?)")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved.")

print("\nDone.")
