"""
collect_hexa_fragments.py — 솔 에르다 조각 소비량 수집 (v1.0)

각 캐릭터의 HEXA 코어 정보를 12개월 월별 스냅샷으로 수집하고
누적 솔 에르다 조각 소비량과 월평균 소비량을 계산.

코어 타입 → 조각 비용 테이블 매핑:
  오리진 코어                             → 스킬 코어 비용
  마스터리 코어                           → 마스터리아 코어 비용
  강화 코어                               → 강화 코어 비용
  공용 코어 + linked_skill ID가 VI로 끝남 → 직업군 코어 비용
  공용 코어 + 그 외                       → 공통 원 코어 비용

출력: data/hexa_fragments.csv
  ocid, character_name, character_class, world_name,
  hexa_fragments_total,               # 최신 스냅샷 기준 누적 조각 소비량
  avg_monthly_delta_hexa_frag,        # 12개월 월평균 조각 소비 변화량
  recent3_delta_hexa_frag,            # 최근 3개월 기울기
  recent6_delta_hexa_frag,            # 최근 6개월 기울기
  [yyyy-mm]_hexa_frag × 12,           # 월별 누적 조각 스냅샷
  num_valid_months, first_valid_month, last_valid_month
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

API_KEY = os.getenv("MAPLE_API_KEY")
BASE_URL = "https://open.api.nexon.com/maplestory/v1"

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=60, pool_maxsize=60))
_session.headers.update({"x-nxopen-api-key": API_KEY})

# ── 설정 ─────────────────────────────────────────────────────────────────────
_DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
_ASSET_DIR  = Path(__file__).resolve().parent.parent / "assets"
INPUT_FILE  = str(_DATA_DIR / "main_characters.csv")
OUTPUT_FILE = str(_DATA_DIR / "hexa_fragments.csv")
FRAGMENT_XLSX = str(_ASSET_DIR / "HEXA_Skill_Fragments_Only.xlsx")
CONCURRENCY = 30
MAX_RPS     = 400

SNAPSHOT_MONTHS = 12
END_YEAR_MONTH  = "2026-05"
RECENT_WINDOWS  = [3, 6]

_end_y, _end_m = map(int, END_YEAR_MONTH.split("-"))
MONTHS = []
_y, _m = _end_y, _end_m
for _ in range(SNAPSHOT_MONTHS):
    MONTHS.append(f"{_y:04d}-{_m:02d}")
    _m -= 1
    if _m < 1:
        _m = 12
        _y -= 1
MONTHS.reverse()
# ─────────────────────────────────────────────────────────────────────────────


# ── 조각 비용 테이블 로드 ─────────────────────────────────────────────────────
def _load_fragment_cost_table(xlsx_path: str) -> dict[str, list[int]]:
    """
    Excel에서 코어 타입별 레벨업 비용을 읽어 누적 비용 배열로 변환.

    반환: {
      'skill':       [0, 100, 130, ..., 4500],   # index = 현재 레벨 (0~30)
      'masteria':    [0,  50,  65, ..., 2252],
      'enhance':     [0,  75,  98, ..., 3383],
      'class_group': [0, 125, 163, ..., 6268],
      'common':      [0,  90, 115, ..., 4035],
    }
    cumcost[L] = 레벨 0에서 L까지 올리는 데 필요한 총 조각 수
    """
    import shutil, tempfile
    # 파일이 열려 있을 수 있으니 임시 경로 복사 후 로드
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(xlsx_path, tmp_path)

    df = pd.read_excel(tmp_path, header=None)
    os.unlink(tmp_path)

    # 행 구조: 0=blank, 1=title, 2=column headers, 3=unit row, 4~33=데이터(0→1~29→30)
    data_rows = df.iloc[4:34, 1:7].copy()
    data_rows.columns = ["level_range", "skill", "masteria", "enhance", "common", "class_group"]
    data_rows = data_rows.reset_index(drop=True)

    cost_cols = ["skill", "masteria", "enhance", "class_group", "common"]
    costs = data_rows[cost_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    result = {}
    for col in cost_cols:
        per_level = costs[col].tolist()           # index i = cost for level i→i+1 (i=0..29)
        cumcost = [0] * 31                        # cumcost[L] = total to reach level L
        for i in range(30):
            cumcost[i + 1] = cumcost[i] + per_level[i]
        result[col] = cumcost
    return result


FRAG_CUMCOST = _load_fragment_cost_table(FRAGMENT_XLSX)


def fragment_cost(level: int, core_cost_type: str) -> int:
    """레벨 L 코어의 누적 조각 소비량 반환. 범위 외는 0."""
    if level <= 0:
        return 0
    level = min(level, 30)
    costs = FRAG_CUMCOST.get(core_cost_type)
    if costs is None:
        return 0
    return costs[level]


def get_core_cost_type(core: dict) -> str | None:
    """
    코어 dict에서 조각 비용 타입 키를 결정.

    hexa_core_type:
      '오리진 코어'  → 'skill'
      '마스터리 코어' → 'masteria'
      '강화 코어'    → 'enhance'
      '공용 코어'    → linked_skill 내 skill_id 가 'VI'로 끝나면 'class_group',
                       아니면 'common'
    """
    raw_type = (core.get("hexa_core_type") or "").strip()

    if raw_type in ("오리진 코어", "스킬 코어"):
        return "skill"
    if raw_type in ("마스터리 코어", "마스터리아 코어"):
        return "masteria"
    if raw_type == "강화 코어":
        return "enhance"
    if raw_type == "공용 코어":
        linked = core.get("linked_skill") or []
        for skill in linked:
            sid = (skill.get("hexa_skill_id") or "").strip()
            if sid.endswith("VI"):
                return "class_group"
        return "common"

    # 미지 타입 — None 반환 (집계 제외)
    return None


def total_fragments_from_hexa(hexa_data: dict) -> int | None:
    """
    헥사 데이터에서 전체 누적 조각 소비량 계산.
    코어가 0개면 0, API 오류면 None.
    """
    if not hexa_data:
        return None
    cores = hexa_data.get("character_hexa_core_equipment") or []
    total = 0
    first_skill_seen = False
    for core in cores:
        level = int(core.get("hexa_core_level") or 0)
        ctype = get_core_cost_type(core)
        if ctype is not None:
            cost = fragment_cost(level, ctype)
            if ctype == "skill" and not first_skill_seen:
                cost = max(0, cost - fragment_cost(1, "skill"))  # level 1 is free
                first_skill_seen = True
            total += cost
    return total
# ─────────────────────────────────────────────────────────────────────────────


class RateLimiter:
    def __init__(self, calls_per_second):
        self._interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_limiter = RateLimiter(MAX_RPS)


def api_get(endpoint, params):
    _limiter.acquire()
    try:
        r = _session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def avg_monthly_delta(indexed_values):
    """[(month_idx, value), ...] → 월평균 변화량. 유효값 2개 미만이면 None."""
    valid = [(i, v) for i, v in indexed_values if v is not None]
    if len(valid) < 2:
        return None
    first_idx, first_val = valid[0]
    last_idx,  last_val  = valid[-1]
    months_elapsed = last_idx - first_idx
    if months_elapsed == 0:
        return None
    return (last_val - first_val) / months_elapsed


def process_character(row):
    """캐릭터 1명: 12개월 헥사 스냅샷 수집 → 조각 소비량 계산 → dict 반환."""
    ocid = str(row["ocid"])

    monthly_frag = []          # [(month_idx, fragment_total_or_None), ...]
    month_cols   = {}          # {f"{ym}_hexa_frag": value}

    for i, ym in enumerate(MONTHS):
        date_01 = f"{ym}-01"
        hexa_data = api_get("character/hexamatrix", {"ocid": ocid, "date": date_01})
        frag = total_fragments_from_hexa(hexa_data)
        monthly_frag.append((i, frag))
        month_cols[f"{ym}_hexa_frag"] = frag

    valid = [(i, v) for i, v in monthly_frag if v is not None]
    if not valid:
        return {
            "character_name":               row["character_name"],
            "ocid":                         ocid,
            "character_class":              row.get("character_class", ""),
            "world_name":                   row.get("world_name", ""),
            "hexa_fragments_total":         None,
            "avg_monthly_delta_hexa_frag":  None,
            "recent3_delta_hexa_frag":      None,
            "recent6_delta_hexa_frag":      None,
            **{k: None for k in month_cols},
            "num_valid_months":             0,
            "first_valid_month":            None,
            "last_valid_month":             None,
        }

    n_months     = len(MONTHS)
    last_frag    = valid[-1][1]
    first_idx    = valid[0][0]
    last_idx     = valid[-1][0]
    avg_delta    = avg_monthly_delta(valid)
    recent_delta = {
        f"recent{w}_delta_hexa_frag": avg_monthly_delta(
            [(i, v) for i, v in valid if i >= n_months - w]
        )
        for w in RECENT_WINDOWS
    }

    return {
        "character_name":               row["character_name"],
        "ocid":                         ocid,
        "character_class":              row.get("character_class", ""),
        "world_name":                   row.get("world_name", ""),
        "hexa_fragments_total":         last_frag,
        "avg_monthly_delta_hexa_frag":  avg_delta,
        **recent_delta,
        **month_cols,
        "num_valid_months":             len(valid),
        "first_valid_month":            MONTHS[first_idx],
        "last_valid_month":             MONTHS[last_idx],
    }


def load_done_ocids(filepath):
    if not os.path.exists(filepath):
        return set()
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return set(df["ocid"].dropna().astype(str))


def save_results(rows, filepath):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, encoding="utf-8-sig")
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset="ocid", keep="first")
    new_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  [저장] {filepath} — 누적 {len(new_df)}행")


# ── 단일 캐릭터 검증 모드 ────────────────────────────────────────────────────
def verify_single(character_name: str, expected_total: int | None = None):
    """
    캐릭터 이름으로 OCID 조회 후 최신 스냅샷 조각 수 계산 및 출력.
    expected_total 지정 시 일치 여부 확인.
    """
    print(f"=== 단일 검증: {character_name} ===")
    r = api_get("id", {"character_name": character_name})
    if not r:
        print("  OCID 조회 실패")
        return

    ocid = r.get("ocid")
    print(f"  OCID: {ocid[:20]}...")

    date = f"{END_YEAR_MONTH}-01"
    hexa_data = api_get("character/hexamatrix", {"ocid": ocid, "date": date})
    if not hexa_data:
        print(f"  hexamatrix 조회 실패 (date={date})")
        return

    cores = hexa_data.get("character_hexa_core_equipment") or []
    print(f"  코어 수: {len(cores)}")
    total = 0
    for core in cores:
        level     = int(core.get("hexa_core_level") or 0)
        ctype     = get_core_cost_type(core)
        cost      = fragment_cost(level, ctype) if ctype else 0
        name      = core.get("hexa_core_name", "?")
        raw_type  = core.get("hexa_core_type", "?")
        linked    = [s.get("hexa_skill_id","?") for s in (core.get("linked_skill") or [])]
        total    += cost
        print(f"    {name!s:30s} type={raw_type!s:8s} → {ctype!s:12s}  lv={level:2d}  cost={cost:5d}")
    print(f"\n  총 조각 소비량: {total}")
    if expected_total is not None:
        ok = total == expected_total
        print(f"  기대값 {expected_total} {'✓ 일치' if ok else f'✗ 불일치 (차이 {total - expected_total})'}")
    return total


def collect():
    chars_df   = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    done_ocids = load_done_ocids(OUTPUT_FILE)
    targets    = chars_df[~chars_df["ocid"].astype(str).isin(done_ocids)]
    total      = len(targets)

    calls_est = total * SNAPSHOT_MONTHS   # 월당 hexamatrix 1회
    print(f"=== 솔 에르다 조각 수집 시작 ===")
    print(f"수집 월: {MONTHS[0]} ~ {MONTHS[-1]} ({len(MONTHS)}개월)")
    print(f"대상: {total}명 (기수집 {len(done_ocids)}명 제외)")
    print(f"예상 API 호출: ~{calls_est:,}회 | 예상 소요: ~{calls_est / MAX_RPS:.0f}초\n")

    run_start = time.monotonic()
    results   = []
    done      = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_character, row): row["character_name"]
            for _, row in targets.iterrows()
        }
        for future in as_completed(futures):
            result = future.result()
            done  += 1
            if result is not None:
                results.append(result)

            if done % 100 == 0 or done == total:
                elapsed = time.monotonic() - run_start
                rate    = done / elapsed if elapsed > 0 else 0
                valid_c = sum(1 for r in results if r.get("num_valid_months"))
                print(f"  [{done}/{total}] {elapsed:.0f}s | {rate:.1f}명/s "
                      f"| 유효 {valid_c}/{len(results)}명(미저장)")
                save_results(results, OUTPUT_FILE)
                results = []

    if results:
        save_results(results, OUTPUT_FILE)

    elapsed = time.monotonic() - run_start
    print(f"\n=== 완료 | 소요: {elapsed:.1f}초 ===")

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        valid = df[df["num_valid_months"].notna() & (df["num_valid_months"] >= 2)]
        print(f"{OUTPUT_FILE}: {len(df)}행 | num_valid_months≥2: {len(valid)}행")
        if "hexa_fragments_total" in df.columns and len(valid) > 0:
            print(valid["hexa_fragments_total"].describe().round(0).to_string())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="솔 에르다 조각 소비량 수집")
    parser.add_argument("--verify", metavar="CHARACTER_NAME",
                        help="단일 캐릭터 검증 모드 (수집 없이 최신 스냅샷만 계산)")
    parser.add_argument("--expected", type=int, default=None,
                        help="--verify 검증 시 기대 조각 수")
    args = parser.parse_args()

    if args.verify:
        verify_single(args.verify, args.expected)
    else:
        collect()
