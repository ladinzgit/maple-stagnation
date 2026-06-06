# H1: 성장 정체 기반 주차 후보 탐색 (6mo 전환 2026-06-05)

## 목적

H1 가설은 **최근 6개월간 주요 성장 지표가 낮은 활성 캐릭터들은 일반 성장 캐릭터들과 구분되는 성장 정체 군집으로 식별될 것이다**이다. H1은 **≥10/12 접속 통제** 270~290 본캐 표본에서 파워 재투자 하위 군집을 찾는다. 공개 API로 주간 보스 수행·메소 생산을 직접 볼 수 없으므로, H1 결과는 확정 주차 유저가 아니라 H2/H3에서 사용할 **후보 라벨**이다.

## 핵심 설계

- **윈도우 = 최근 6개월(2025-12~2026-05)**: 주차=현재 행동 → 12mo는 과거 활발기가 섞여 정체신호 희석(recall 0.57). 6mo로 좁히면 recall 0.83.
- **재현성**: 6mo delta = features_monthly.csv의 `recent6_delta_*` 컬럼. raw 6mo 재산출과 corr=1.0·diff=0 동등 → features_monthly 단독 재현(monthly_snapshots_raw 불필요).
- **접속 = 통제변인**: 표본 전원 `access_active_months ≥ 10` → 접속이 군집 축이 되지 못함. access family는 클러스터링 피처에서 제외.
- **주차 후보군 = 클러스터링 결과(`is_stagnant_cluster`) 자체** (절대-0 동결 게이트 폐기).

## 표본

- 본캐 표본: 2,000명 (전원 ≥10/12 접속) / 클러스터링 유효 표본: 2,000명 (core delta dropna 제거 0)

## 채택 피처 (4-시스템 다축)

| Alias | 실제 값 | 시스템 |
|---|---|---|
| `cp_slog` | `sign(Δcp)·log1p(\|Δcp\|)` — 전투력 월평균 증가량(winsor) signed-log | 전투력 |
| `hexa_avg` | `avg_monthly_delta_hexa` clip≥0 | 헥사 코어 |
| `union_log` | `log1p(clip≥0 avg_monthly_delta_union_level)` | 유니온 |
| `auth_log` | `log1p(clip≥0 avg_monthly_delta_authentic_symbol)` | 어센틱 심볼 |

- 4개 게임 시스템의 **미재투자 다면 포착**. 전투력 Δ는 음수(감소=주차) → **signed-log**로 부호 보존. 다중공선성 통과(|corr| 최대 0.54, VIF<1.6). 클러스터링 전 `StandardScaler`, 성장 정체 군집 = 스케일 중심좌표 합 최저.

## 결과

| 항목 | 값 |
|---|---:|
| K-Means k | 4 (elbow; best-sil k=6 sil 0.384) |
| Silhouette @k4 | 0.357 |
| 주차 후보 수 | **380명 (19.0%)** |
| park(stag≥4) enrich / recall | 4.37x / **83.0%** |

| Cluster | 인원 | 해석 |
|---|---:|---|
| 0 | 882명 | 일반 성장 군집 (전투력 증가, 보통 재투자) |
| 1 (주차 후보) | 380명 | **전투력 감소**(median Δcp ≈ −2.07M) + 헥사·유니온·심볼 저재투자, 접속 거의 만점(고활성) |
| 2 | 551명 | 고성장 군집 (헥사 Δ 6.1, 286-290 64%) |
| 3 | 187명 | 전투력만 증가, 유니온·심볼 정체 |

운영점 = **elbow k=4(광의: recall 0.83)**. best-sil k=6은 고순도 코어(162명, enrich 8.6) 협의 옵션. 12mo에선 다축 추가가 무익(cp 단일축 지배)이었으나 6mo에선 고-k 고순도 코어 분리로 의미 회복. DBSCAN(3군집·noise 2.9%)은 정체 380 ↔ DBSCAN cluster1(355) **강한 정합**. 상세는 `RESULT.md`.

**silhouette 0.36은 주 지표 아님** — 성장량이 연속 스펙트럼이라 경계 흐림이 정상이고, H1 목적은 확정 분류가 아닌 H2/H3용 후보 라벨 생성. 주 지표 = 정체군 profile · enrich/recall · DBSCAN 정합 · H2/H3 다운스트림 재현성. silhouette은 k 선택 보조 신호.

## 실행

```bash
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
```

## 산출물

| 파일 | 역할 |
|---|---|
| `optimal_feature_set.json` | 채택셋 (`[cp_slog, hexa_avg, union_log, auth_log]`, k=4, window 6mo, user-pinned) |
| `data/cluster_labels.csv` | `cluster_km`, `is_stagnant_cluster` (주차 후보군 380명) — H2/H3 단일 입력 라벨 |
| `figures/` | silhouette, PCA, DBSCAN 진단 그림 |
| `RESULT.md` | 최종 H1 결과 요약 |
