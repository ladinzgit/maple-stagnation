# H1: 성장 정체 기반 주차 후보 탐색

## 목적

H1은 active checkpoint 조건을 통과한 270~290 본캐 표본에서 성장 정체 특성이 강한 군집을 찾는다. 공개 API로 주간 보스 수행이나 메소 생산량을 직접 볼 수 없으므로, H1 결과는 확정 주차 유저가 아니라 이후 H2/H3에서 사용할 후보 라벨이다.

## 표본

- active checkpoint: `2025-06`, `2025-12`, `2026-05`
- 본캐 표본: 2,000명
- 클러스터링 유효 표본: 1,979명

## 최종 피처

`feature_selection.ipynb`에서 clipping/winsorize 정책을 적용한 뒤 다음 피처 조합을 선택했다.

| Alias | 실제 값 |
|---|---|
| `cumexp_avg` | `log1p_avg_monthly_delta_cumexp` |
| `union_avg` | `avg_monthly_delta_union_level`, `[0, p99]` clipping |
| `access_ratio` | 월별 접속 비율 |

전처리 정책:

- 누적 경험치 성장량: `log1p` 값 사용
- 레벨/유니온/어센틱/HEXA 조각류 증가량: 하한 0, 상한 p99 clipping
- 전투력 증가량: p01~p99 winsorize
- 클러스터링 전 `StandardScaler` 적용

## 결과

| 항목 | 값 |
|---|---:|
| K-Means k | 3 |
| Silhouette | 0.5446 |
| 주차 후보 cluster id | 1 |
| H1 주차 후보 수 | 412명 |

클러스터별 해석:

| Cluster | 인원 | 해석 |
|---|---:|---|
| 0 | 1,419명 | 일반 성장 군집 |
| 1 | 412명 | 성장 정체/주차 후보 군집 |
| 2 | 148명 | 유니온 성장 특화 군집 |

자세한 피처 프로파일은 `RESULT.md`에 정리했다.

## 실행

```bash
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/temporal_external_validation.ipynb
```

## 산출물

| 파일 | 역할 |
|---|---|
| `optimal_feature_set.json` | feature selection 결과 |
| `data/cluster_labels.csv` | cluster label 및 `is_stagnant_cluster` |
| `data/h1_current_candidates.csv` | 현재성 조건을 반영한 H2 보조 분석/H3 최종 평가 라벨 |
| `figures/` | silhouette, PCA, DBSCAN 등 진단 그림 |
| `RESULT.md` | 최종 H1 결과 요약 |
