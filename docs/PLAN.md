# 수집 및 분석 계획

## 수집

현재 파이프라인은 `scripts/collect_main_characters.py`에서 active checkpoint 필터를 먼저 적용한다.

- checkpoint months: `2025-06`, `2025-12`, `2026-05`
- 기본 조건: 세 checkpoint month 모두에서 접속 관측
- 목표 표본: 5계열 × 400명 = 2,000명
- 현재 산출물: `data/main_characters.csv` 2,000명

피처 수집:

```bash
python scripts/collect_features.py --refresh-raw
python scripts/collect_hexa_fragments.py
```

현재 산출물:

| 파일 | 행 수 |
|---|---:|
| `data/features_monthly.csv` | 2,000 |
| `data/monthly_snapshots_raw.csv` | 24,000 |
| `data/hexa_fragments.csv` | 2,000 |

## H1 실행

```bash
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/temporal_external_validation.ipynb
```

현재 H1 기준:

- feature selection 결과: `cumexp_avg`, `union_avg`, `access_ratio`
- K-Means k: 3
- silhouette: 0.5437
- cluster label 유효 표본: 1,979명
- 최종 주차 후보 cluster: `cluster_km == 1`
- 최종 H1 주차 후보: 412명

## H2 실행

```bash
python h2_distribution/run_analysis.py
jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb
```

현재 H2 기준:

- 기본 분석 표본: 1,979명
- H1 클러스터 후보: 412명
- 보조 현재성 후보: 63명
- 보조 고신뢰 후보: 35명
- H1 후보 기준 레벨 구간 검정: 유의
- H1 후보 기준 직업 계열 검정: 유의하지 않음

## H3 계획

H3는 아직 구현 예정이다.

- 1단계 supervised target: `is_stagnant_cluster` 412명
- 최종 rule 평가 target: `is_current_parking_candidate` 63명
- 민감도 target: `is_high_confidence_candidate` 35명
- 입력 피처: H1에서 직접 사용한 피처를 그대로 복제하지 않는 성장/상태/접속 피처
- 최종 rule: 성장 정체 rule AND 현재성 접속 게이트

## 해석 원칙

- 본 프로젝트는 확정 주차 유저 ground truth를 보유하지 않는다.
- H1은 성장 정체 후보군 탐색이다.
- H2는 후보군이 레벨/직업 계열에 불균일하게 분포하는지 확인하는 통계 검정이다.
- H3는 운영 가능한 rule 근사 실험이며, 실제 주차 유저 검증으로 해석하지 않는다.
