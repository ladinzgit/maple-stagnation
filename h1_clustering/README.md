# 가설 1 — K-Means / DBSCAN 클러스터링

**가설:** 주차 유저는 성장 정체 피처 기반으로 일반 유저와 구별되는 독립 군집을 형성한다.

## 데이터

- 소스: `data/features_monthly.csv` (24개월 월별 스냅샷 delta 피처)
- 전처리: `eda/eda.ipynb` Sec 0–10 재현 (`h1_clustering.ipynb` 자체 완결)
- 최종 샘플: **1,380명** (level ≥ 260 필터 적용)

## 피처 세트 (Feature Set A — raw)

| 피처 | 전처리 | 설명 |
|---|---|---|
| `avg_monthly_delta_level` | 그대로 사용 | 월평균 레벨 변화량 (파킹 핵심 신호) |
| `delta_cp` | Winsorize P5–P95 | 월평균 전투력 변화량 (이상치 제거) |
| `delta_union` | clip(lower=0) | 월평균 유니온 레벨 변화량 (음수 → 0) |
| `avg_monthly_delta_authentic_symbol` | 그대로 사용 | 월평균 어센틱 심볼 변화량 |
| `arcane_stagnant` | 이진화 | 아케인심볼 미만족(< 120) & delta == 0 → 1 |

> `normalized_delta_level`(경험적 P75 정규화)은 파킹 유저 집중으로 인한 편향(Spearman r=−0.429) 때문에 제외하고 raw 값 사용.

모든 피처에 `StandardScaler` 적용 후 클러스터링.

## 방법

1. **K-Means** (k=2~8, n_init=20, random_state=42)
   - Elbow Curve + Silhouette Score 비교
   - Silhouette 최대 k 자동 선택
2. **DBSCAN** (min_samples=10, eps=자동 탐지)
   - 5-NN k-distance plot → 정규화 대각선 거리 최대점으로 eps 결정
   - K-Means 결과와 교차표 비교
3. **PCA 2D** 시각화 (3-panel: 파킹 레이블 / stagnation_score / level_band)

## 결과

### K-Means

| k | Silhouette |
|---|---|
| 2 | 0.3738 |
| 3 | 0.4429 |
| 4 | 0.4429 |
| **5** | **0.4517 ← best** |
| 6 | 0.4143 |
| 7 | 0.3445 |
| 8 | 0.3586 |

**최적 k=5, Silhouette=0.4517**

### 클러스터 프로파일 (k=5)

| cluster | n | delta_level | delta_cp | delta_union | delta_authentic | arcane_stagnant | stagnation_score | parked_proxy% |
|---|---|---|---|---|---|---|---|---|
| **1 (파킹)** | **52** | **0.106** | **-28,000** | **6.1** | **0.04** | **1.000** | **4.46** | **78.8%** |
| 0 | 890 | 0.342 | 1,669,930 | 32.5 | 0.72 | 0.000 | 1.42 | 25.7% |
| 2 | 400 | 0.705 | 14,497,170 | 55.4 | 1.70 | 0.000 | 0.05 | 0.0% |
| 3 | 8 | 10.576 | 5,226,256 | 218.1 | 0.79 | 0.125 | 0.88 | 0.0% |
| 4 | 30 | 1.188 | 5,354,839 | 444.5 | 1.57 | 0.000 | 0.50 | 6.7% |

- 파킹 클러스터(#1): `arcane_stagnant=1.000` (전원 아케인 정체), `delta_level≈0.1`
- 레벨 분포: 260s 71.2%, 270s 26.9% → 파킹 주요 구간 집중
- **stagnation_score=5 인원 36명 전원(100%)** 파킹 클러스터 소속

### DBSCAN (eps=0.874, min_samples=10)

- 클러스터 수: 2개, noise=6.7% (93명)
- DBSCAN 클러스터 1 (51명) ↔ K-Means 파킹 클러스터 (52명): **1명 차이, 거의 완벽 일치**
- noise 93명 중 92명이 비파킹 → 파킹 클러스터는 밀도 기반으로도 뚜렷이 분리

### PCA 시각화

`figures/03_pca_3panel.png` 참조 (PC1=38.3%, PC2=21.8%)

## 판정

| 지표 | 값 | 기준 | 결과 |
|---|---|---|---|
| Silhouette Score | **0.4517** | > 0.3 | **H1 지지** |
| 파킹 클러스터 규모 | 52명 (3.8%) | — | 확인 |
| DBSCAN 일치율 | 51/52명 (98%) | — | 강한 재현성 |
| stagnation=5 집중도 | 36/36명 (100%) | — | 완벽 일치 |

**결론: H1 지지 — 파킹 유저는 성장 정체 피처 기반으로 일반 유저와 명확히 분리되는 독립 군집을 형성한다.**

## 출력 파일

| 파일 | 내용 |
|---|---|
| `h1_clustering.ipynb` | 전체 실험 코드 |
| `figures/01_kmeans_elbow_silhouette.png` | Elbow & Silhouette 곡선 |
| `figures/02_dbscan_kdist.png` | 5-NN k-distance plot |
| `figures/03_pca_3panel.png` | PCA 2D 3-panel 시각화 |
| `data/cluster_labels.csv` | 클러스터 레이블 (H2/H3 입력) |
