# H1: 성장 정체 군집 분석

**상태: 완료** (2026-06-05, 6mo 전환)

## 가설

최근 6개월간 주요 성장 지표가 낮은 활성 캐릭터들은 일반 성장 캐릭터와 구분되는 성장 정체 군집으로 식별될 것이다.

## 분석 설계

| 항목 | 내용 |
|---|---|
| 분석 윈도우 | 최근 6개월 (2025-12~2026-05), `recent6_delta_*` 컬럼 사용 |
| 표본 | 레벨 270~290 본캐 2,000명, 전원 ≥10/12개월 접속 통제 |
| 클러스터링 | K-Means(k=4, elbow 기준) + DBSCAN 보조 검증 |
| 접속 변수 | 통제변인으로만 사용, 클러스터링 피처에서 제외 |

## 채택 피처 (4-시스템 다축)

| Alias | 정의 | 시스템 |
|---|---|---|
| `cp_slog` | `sign(Δcp)·log1p(\|Δcp\|)` | 전투력 |
| `hexa_avg` | `clip≥0(avg_monthly_delta_hexa)` | HEXA 코어 |
| `union_log` | `log1p(clip≥0(Δunion_level))` | 유니온 |
| `auth_log` | `log1p(clip≥0(Δauthentic_symbol))` | 어센틱 심볼 |

다중공선성 통과: |corr| 최대 0.54 (hexa↔auth), VIF < 1.6.

## 결과

| 항목 | 값 |
|---|---:|
| K-Means k | 4 (elbow; best-sil k=6 sil 0.384) |
| Silhouette @k4 | 0.357 |
| 주차 후보군 | **380명 (19.0%)** |
| 외부 신호 enrich / recall | 4.37× / **83.0%** |

| Cluster | 인원 | 해석 |
|---|---:|---|
| 0 | 882명 | 일반 성장 (전투력 증가, 보통 재투자) |
| **1 (주차 후보)** | **380명** | 전투력 감소(median Δcp ≈ −2.07M) + 4시스템 저재투자, 고활성 접속 |
| 2 | 551명 | 고성장 (HEXA Δ 6.1, 286-290 64%) |
| 3 | 187명 | 전투력만 증가, 유니온·심볼 정체 |

DBSCAN 정합: 후보 380명 중 355명이 동일 밀도 군집에 포함됨 → 구조 안정성 확인.

> silhouette 0.36은 주 평가 지표가 아닙니다. 성장량이 연속 스펙트럼이라 경계 흐림이 정상이며, H1 목적은 확정 분류가 아닌 H2/H3용 후보 라벨 생성입니다.

## 실행

```bash
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
```

## 산출물

| 파일 | 내용 |
|---|---|
| `optimal_feature_set.json` | 채택 피처셋, k, 윈도우 정보 |
| `data/cluster_labels.csv` | `cluster_km`, `is_stagnant_cluster` — H2/H3 단일 입력 라벨 |
| `figures/` | Silhouette, PCA, DBSCAN 진단 그림 |
| `RESULT.md` | 상세 결과 |
