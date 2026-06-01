# 가설 2 (통계) — 레벨/직업 분포 검정 (Chi-Square)

주차 유저 군집이 특정 레벨 구간 및 직업 계열에 통계적으로 유의하게 집중 분포하는지 검정한다.

## 입력

- `data/cluster_labels.csv` (H1 출력 — `ocid`, `cluster_label`)
- `data/features_monthly.csv` 에서 파생한 `level_band`, `class_group` (EDA Sec 4/5 정의)

## 검정 대상

| 교차표 | 변수 | 구분 |
|---|---|---|
| 1 | 레벨 구간 (`level_band`) | `260~269` / `270~279` / `280~285` |
| 2 | 직업 계열 (`class_group`) | 전사 / 마법사 / 궁수 / 도적 / 해적 |

## 방법

1. **Chi-Square Test of Independence** (α = 0.05)
   - 귀무가설: `cluster_label` 과 변수가 독립
   - 대립가설: 두 변수 간 분포 의존성 존재
2. **사전 조건 확인**: 모든 셀의 기대 빈도 ≥ 5 (EDA Sec 5에서 사전 확인 완료)
3. **사후 분석**:
   - **표준화 잔차**(standardized residuals) 계산 → |z| > 2 셀 식별
   - **Cramer's V** 로 효과 크기 보고 (검정통계량만으로는 실용적 의미 파악 어려움)

## 수용 기준

두 교차표 중 최소 1개에서 **p < 0.05** + Cramer's V 보고로 효과 크기 명시.

## 출력

- 교차표 (관측 빈도 + 기대 빈도)
- Chi² 통계량, p-value, 자유도, Cramer's V
- 표준화 잔차 히트맵 → over-represented (z > 2) / under-represented (z < −2) 셀 강조
- 주차 유저 비율이 통계적으로 높은 레벨 구간·직업 계열 정리 (디렉터 타겟팅 권고)
