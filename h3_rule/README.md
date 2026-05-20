# 가설 3 — Feature Importance 기반 Rule 평가

Feature Importance로 도출한 Rule이 일반 유저 피해 없이 주차 유저를 식별할 수 있는지 평가한다.

## 입력

H1 클러스터 레이블 (pseudo-label) + 전체 피처

## 방법

1. Random Forest / XGBoost 학습 (pseudo-label 기준)
2. Feature Importance 산출 → 핵심 피처 기반 Rule 도출
   - 예: `Δ레벨 < N AND Δ전투력 < M`
3. Precision / Recall / F1 / **False Positive Rate** / ROC-AUC 평가
4. Threshold별 Trade-off 시각화

## 출력

- Feature Importance 차트
- 최적 Rule 조건
- 평가 지표 테이블 및 ROC Curve

## 주의

H3 pseudo-label 품질은 H1 Silhouette Score에 의존한다.
H1 Silhouette Score > 0.4 달성 시 H3 신뢰도 확보.
