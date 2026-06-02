# 가설 3 (지도 학습) — Feature Importance 기반 Rule 평가

Feature Importance로 도출한 단순 Rule이 일반 유저 피해를 최소화하면서 주차 후보를 식별할 수 있는지 평가한다. 분류기는 핵심 피처 식별 도구이며, **본 평가 대상은 Rule** 이다.

## 설계 원칙 — 접속 피처를 분류기 입력에 포함

주차의 정의는 **성장 정체 + 활성 접속**의 2차원 신호다. 성장 정체 군집만으로는 주차 후보와 휴면(접속 끊긴) 캐릭터가 분리되지 않는다(H1 결과: 정체 군집의 약 94%가 휴면).

→ 휴면을 **수집·전처리 단계에서 사전 필터로 제거하지 않는다.** 대신 접속 피처(`access_active_months`, `access_ratio`, `access_recent`)를 **분류기 입력 피처로 넣어** 모델이 성장 정체와 접속 활성을 결합한 패턴을 직접 학습하게 한다. 이렇게 하면 지도 학습 제약을 지키면서 성장×접속 결합을 모델이 자동으로 수행한다.

## 입력

- **pseudo-label (positive)**: `data/h1_current_candidates.csv`의 `is_current_parking_candidate` (성장 정체 ∩ 접속 활성). 민감도 분석은 `is_high_confidence_candidate`. 접속 조건이 라벨 정의에 내재되어 휴면이 positive에서 제외된다.
- **입력 피처**: `data/features_monthly.csv`의 **넓은 원시 관측 피처** — `avg_monthly_delta_level`, `avg_monthly_delta_combat_power`, `avg_monthly_delta_authentic_symbol`, `avg_monthly_delta_hexa`, `union_level`, `level`, `hexa_level_sum`, `character_age_months`, 그리고 **접속 피처** `access_active_months` / `access_ratio` / `access_recent`.
  - **클러스터링 3피처(`log1p_avg_monthly_delta_cumexp`, `avg_monthly_delta_union_level`, `avg_monthly_delta_hexa_frag`)는 순환 방지를 위해 제외 또는 제한**한다. H1 군집 경계를 그대로 재현하면 importance가 무의미해진다.
- 전제: H1 수용 기준(Silhouette ≥ 0.4) 충족.

## 방법

1. **분류기 학습** (importance 추출 용도, 지도 학습)
   - Random Forest, XGBoost
   - 5-fold stratified CV
   - pseudo-label = H1 현재 후보 레이블 (접속 포함)
   - **class imbalance 처리**: positive 비율이 낮으므로 `class_weight='balanced'` / `scale_pos_weight` 적용, stratified fold로 fold별 positive 보존
   - **2-tier 라벨 (positive n 부족 시)**: 접속 임계를 완화한 라벨(예: `access_active_months ≥ 2`)로 학습 신호를 확보하고, 엄격 라벨(`≥ 4`)로 최종 정밀도를 보고
2. **Feature Importance 산출**
   - SHAP values (mean |SHAP|) 또는 permutation importance
   - 접속 피처와 성장 피처의 상대 중요도를 함께 확인 (접속이 상위면 성장×접속 결합이 작동한다는 증거)
   - 상위 2~3개 핵심 피처 선정
3. **단순 Rule 도출**
   - 접속 항을 포함한 규칙, 예: `ΔcumEXP ≈ 0 AND Δunion_level ≈ 0 AND access_active_months >= 2`
   - 임계값은 핵심 피처 분포의 분위수에서 후보 → grid search
4. **Rule 단독 평가** (분류기 성능과 별도)
   - Precision, Recall, F1, **FPR (오타겟팅률)**, ROC-AUC
5. **Threshold sweep**
   - 임계값 변경에 따른 Precision / Recall / FPR 곡선
   - Precision-Recall curve로 운영 임계값 권고

## 수용 기준

**Precision > 0.95 AND FPR < 5%** (디렉터의 "일반 유저 피해 최소화" 요구 직접 반영)

## 출력

- Feature Importance 차트 (SHAP summary plot 또는 bar chart) — 접속·성장 피처 비교 포함
- 최종 Rule 정의 (임계값 포함, 접속 항 포함)
- 평가 지표 테이블 (Precision / Recall / F1 / FPR / ROC-AUC)
- Threshold sweep 시각화 (PR curve, FPR-Recall trade-off)
- 운영 임계값 권고 (디렉터 타겟팅 시나리오 별)

## 주의

- 분류기 자체 성능 보고는 부차적 (importance 추출의 sanity check 용도)
- Rule의 단순성 ↔ 성능 trade-off를 threshold sweep으로 명시적으로 제시
- **Ground truth 없음**: 모든 지표는 pseudo-label 대비값이다. FPR은 휴면/일반 캐릭터 오분류율로 해석한다.
- 접속을 **사전 필터가 아니라 입력 피처**로 둔다 — 사전 필터하면 외부 검증 변수가 표본에 조건화되어 순환이 발생한다.
