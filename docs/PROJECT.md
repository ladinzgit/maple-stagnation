# 응용데이터분석 텀프로젝트 — 메이플스토리 주차 유저 클러스터링

> **과목**: 응용데이터분석 | **학과**: 소프트웨어융합학과 3학년 | **학번**: 2026123456 | **이름**: 박상근

---

## 한 줄 요약

메이플스토리에서 경제 왜곡을 유발하는 **'주차 유저'를 클러스터링으로 검출**하고, 이들의 행동 특징을 데이터 기반으로 분석하여 디렉터가 직면한 타겟팅 문제에 대한 **Data-Driven 대안**을 제시한다.

---

## 현재 진행 상황 (2026-05-26 기준)

| 단계 | 상태 | 비고 |
|---|---|---|
| 주제 확정 및 가설 설계 | ✅ 완료 | 가설 3개 확정 |
| Nexon OpenAPI 탐색 및 스크립트 개발 | ✅ 완료 | v2 재설계 완료 |
| 본캐릭터 샘플링 | 🔄 재수집 중 | 260~285 / 직업별 균등 / **2,000명** 목표 (H3 안정성 확보 위해 상향) |
| 월별 피처 수집 (12개월 스냅샷) | ⏳ 재수집 완료 후 실행 | `collect_features.py` (스냅샷 기간 24→12 단축) |
| 데이터 전처리 및 EDA | ⏳ 새 데이터 기준 재실행 필요 | 이전 EDA 결과는 아래 참고 |
| 클러스터링 (가설 1) | ⏳ 예정 | `h1_clustering/h1_clustering.ipynb` |
| 카이제곱 검정 (가설 2) | ⏳ 예정 | `h2_distribution/h2_distribution.ipynb` |
| Feature Importance 및 Rule 평가 (가설 3) | ⏳ 예정 | `h3_rule/h3_rule.ipynb` |
| 보고서 작성 | ⏳ 예정 | — |

---

## 문제 배경

### 디렉터 발언 (라이브 방송 직접 언급)

> *"성장 수준을 고정하고 시장에 공급하는 경우, 소위 주차 유저들의 메소 생산량만 줄이려고 해봤지만 그분들만 타겟팅해서 줄이는 게 현재 시점에서는 어려웠다."*
> — 메이플스토리 디렉터, 라이브 방송 중

### 주차 유저란?

- **정의**: 캐릭터의 성장(레벨업, 전투력 향상)을 의도적으로 멈추고, 특정 성장 구간에 고정된 채 **주간 보스 레이드를 반복**하여 **메소(게임 내 화폐)를 대량 생산**하는 유저
- **문제**: 대규모 메소 공급이 인플레이션을 유발하며 게임 경제 밸런스를 교란함
- **현재 한계**: 일반 유저와 행동 패턴이 외관상 유사해 **규칙 기반 타겟팅이 어려운 상황**

---

## 연구 목적

1. Nexon OpenAPI 데이터로 **주차 유저 집단을 클러스터링으로 검출**
2. 클러스터별 통계 분석으로 **주차 유저가 집중되는 레벨 구간·직업군을 특정**
3. 분석 결과로 **낮은 오분류율을 유지하는 Data-Driven 타겟팅 Rule** 제안

---

## 가설 3개 (방법론 각 1개: 비지도 / 통계 / 지도)

세 가설은 주차 유저 탐지 문제를 **발견(비지도) → 검증(통계) → 운영화(지도)** 흐름으로 접근한다.

### 가설 1 (비지도 학습) — 클러스터링으로 주차 유저 집단 분리 가능성

> **"주차 유저는 일반 유저와 성장 변화량 feature 공간에서 구별되는 군집을 형성한다."**

- **검증 방법**: K-Means + DBSCAN
  - K-Means: k=2~8, Elbow / Silhouette Score로 최적 k 선정
  - DBSCAN: min_samples=10, eps 자동 탐지 (k-distance plot)
- **주요 Feature** (Feature Set A):
  - `avg_monthly_delta_level` — 레벨 변화량 (파킹 핵심 신호)
  - `avg_monthly_delta_combat_power` — 전투력 변화량 (Winsorize P5-P95)
  - `avg_monthly_delta_union_level` — 유니온 변화량 (clip ≥ 0)
  - `avg_monthly_delta_authentic_symbol` — 어센틱심볼 성장
  - `arcane_stagnant` — 아케인심볼 정체 이진 변수 (미포화 & delta=0)
- **수용 기준**:
  - K-Means **Silhouette ≥ 0.4** (군집 분리 품질)
  - K-Means/DBSCAN **ARI ≥ 0.7** (알고리즘 robustness — 단일 알고리즘 artifact 배제)
- **출력**: `data/cluster_labels.csv` (H2/H3 입력)

### 가설 2 (통계) — 주차 유저는 특정 레벨 구간·직업 계열에 불균형 집중 분포

> **"주차 유저 군집은 전체 유저 대비 특정 레벨 구간 및 직업 계열에서 통계적으로 유의하게 높은 비율로 나타난다."**

- **검증 방법**: Chi-Square Test (α = 0.05)
- **검정 대상**:
  - 교차표 1: `cluster_label × level_band` (260~269 / 270~279 / 280~285)
  - 교차표 2: `cluster_label × class_group` (전사 / 마법사 / 궁수 / 도적 / 해적)
- **사후 분석**: 표준화 잔차(standardized residuals)로 |z| > 2 셀 식별 → 파킹 over/under-representation 핫스팟 특정
- **사전 조건**: 모든 셀 기대 빈도 ≥ 5 (EDA Sec 5에서 확인 완료)
- **입력**: `data/cluster_labels.csv`
- **수용 기준**: 두 교차표 중 최소 1개에서 p < 0.05 + Cramer's V로 효과 크기 보고

### 가설 3 (지도 학습) — Rule-Based 기준의 낮은 오분류율

> **"Feature Importance 기반 Rule-Based 타겟팅 기준은 주차 유저를 Precision > 0.95, FPR < 5%로 식별한다."**

- **검증 방법**:
  1. Random Forest / XGBoost 학습 (H1 클러스터 레이블 = pseudo-label, 5-fold stratified CV)
  2. SHAP 또는 permutation importance로 상위 2~3개 핵심 피처 선정
  3. 임계값 기반 단순 Rule 도출 (예: `Δlevel < a AND Δunion_level < b AND arcane_stagnant = 1`)
  4. Rule 단독 평가: Precision, Recall, F1, **FPR (오타겟팅률)**, ROC-AUC
  5. Threshold sweep으로 Precision-Recall trade-off 시각화 → 운영 임계값 권고
- **입력**: H1 클러스터 레이블 (pseudo-label) + 전체 피처
- **수용 기준**: **Precision > 0.95 AND FPR < 5%** (디렉터의 "일반 유저 피해 최소화" 요구 반영)
- **전제**: H1 수용 기준(Silhouette ≥ 0.4, ARI ≥ 0.7) 충족 시 pseudo-label 신뢰도 확보

---

## 데이터 수집 설계 (v2, 2026-05-26 재설계)

### 재설계 배경

v1(유니온 랭킹 층화 샘플링, 1,497명, 전 레벨)으로 H1 예비 실험을 수행한 결과:
- 파킹 클러스터: **52명 / 1,380명 (3.8%)** — 절대 수가 너무 적음
- **285+ 레벨 오염**: 280→285 레벨업에 70~99시간, 285→290에 160~242시간 소요 → 285+ 활성 유저도 `delta_level ≈ 0` → 파킹 신호와 구분 불가
- 샘플의 58.5%(876명)가 레벨 286+ → 비파킹 클러스터에 노이즈 대거 혼입

→ **레벨 260~285로 범위 제한 + 직업 계열별 균등 배분** 으로 재설계

### 샘플링 방법 (v2)

| 항목 | v1 (구) | v2 (신) |
|---|---|---|
| 엔드포인트 | `ranking/union` | `ranking/overall` (class 파라미터) |
| 레벨 범위 | 전체 (주로 260+) | **260 ~ 285** |
| 목표 인원 | 1,300명 (무작위) | **2,000명 (5계열 × 400명)** |
| 레벨 균등 | 없음 | 260~269 / 270~279 / 280~285 각 133/133/134명 (계열별) |
| 페이지 탐색 | 5 tier × 5 random pages | **직업명별 이진 탐색** (260~285 페이지 범위 확정) |

### API 흐름

```
ranking/overall (class=직업명, page)
  → id (OCID 조회)
  → user/union-raider (본캐 판별)
  → [합격 시] features 수집: character/basic, character/stat×7, user/union, character/symbol-equipment
```

### 본캐 판별

`user/union-raider`의 `union_block` 중 `max(block_level) <= character_level` → 본캐. 더 높은 블록이 있으면 부캐로 제외.

### 피처 수집 (collect_features.py, 변경 없음)

| 항목 | 내용 |
|---|---|
| 수집 기간 | 2025-06 ~ 2026-05 (12개월) — 24개월에서 단축 |
| 호출 수 | 캐릭터당 × 12개월 × 10회 (~2,000명 × 120회 ≈ 240,000회, ~10분) |
| 변화량 계산 | 유효 월이 2개 이상인 구간에서 `avg_monthly_delta` |

### 직업 계열 (CLASS_GROUP_MAP)

| 계열 | 직업 (12/11/8/8/8개) |
|---|---|
| 전사 (13) | 히어로, 팔라딘, 다크나이트, 소울마스터, 미하일, 블래스터, 데몬슬레이어, 데몬어벤져, 아란, 카이저, 아델, 렌, 제로 |
| 마법사 (10) | 아크메이지(불,독), 아크메이지(썬,콜), 비숍, 플레임위자드, 배틀메이지, 에반, 루미너스, 일리움, 라라, 키네시스 |
| 궁수 (7) | 보우마스터, 신궁, 패스파인더, 윈드브레이커, 와일드헌터, 메르세데스, 카인 |
| 도적 (9) | 나이트로드, 섀도어, 듀얼블레이더, 나이트워커, 팬텀, 카데나, 칼리, 호영, **제논** |
| 해적 (8) | 바이퍼, 캡틴, 캐논마스터, 스트라이커, 메카닉, 은월, 엔젤릭버스터, 아크 |

> **제논 처리 결정**: 제논은 도적/해적 하이브리드 직업. **도적 단일 분류** 채택.
> - 근거: Nexon 원분류(2013 출시), LUK 메인스탯, H2 Chi-Square 독립성 가정 충족 필요
> - 한계: 해적 특성 손실 → 향후 sensitivity 분석으로 제논만 해적 재분류 후 H2 재실행하여 결과 안정성 확인 가능
> **분류 변경 사항** (이전 분류와 차이): 렌(해적→전사), 미하일(도적→전사), 카데나(궁수→도적), 칼리(마법사→도적), 호영(전사→도적), 은월(도적→해적), 제논(도적 단일 유지)

---

## EDA 주요 발견 사항 (v1 데이터 기준 — 새 데이터 수집 후 재실행 필요)

> **주의**: 아래 수치는 v1(유니온 랭킹 샘플링, 전 레벨, 1,463행)에서 얻은 결과. v2 재수집 후 변경될 수 있음.

### 전처리 결정 (v2에도 동일 적용 예정)

| 처리 항목 | 방법 | 근거 |
|---|---|---|
| NaN 34행 제거 | listwise deletion | MAR 확인 (레벨 구간별 편향 없음) |
| delta_cp 음수 (21.8%) | Winsorize(P5-P95) | 0 클리핑 시 "성장 없음" 오해 발생 |
| delta_union 음수 3행 | → 0 처리 | 계정 초기화 케이스 |
| delta_arcane 음수 1행 | → 0 처리 | API 오류 가능성 |
| delta_arcane (65.8% 점질량) | `arcane_stagnant` 이진 변수 대체 | 아케인 포화 유저 84% → 연속 변수 부적합 |

### 클러스터링 피처 세트 (Feature Set A — 확정)

```python
CLUSTER_FEATURES_A = [
    'avg_monthly_delta_level',
    'avg_monthly_delta_combat_power',    # Winsorize P5-P95 적용
    'avg_monthly_delta_union_level',     # clip(lower=0)
    'avg_monthly_delta_authentic_symbol',
    'arcane_stagnant',                   # delta_arcane 이진화 대체
]
```

> `normalized_delta_level`(경험적 P75 정규화)은 파킹 유저 집중 구간(260~270) 편향(Spearman r=−0.429) 때문에 **제외 확정**.

### VIF 확인 결과

| 피처 | VIF |
|---|---|
| Δlevel | 1.47 |
| Δcombat_power | 1.88 |
| Δunion_level | 1.66 |
| Δarcane_symbol | 1.99 |
| Δauthentic_symbol | 3.41 |

전원 ≤ 5 → PCA 불필요, StandardScaler 후 직접 클러스터링 가능.

### stagnation_score 분포

- 5점(전 피처 정체) 54명(3.7%)
- 4점 이상 216명(14.8%)

---

## H1 예비 실험 결과 (v1 데이터, 참고용)

> v2 데이터 수집 후 `h1_clustering.ipynb` 재실행 필요.

`h1_clustering/h1_clustering.ipynb` — level ≥ 260 필터 후 1,380명 대상, Feature Set A 사용.

| 항목 | 값 |
|---|---|
| 최적 k | 5 (Silhouette=0.4517) |
| 파킹 클러스터 | cluster #1 — 52명 (3.8%) |
| 파킹 클러스터 특징 | arcane_stagnant=1.0, delta_level≈0.1, delta_cp≈−28,000 |
| 레벨 분포 | 260s 71.2%, 270s 26.9% |
| stagnation=5 집중도 | 36/36명 (100%) |
| DBSCAN 일치율 | 51/52명 (98%, eps=0.874 min_samples=10) |

**재설계 이유**: 52명은 절대 수가 너무 적음. v2에서 260~285 집중 + 레벨 빈 균등 배분으로 파킹 유저 비율 및 절대 수 증가 기대.

---

## 분석 방법론 스택

| 단계 | 방법 | 도구 |
|---|---|---|
| 데이터 수집 | Nexon OpenAPI (400 req/s, 30 스레드) | Python, requests, ThreadPoolExecutor |
| 전처리·EDA | Winsorize, 이진화, VIF, 분포 시각화 | pandas, scipy, matplotlib, seaborn |
| 클러스터링 (H1) | K-Means (Elbow/Silhouette) + DBSCAN | scikit-learn |
| 분포 검정 (H2) | Chi-Square Test (α=0.05) | scipy.stats |
| 분류 평가 (H3) | Random Forest, XGBoost, ROC-AUC | sklearn, xgboost |

---

## 파일 구조

```
maple_parking_detect/
├── data/                           # gitignored
│   ├── main_characters.csv         #   본캐릭터 (v2: 260~285, 직업별 균등, 2,000명)
│   ├── features_monthly.csv        #   12개월 월별 피처
│   └── cluster_labels.csv          #   H1 클러스터 레이블 (H2/H3 입력)
├── scripts/
│   ├── collect_main_characters.py  #   v2: 종합 랭킹 + 이진 탐색 + 레벨 빈 균등
│   └── collect_features.py         #   12개월 스냅샷 (SNAPSHOT_MONTHS 상수)
├── eda/
│   └── eda.ipynb                   #   EDA 전용 (실험 코드 추가 금지)
├── h1_clustering/
│   └── h1_clustering.ipynb         #   H1 실험 (v2 데이터 기준 재실행 필요)
├── h2_distribution/
│   └── h2_distribution.ipynb       #   H2 실험 (작성 예정)
├── h3_rule/
│   └── h3_rule.ipynb               #   H3 실험 (작성 예정)
├── docs/
│   ├── PROJECT.md                  #   이 파일 — 프로젝트 상세 기록
│   ├── PLAN.md                     #   collect_features.py v2 설계 이력 (참고용)
│   └── level.txt                   #   레벨별 경험치 참고 데이터
└── .env                            # API 키 (미포함)
```

---

## 일정 계획

| 주차 | 작업 내용 | 완료 여부 |
|---|---|---|
| 1~2주차 | 주제 확정 및 관련 연구 조사 | ✅ |
| 3~4주차 | Nexon OpenAPI 탐색 및 데이터 수집 스크립트 개발 | ✅ |
| 5~6주차 | 데이터 전처리 및 EDA (v1) | ✅ |
| 7주차 | 데이터 재수집 (v2: 260~285, 직업별 균등) | 🔄 진행 중 |
| 7~8주차 | EDA 재실행 + 클러스터링 (가설 1) | ☐ |
| 9주차 | 카이제곱 검정 (가설 2) | ☐ |
| 10주차 | Feature Importance 및 Rule 성능 평가 (가설 3) | ☐ |
| 11주차 | 보고서 초안 작성 | ☐ |
| 12주차 | 보고서 최종 검토 및 제출 | ☐ |

---

## 보고서 작성 체크리스트

- [ ] 초록 (Abstract) 작성
- [ ] 서론: 배경, 문제 제기, 가설 정의
- [ ] 관련 연구 조사 (0.5~1p)
- [ ] 본론: 가설 1 — 클러스터링 결과 및 그래프
- [ ] 본론: 가설 2 — 카이제곱 검정 결과 및 논의
- [ ] 본론: 가설 3 — Rule 성능 평가 및 타겟팅 기준 제안
- [ ] 결론 및 인사이트 (5p 이내 완료 확인)
- [ ] 참고문헌 인용 형식 확인 ([1], [2] 형식)
- [ ] 그림 명칭 하단 표기 확인 ([그림 1])
- [ ] 표 명칭 상단 표기 확인 ([표 1])
- [ ] 글자 크기 9pt 이상 확인

---

## 참고 자료

- [Nexon OpenAPI 공식 문서](https://openapi.nexon.com/)
- 경희대 UXC 학부생 논문 샘플
- Nexon 메이플스토리 공식 커뮤니티 (디렉터 방송 아카이브)

---

## 미결 사항

- [x] History API 수집 제약 확인: 큐브/스타포스 이력은 계정주 본인만 조회 가능 → 제외
- [x] EDA 완료 (v1): arcane_stagnant 이진화, delta_cp Winsorize, VIF 전원 ≤ 5
- [x] Feature Set A 확정: normalized_delta_level 제외 (260~270 편향, Spearman r=−0.429)
- [x] 데이터 재수집 필요 판단: 285+ 레벨 신호 오염, 파킹 클러스터 52명(3.8%)으로 부족
- [x] 수집 스크립트 v2 작성 완료: 종합 랭킹 + 이진 탐색 + 레벨 빈 균등
- [ ] v2 데이터 수집 실행
- [ ] EDA 재실행 (v2 데이터 기준)
- [ ] H1 클러스터링 재실행
- [ ] H2 카이제곱 검정 노트북 작성
- [ ] H3 Rule 평가 노트북 작성
- [ ] 샘플링 편향 보고서 한계 명시 (일반 서버만 수집, world_type=0)
