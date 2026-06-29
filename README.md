# 메이플스토리 주차 유저 후보군 탐지: 성장 데이터 군집 분석

Nexon OpenAPI 공개 데이터만으로 270~290레벨 본캐 표본에서 성장 정체 기반 **주차 유저 후보군**을 탐색한 응용데이터분석 텀프로젝트입니다.

> 공개 API에는 보스 클리어·메소 획득·거래 기록이 없으므로, 이 저장소의 결과는 확정 제재 라벨이 아닙니다.
> 운영 검토에 참고할 수 있는 후보군과 설명 가능한 패턴을 만드는 것이 목적입니다.

---

## 최종 보고서

- **PDF**: [`docs/report.pdf`](docs/report.pdf)
- **Markdown**: [`docs/REPORT.md`](docs/REPORT.md)

---

## 핵심 요약

| 항목 | 내용 |
|---|---|
| 분석 표본 | 레벨 270~290 본캐 2,000명 |
| 표본 조건 | 5개 직업 계열 균등, 전투력 5,000만 이상, 12개월 중 10개월 이상 접속 |
| 분석 기간 | 2025-06~2026-05 월별 스냅샷 (H1 핵심 분석: 최근 6개월) |
| 클러스터링 피처 | 전투력·HEXA·유니온·어센틱 심볼의 최근 6개월 변화량 |
| 주차 후보 라벨 | `is_stagnant_cluster` — **380명 (19.0%)** |

---

## 연구 가설과 결과

| | 가설 | 결과 |
|---|---|---|
| H1 | 최근 6개월 성장 지표가 낮은 활성 캐릭터들이 일반 성장 캐릭터와 구분되는 군집으로 묶이는가? | **지지** — K-Means(k=4)로 전투력 감소+4-시스템 저재투자 동시 발생 후보군 380명(19.0%) 식별, DBSCAN 보조 검증(355명 정합) |
| H2 | 후보군 비율이 레벨 구간과 직업 계열에 따라 다른가? | **부분 지지** — 레벨 구간 유의(p=4.27e-9, 280~285 집중), 직업 계열 무유의(p=0.142) |
| H3 | 후보군을 H1에서 쓰지 않은 공개 지표만으로 단순 규칙으로 근사할 수 있는가? | **제한적 지지** — RandomForest 5-fold ROC-AUC 0.850, 보수 운영 규칙 FPR 0.068 |

---

## 주요 수치

### H1 — 후보군 프로파일

| 항목 | 후보군(cluster 1) | 비후보군 |
|---|---:|---:|
| 인원 | 380명 (19.0%) | 1,620명 |
| 전투력 중앙값 | 88.0M | ~ |
| 전투력 6개월 변화량 중앙값 | **−2.47M** (감소) | 양(+) |
| 접속 활성도 | 비후보군과 동일 (휴면 아님) | — |

클러스터링 피처 4개:

| Alias | 정의 | 시스템 |
|---|---|---|
| `cp_slog` | `sign(Δcp)·log1p(\|Δcp\|)` | 전투력 |
| `hexa_avg` | `clip≥0(avg_monthly_delta_hexa)` | HEXA 코어 |
| `union_log` | `log1p(clip≥0(Δunion_level))` | 유니온 |
| `auth_log` | `log1p(clip≥0(Δauthentc_symbol))` | 어센틱 심볼 |

### H2 — 분포 검정

| 속성 | χ² | p | Cramer's V | 판정 |
|---|---:|---:|---:|---|
| 레벨 구간 | 38.54 | 4.27e-9 | 0.139 | 유의 |
| 직업 계열 | 6.89 | 0.142 | 0.059 | 무유의 |

### H3 — 운영 규칙

```
정체(stagnant) IF  log1p_recent3_delta_cumexp <= 30.84
                AND combat_power_latest <= 70,000,000
                AND access_active_months >= 2
```

- RandomForest 5-fold OOF ROC-AUC: **0.8495**
- 운영점 성능 (vs H1 라벨 380명, in-sample): Precision 0.500 / Recall 0.289 / FPR **0.068**

---

## 저장소 구조

```text
maple_parking_detect/
├── scripts/              # Nexon OpenAPI 수집 스크립트
├── eda/                  # 탐색적 분석 노트북
├── h1_clustering/        # 피처 선택 + K-Means/DBSCAN 군집 분석
├── h2_distribution/      # 레벨 구간·직업 계열 분포 검정
├── h3_rule/              # RandomForest 중요도와 설명 가능한 규칙 근사
├── docs/                 # 보고서 (PDF + Markdown) + 결과 그림
├── submission/           # 제출 패키지 (노트북 + 데이터 + 보고서)
├── assets/               # 한글 폰트
└── data/                 # 로컬 데이터 CSV (Git 제외)
```

---

## 재현 방법

`.env`에 Nexon OpenAPI 키 설정:

```
MAPLE_API_KEY=...
```

패키지 설치:

```bash
pip install requests pandas python-dotenv scikit-learn scipy matplotlib seaborn statsmodels numpy openpyxl jupyter
```

수집 → 분석 순서:

```bash
# 1. 데이터 수집
python scripts/collect_main_characters.py
python scripts/collect_features.py --refresh-raw

# 2. H1 군집 분석
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb

# 3. H2 분포 검정
python h2_distribution/run_analysis.py
jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb

# 4. H3 규칙 근사
jupyter nbconvert --to notebook --execute --inplace h3_rule/h3_rule.ipynb
```

> 이미 수집된 CSV가 있으면 수집 단계를 건너뛰고 분석부터 실행할 수 있습니다.
> 제출 패키지(`submission/`)는 수집 없이 분석만 재현 가능한 독립 환경입니다.

---

## 해석 범위

- 확정 주차 유저 탐지가 아니라 **공개 API 기반 후보군 탐색**입니다.
- 보스 클리어·메소 획득·거래 기록은 공개 API에 없어 직접 검증하지 못했습니다.
- 결론은 레벨 270~290, 전투력 5,000만 이상, 10/12개월 이상 접속한 active/capable 본캐 표본 안에서만 유효합니다.
- H3 규칙 평가는 held-out 외부 검증이 아닌 동일 표본 in-sample 적합도입니다.

---

소프트웨어융합학과 3학년 김연길 (2022105449) · 응용데이터분석 텀프로젝트 2026-1
