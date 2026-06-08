# Maple Parking Detect

메이플스토리 Nexon OpenAPI 공개 데이터만으로 270~290레벨 본캐 표본에서 성장 정체 기반 "주차 유저 후보군"을 탐색한 응용데이터분석 프로젝트입니다.

공개 API에는 주간 보스 클리어, 메소 획득량, 거래 기록이 없으므로 이 저장소의 결과는 확정 제재 라벨이 아닙니다. 분석 목적은 공개적으로 관측 가능한 성장 이력과 접속 흔적만으로 운영 검토에 참고할 수 있는 후보군과 설명 가능한 패턴을 만드는 것입니다.

## 보고서

최종 보고서는 [docs/REPORT.md](docs/REPORT.md)에 정리되어 있습니다.

## 핵심 요약

- 분석 표본: 레벨 270~290 본캐 2,000명
- 표본 조건: 5개 직업 계열 균등, 전투력 5,000만 이상, 최근 12개월 중 10개월 이상 접속
- 분석 기간: 2025-06~2026-05 월별 스냅샷, H1 핵심 분석은 최근 6개월(2025-12~2026-05)
- 핵심 피처: 전투력, HEXA, 유니온, 어센틱 심볼의 최근 6개월 변화량
- 최종 후보 라벨: `is_stagnant_cluster`, 380명(19.0%)

## 연구 질문

내부 로그 없이 공개 OpenAPI로 관측 가능한 성장·접속 신호만으로, 주차 유저로 의심되는 후보군을 통계적으로 식별하고 설명할 수 있는가?

## 가설과 결과

| 구분 | 질문 | 결과 |
|---|---|---|
| H1 | 최근 6개월 주요 성장 지표가 낮은 활성 캐릭터들이 일반 성장 캐릭터와 구분되는 군집으로 묶이는가? | 지지. K-Means(k=4)로 전투력 감소와 4개 성장 시스템 저재투자가 동시에 나타나는 후보군 380명(19.0%)을 식별 |
| H2 | 후보군 비율이 레벨 구간과 직업 계열에 따라 다른가? | 부분 지지. 레벨 구간은 유의(280~285 집중), 직업 계열은 무유의 |
| H3 | 후보군을 군집화에 쓰지 않은 다른 공개 지표만으로 설명 가능한 규칙으로 근사할 수 있는가? | 제한적 지지. RandomForest 5-fold ROC-AUC 0.850, 보수 규칙 FPR 0.068 |

## 주요 결과

### H1: 주차 후보군 식별

최근 6개월 변화량을 기준으로 다음 4개 성장 축을 사용했습니다.

| Alias | 의미 | 시스템 |
|---|---|---|
| `cp_slog` | 전투력 변화량 signed-log | 전투력 |
| `hexa_avg` | HEXA 레벨 변화량 | HEXA |
| `union_log` | 유니온 레벨 변화량 log | 유니온 |
| `auth_log` | 어센틱 심볼 변화량 log | 어센틱 심볼 |

K-Means(k=4) 결과, cluster 1이 주차 후보군으로 해석되었습니다.

| 항목 | 값 |
|---|---:|
| 후보군 수 | 380명 |
| 후보군 비율 | 19.0% |
| 평균 전투력 | 88.0M |
| 전투력 6개월 변화량 중앙값 | -2.47M |
| DBSCAN 보조 검증 | K-Means 후보 380명 중 355명이 동일 밀도 군집으로 정합 |

### H2: 분포 검정

후보 비율은 레벨 구간에서 유의하게 달랐고, 직업 계열에서는 유의하지 않았습니다.

| 속성 | 결과 |
|---|---|
| 레벨 구간 | 유의. χ²=38.54, p=4.27e-9, Cramer's V=0.139 |
| 직업 계열 | 무유의. χ²=6.89, p=0.142, Cramer's V=0.059 |

레벨 구간별로는 `280~285` 구간에 후보군이 집중되고, `286~290` 구간에서는 희박했습니다. 이는 주차 후보가 특정 직업보다 성장 단계와 더 관련될 가능성을 시사합니다.

### H3: 설명 가능한 규칙 근사

H1 군집화에 사용한 4개 변화량 축을 제외하고, 상태값과 누적 경험치 변화량만으로 후보군을 근사했습니다.

- RandomForest 5-fold out-of-fold ROC-AUC: 0.8495
- 주요 신호: 최근 3개월 누적 경험치 증가량, 현재 전투력 절대값
- 보수 운영 규칙: `log1p_recent3_delta_cumexp <= 30.84 AND combat_power_latest <= 70,000,000 AND access_active_months >= 2`
- H1 후보 라벨 기준 성능: Precision 0.500, Recall 0.289, FPR 0.068

단, 규칙 평가는 held-out 외부 검증이 아니라 동일 표본에서 H1 후보 라벨을 근사한 결과입니다. 실제 주차 탐지력은 내부 로그와 결합한 별도 검증이 필요합니다.

## 해석 범위

- 확정 주차 유저 탐지가 아니라 공개 API 기반 후보군 탐색입니다.
- 보스 클리어, 메소 획득, 거래 기록은 공개 API에 없어 직접 검증하지 못했습니다.
- 결론은 레벨 270~290, 전투력 5,000만 이상, 10/12개월 이상 접속한 active/capable 본캐 표본 안에서만 일반화됩니다.
- 후보군 안에는 정상적인 플레이 패턴의 유저도 포함될 수 있으므로 자동 제재 기준으로 사용할 수 없습니다.

## 저장소 구조

```text
maple_parking_detect/
├── scripts/                 # Nexon OpenAPI 수집 스크립트
├── eda/                     # 탐색적 분석 노트북
├── h1_clustering/           # 피처 선택, K-Means/DBSCAN 군집 분석
├── h2_distribution/         # 레벨 구간/직업 계열 분포 검정
├── h3_rule/                 # RandomForest 중요도와 설명 가능한 규칙 근사
├── docs/                    # 최종 보고서와 프로젝트 문서
├── docs/figure/             # 보고서용 그림
├── assets/                  # 폰트 및 참고 자산
└── data/                    # 로컬 생성 데이터, Git 제외
```

## 재현 방법

`.env`에 Nexon OpenAPI 키를 설정합니다.

```text
MAPLE_API_KEY=...
```

필요 패키지를 설치합니다.

```bash
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn statsmodels numpy openpyxl jupyter
```

수집 및 분석을 순서대로 실행합니다.

```bash
python scripts/collect_main_characters.py
python scripts/collect_features.py --refresh-raw
python scripts/collect_hexa_fragments.py
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
python h2_distribution/run_analysis.py
jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb
jupyter nbconvert --to notebook --execute --inplace h3_rule/h3_rule.ipynb
```

API 호출은 수십만 회 발생할 수 있으며, 수집 스크립트는 초당 호출 수를 제한합니다. 이미 생성된 로컬 CSV가 있으면 불필요한 재수집을 피하는 것이 좋습니다.

## 보안 및 데이터 주의

- `.env`와 API 키는 커밋하지 않습니다.
- `data/`의 CSV 파일은 대용량이거나 캐릭터 단위 연구 데이터를 포함할 수 있어 Git에서 제외합니다.
- 본 저장소의 공개 산출물은 집계 결과와 재현 가능한 분석 코드 중심으로 구성합니다.
