# 메이플스토리 주차 유저 후보군 탐지: 성장 데이터 군집 분석

Nexon OpenAPI 공개 데이터만으로 270~290레벨 본캐 표본에서 성장 정체 기반 **주차 유저 후보군**을 탐색한 데이터 분석 프로젝트입니다.

> 공개 API에는 보스 클리어·메소 획득·거래 기록이 없으므로 결과는 확정 제재 라벨이 아닙니다.
> 운영 검토에 참고할 수 있는 후보군과 설명 가능한 패턴을 제안합니다.

---

## Portfolio Summary

이 프로젝트는 공개 API만으로 MMORPG의 고활성·저성장 유저 후보군을 탐색한 데이터 분석 프로젝트입니다.
내부 로그와 제재 라벨이 없는 상황에서, 성장 시스템별 변화량을 proxy feature로 설계하고
비지도 군집화, 분포 검정, 설명 가능한 규칙 근사를 통해 운영 검토 가능한 후보군을 제안했습니다.

### What I demonstrated

- **데이터 수집**: Public API 기반 수집 파이프라인 설계 (멀티스레드, rate limiting, 점진적 저장)
- **피처 엔지니어링**: 도메인 문제(주차 행동)를 분석 가능한 proxy 지표(성장 시스템 delta)로 변환
- **비지도 학습**: K-Means + DBSCAN 기반 군집 분석, 알고리즘 간 교차 검증
- **통계 검정**: 카이제곱 + Cramer's V + Monte Carlo 보정을 통한 분포 분석
- **설명 가능성**: RandomForest 중요도 기반 단순 규칙 도출, FPR 제어 sweep
- **한계 명시**: 데이터 한계·해석 범위·in-sample 한계를 분석 전반에 명시

---

## 주요 결과 시각화

| K-Means 군집 (PCA 2D) | 후보군 레벨 구간 분포 |
|:---:|:---:|
| ![PCA 군집](docs/figure/fig2_pca_clusters.png) | ![레벨 분포](docs/figure/fig4_h2_distribution.png) |

| H3 결정트리 규칙 | 전투력 임계 sweep |
|:---:|:---:|
| ![결정트리](docs/figure/fig7_h3_tree.png) | ![Sweep](docs/figure/fig8_h3_sweep.png) |

---

## 최종 보고서

- **PDF**: [`docs/report.pdf`](docs/report.pdf)
- **Markdown**: [`docs/REPORT.md`](docs/REPORT.md)

---

## 핵심 요약

| 항목 | 내용 |
|---|---|
| 분석 표본 | 레벨 270~290 본캐 2,000명 |
| 표본 조건 | 5계열 균등, 전투력 5,000만 이상, 12개월 중 10개월 이상 접속 |
| 분석 기간 | 2025-06~2026-05 월별 스냅샷 (H1 핵심 분석: 최근 6개월) |
| 클러스터링 피처 | 전투력·HEXA·유니온·어센틱 심볼의 최근 6개월 변화량 |
| 주차 후보 라벨 | `is_stagnant_cluster` — **380명 (19.0%)** |

---

## 연구 가설과 결과

| | 가설 | 결과 |
|---|---|---|
| H1 | 최근 6개월 성장 지표가 낮은 활성 캐릭터들이 일반 성장 캐릭터와 구분되는 군집으로 묶이는가? | **지지** — K-Means(k=4)로 전투력 감소+4-시스템 저재투자 후보군 380명(19.0%) 식별, DBSCAN 보조 검증(355명 정합) |
| H2 | 후보군 비율이 레벨 구간과 직업 계열에 따라 다른가? | **부분 지지** — 레벨 구간 유의(p=4.27e-9, 280~285 집중), 직업 계열 무유의(p=0.142) |
| H3 | 후보군을 H1에서 쓰지 않은 공개 지표만으로 단순 규칙으로 근사할 수 있는가? | **제한적 지지** — RandomForest 5-fold ROC-AUC 0.850, 보수 운영 규칙 FPR 0.068 |

---

## 주요 수치

### H1 — 후보군 프로파일

| 항목 | 후보군 (cluster 1) | 비후보군 |
|---|---:|---:|
| 인원 | 380명 (19.0%) | 1,620명 |
| 전투력 중앙값 | 88.0M | ~ |
| 전투력 6개월 변화량 중앙값 | **−2.47M** (감소) | 양(+) |
| 접속 활성도 | 비후보군과 동일 (휴면 아님) | — |

클러스터링 피처:

| Alias | 정의 | 시스템 |
|---|---|---|
| `cp_slog` | `sign(Δcp)·log1p(\|Δcp\|)` | 전투력 |
| `hexa_avg` | `clip≥0(avg_monthly_delta_hexa)` | HEXA 코어 |
| `union_log` | `log1p(clip≥0(Δunion_level))` | 유니온 |
| `auth_log` | `log1p(clip≥0(Δauthentic_symbol))` | 어센틱 심볼 |

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

| rule | 양성수 | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| loose tree (대조) | 581 | 0.446 | 0.682 | 0.199 |
| **최종 운영점** | **220** | **0.500** | **0.289** | **0.068** |

RandomForest 5-fold OOF ROC-AUC: **0.8495**

---

## 검증

| 검증 | 목적 | 결과 |
|---|---|---|
| k=3~6 민감도 | 후보군이 k=4에만 의존하는지 확인 | k=3~6 전 구간에서 저재투자 군집 유지, k=6 고순도 코어 162명(enrich 8.6) |
| 6mo vs 12mo 윈도우 | 분석 기간 변화에 따른 안정성 확인 | 12mo recall 0.57 → 6mo recall 0.83, 현재 행동 포착에 6mo 우위 |
| DBSCAN 구조 정합 | 알고리즘 독립적 구조 확인 | K-Means 후보 380명 중 355명이 DBSCAN 동일 밀도 군집에 포함 |

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
├── submission/           # 제출 패키지 (노트북 + 보고서)
├── assets/               # 한글 폰트
├── requirements.txt      # Python 패키지 목록
├── run_all.py            # 분석 전체 재현 스크립트
├── Makefile              # make reproduce / make install
└── data/                 # 로컬 데이터 CSV (Git 제외)
```

---

## 재현 방법

`.env.example`을 복사해 API 키 설정:

```bash
cp .env.example .env
# .env 파일에 MAPLE_API_KEY 입력
```

패키지 설치:

```bash
pip install -r requirements.txt
```

### 수집 (최초 1회)

```bash
python scripts/collect_main_characters.py
python scripts/collect_features.py --refresh-raw
```

### 분석 재현

```bash
# 한 번에 전체 실행
python run_all.py

# 또는 make 사용
make reproduce

# 개별 실행
make h1   # K-Means/DBSCAN 군집 분석
make h2   # 분포 검정
make h3   # RandomForest + 규칙 근사
```

> 이미 수집된 CSV가 있으면 수집 단계를 건너뛰고 분석부터 실행할 수 있습니다.
> `submission/` 폴더는 수집 없이 분석만 재현 가능한 독립 패키지입니다.

---

## 해석 범위

- 확정 주차 유저 탐지가 아닌 **공개 API 기반 후보군 탐색**입니다.
- 보스 클리어·메소 획득·거래 기록은 공개 API에 없어 직접 검증하지 못했습니다.
- 결론은 레벨 270~290, 전투력 5,000만 이상, 10/12개월 이상 접속한 active/capable 본캐 표본 안에서만 유효합니다.
- H3 규칙 평가는 held-out 외부 검증이 아닌 동일 표본 in-sample 적합도입니다.

---

소프트웨어융합학과 3학년 김연길 (2022105449) · 응용데이터분석 텀프로젝트 2026-1
