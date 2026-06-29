# 제출물 - 공개 OpenAPI 기반 메이플스토리 주차 유저 후보군 탐지

응용데이터분석 텀프로젝트 제출물입니다. 제출 요구사항인 분석 데이터 파일, 주피터 노트북, 최종 보고서로 구성되어 있습니다.

## 구성

```text
submission/
├─ 김연길_2022105449_데이터분석_보고서.pdf  # 최종 보고서, A4 6p
├─ requirements.txt               # 분석 노트북 실행용 Python 패키지 목록
├─ assets/
│  └─ NanumSquareNeo-bRg.ttf      # 노트북 한글 폰트
├─ data/                          # 분석에 사용한 데이터
│  ├─ main_characters.csv         # 수집 본캐 2,000명, 레벨 270-290, 5계열 균등
│  ├─ monthly_snapshots_raw.csv   # 12개월 월별 원본 스냅샷
│  ├─ features_monthly.csv        # 캐릭터별 월평균 변화량 피처
│  ├─ exp_requirement_table.csv   # 누적경험치 환산 참조표
│  ├─ cluster_labels.csv          # H1 산출 라벨, is_stagnant_cluster 380명
│  ├─ h3_rule_eval.csv            # H3 산출 rule 예측 결과
│  └─ h3_metrics.csv              # H3 평가 지표
├─ scripts/                       # 데이터 수집 코드, Nexon OpenAPI 사용
│  ├─ collect_main_characters.py
│  └─ collect_features.py
├─ eda/
│  └─ eda.ipynb
├─ h1_clustering/
│  ├─ feature_selection.ipynb
│  ├─ h1_clustering.ipynb
│  └─ optimal_feature_set.json
├─ h2_distribution/
│  ├─ h2_distribution.ipynb
│  ├─ run_analysis.py
│  ├─ results.json
│  └─ RESULTS.md
└─ h3_rule/
   ├─ h3_rule.ipynb
   └─ operating_point.json
```

각 분석 폴더의 `figures/`에는 노트북 실행으로 생성된 결과 그림이 포함되어 있습니다.

## 재현 범위

이 제출물은 이미 수집된 CSV를 기준으로 분석을 재현합니다. `data/`에는 노트북과 보조 스크립트가 실제로 읽는 파일만 포함했습니다.

- `hexa_fragments.csv` 및 해당 수집 스크립트는 최종 분석에서 사용하지 않아 제외했습니다.
- `exp_requirement_table.csv`는 노트북 직접 입력은 아니지만 `scripts/collect_features.py` 재실행 시 필요한 입력이라 포함했습니다.
- `scripts/`의 데이터 수집 코드는 Nexon OpenAPI 키가 필요합니다. 수집을 다시 하려면 `submission/.env` 또는 실행 환경에 `MAPLE_API_KEY`를 설정해야 합니다.
- 분석 노트북은 API 재수집 없이 `data/`의 CSV만으로 실행됩니다.

## 실행 환경

Python 3.10 이상을 권장합니다. 새 환경에서는 다음 명령으로 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

## 실행 순서

`submission/` 폴더를 작업 디렉터리로 둔 뒤 아래 순서로 노트북을 실행하면 됩니다.

1. `eda/eda.ipynb`
2. `h1_clustering/feature_selection.ipynb`
3. `h1_clustering/h1_clustering.ipynb`
4. `h2_distribution/h2_distribution.ipynb`
5. `h3_rule/h3_rule.ipynb`

명령줄에서 일괄 실행하려면 `submission/` 폴더에서 다음과 같이 실행할 수 있습니다.

```bash
jupyter nbconvert --to notebook --execute eda/eda.ipynb --inplace
jupyter nbconvert --to notebook --execute h1_clustering/feature_selection.ipynb --inplace
jupyter nbconvert --to notebook --execute h1_clustering/h1_clustering.ipynb --inplace
jupyter nbconvert --to notebook --execute h2_distribution/h2_distribution.ipynb --inplace
jupyter nbconvert --to notebook --execute h3_rule/h3_rule.ipynb --inplace
```

검토 시 `submission/`을 별도 위치로 복사한 뒤 위 5개 노트북을 순서대로 실행했으며, 모두 에러 없이 완료되었습니다.

## 결과 요약

- **H1**: 최근 6개월 4-시스템 delta 피처로 K-Means(k=4)를 적용해 성장 정체 후보군 380명(19.0%)을 도출했습니다. DBSCAN 보조 검증에서 후보군 recall 83.0%를 확인했습니다.
- **H2**: H1 후보군은 레벨 구간별 분포 차이가 유의했습니다(p=4.27e-9). 직업 계열별 분포 차이는 유의하지 않았습니다(p=0.142).
- **H3**: H1 채택 피처를 제외한 공개 관측 피처 기반 RandomForest로 H1 후보군을 근사했고, ROC-AUC 0.850과 운영 rule FPR 0.068을 확인했습니다.
