# Data

원본 데이터는 용량(약 8GB)이 커서 저장소에 포함하지 않습니다.

- 출처 : Kaggle — [Data-driven prediction of battery cycle life](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)
- 원논문 : Severson et al. (2019), Nature Energy

| 파일 | Batch | 사용 |
| --- | --- | --- |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | Batch 1 | 학습 (CV + Hold-out) |
| 2018-02-20_batchdata_updated_struct_errorcorrect.mat | Batch 2 | 테스트 |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | Batch 3 | 추가 테스트 |
| 2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat | extra | 미사용 (다른 실험) |

## 재현 방법

```bash
# 1) 위 .mat 파일을 data/raw/ 에 저장
# 2) 셀 단위 추출 + Batch 1·2 병합 + EOL 미도달 셀 제거 → data/processed/batch_b*.pkl
python src/preprocess.py --data_dir data/raw --out_dir data/processed
# 3) 셀 단위 피처 생성 → data/processed/features.csv
python src/features.py --in_dir data/processed --out data/processed/features.csv
```
