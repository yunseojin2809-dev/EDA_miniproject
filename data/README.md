# data/

원본 `.mat`(약 2~3GB/파일)과 추출 캐시는 용량 문제로 저장소에 포함하지 않습니다.

| 파일 | 설명 | 생성 위치 |
|---|---|---|
| `meta_clean.csv` | 정제 후 셀 목록 (120셀: Batch1 41 / Batch2 39 / Batch3 40), `cycle_life`, 충전 정책 | `src/preprocess.py` |
| `features_cell_level.csv` | 셀 단위 피처 (모두 cycle ≤ 100 데이터만 사용) | `src/features.py` |

## 원본 데이터 받기
MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019) — `2017-05-12`, `2018-02-20`, `2018-04-12` batchdata `.mat` 를 `archive/` 에 둡니다
(`2018-04-03 varcharge` 파일은 사용하지 않음).

```bash
python -m src.preprocess --base <archive/ 가 있는 폴더>     # .mat → cache → data/meta_clean.csv
```
이 CSV 두 개만 있으면 `python -m src.train` 으로 모델링 결과를 재현할 수 있습니다.

## 정제 요약 (139셀 → 120셀)
`cycle_life` 결측 10셀 제거(VarCharge / SLOWCYCLE 등 특수 프로토콜 + 결측) → 논문 공개 제거 목록 9셀 제거(Batch1 80% 미도달 5, Batch3 채널 노이즈 4).
논문의 "Batch1←Batch2 이어 측정 병합"은 **Batch2 사이클 수가 논문 값과 일치하지 않아(예: b2c7 806 vs 662) 검증 실패 → 병합하지 않음**. 5개 쌍의 충전 정책도 서로 달라 같은 셀이라고 볼 근거가 부족.
