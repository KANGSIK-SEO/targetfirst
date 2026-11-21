# targetfirst — 쇼핑몰 단골(VIP) 고객 찾기

Codyssey "AI/SW 심화 — 데이터 분석" 미션: 이커머스 멀티모달 데이터(수치/텍스트/범주/날짜/이미지)를
NumPy·Pandas 벡터 연산으로 전처리·분석하고, RFM 기법으로 고객을 세분화해 단골(VIP) 고객을
찾아낸다.

## 1. 프로젝트 개요

- **목표**: 상품 데이터를 EDA하고, 고객 거래 데이터를 RFM(Recency·Frequency·Monetary)
  분석으로 세분화하여 데이터 근거 기반의 비즈니스 전략을 제안한다.
- **핵심 산출물**
  1. `src/pipeline.py` — 전처리/피처엔지니어링/통계/RFM을 담당하는 `DataAnalyzer` 클래스(OOP)
  2. `notebooks/analysis_report.ipynb` — 실행 결과(표·그래프)가 포함된 전체 분석 리포트
  3. 본 `README.md` — 데이터 설명, 실행 방법, 비즈니스 인사이트

## 2. 폴더 구조

```
targetfirst/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_data.py     # 합성 데이터 생성 스크립트
│   ├── pipeline.py          # DataAnalyzer 클래스 (핵심 제출물)
│   └── build_notebook.py    # analysis_report.ipynb 생성 스크립트
├── notebooks/
│   └── analysis_report.ipynb
└── data/
    ├── products.csv          # 상품 마스터 (1,500행 x 7컬럼)
    ├── product_images.npy    # 상품 썸네일 배열 (1500, 16, 16, 3)
    └── transactions.csv      # 고객 거래 내역 (5,104행)
```

## 3. 데이터 설명

**데이터 출처에 대한 안내**: 이 저장소의 `data/`는 Kaggle 이커머스 데이터셋의 전형적인
컬럼 구성(상품명/카테고리/가격/등록일/재고/평점 + 고객 거래 로그)을 그대로 따르되,
실행 환경에서 외부 API 키 없이도 누구나 바로 재현·검증할 수 있도록
`src/generate_data.py`로 **직접 생성한 합성(synthetic) 데이터**다. 생성 로직 전체가
공개되어 있어(MIT 라이선스에 해당하는 이 저장소 코드 그대로), 실제 공개 데이터셋을
사용할 때와 동일한 분석 파이프라인(`DataAnalyzer`)을 100% 동일하게 적용할 수 있다.
(`python src/generate_data.py`로 언제든 동일한 구조의 데이터를 재생성 가능)

### products.csv (상품 마스터, 1,500행 x 7컬럼)
| 컬럼 | 유형 | 설명 |
|---|---|---|
| product_id | 수치 | 상품 고유 ID |
| product_name | 텍스트 | 상품명 (형용사+명사 조합) |
| category | 범주 | 전자제품/의류/식품/뷰티/가구/도서/스포츠/완구 (8종) |
| price | 수치 | 가격(원). 카테고리별 평균이 다르며 일부 이상치 주입 |
| registered_date | 날짜 | 등록일 |
| stock_qty | 수치 | 재고 수량 (6% 의도적 결측) |
| rating | 수치 | 평점 1.0~5.0 (4% 의도적 결측) |

### product_images.npy (이미지 배열)
- `products.csv`와 같은 순서, shape `(1500, 16, 16, 3)`, dtype `uint8`.
- 카테고리별로 평균 밝기/노이즈 강도를 다르게 설계하여, 이미지 통계치(`image_mean`,
  `image_std`)와 가격·카테고리 간의 현실적인 상관관계를 재현했다.

### transactions.csv (고객 거래, 5,104행)
| 컬럼 | 설명 |
|---|---|
| order_id | 주문 고유 ID |
| customer_id | 고객 ID (800명) |
| product_id | 구매 상품 ID |
| order_date | 주문일 (2025-05 ~ 2026-11) |
| quantity | 수량 |
| unit_price | 단가 |
| amount | quantity × unit_price (RFM의 Monetary 계산에 사용) |

## 4. 실행 방법

```bash
cd targetfirst
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 데이터 생성 (이미 data/ 에 생성되어 있으므로 재실행은 선택 사항)
python src/generate_data.py

# 2) 분석 노트북 실행 (표/그래프를 모두 재계산하여 .ipynb에 반영)
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis_report.ipynb

# 또는 직접 열어서 보기
jupyter notebook notebooks/analysis_report.ipynb
```

`DataAnalyzer`만 코드에서 바로 쓰고 싶다면:

```python
from src.pipeline import DataAnalyzer

da = DataAnalyzer(data_path="data").load_data()
da.engineer_image_features()
da.engineer_text_features()
da.handle_missing_values(strategy="group_mean", group_col="category")
rfm = da.calculate_rfm(customer_col="customer_id", date_col="order_date", amount_col="amount")
```

## 5. 분석 파이프라인 설계 (`DataAnalyzer`)

| 메서드 | 역할 |
|---|---|
| `load_data()` | products/images/transactions 로드 |
| `engineer_image_features()` | 이미지 배열에서 `axis=(1,2,3)` 벡터 연산으로 평균/표준편차 추출 (for-loop 없음) |
| `engineer_text_features()` | 상품명에서 단어 수/글자 길이 파생 |
| `handle_missing_values(strategy, group_col, columns)` | 그룹별(카테고리) 평균/중앙값 결측치 대치 |
| `detect_outliers(column, threshold)` | IQR 기반 이상치 탐지 |
| `cap_outliers(column, threshold)` | IQR 경계값으로 클리핑(winsorize), 전/후 분포 비교용 |
| `summary_statistics(columns)` | 평균/중앙값/표준편차/사분위수 |
| `correlation_matrix(columns)` | 상관계수 행렬 |
| `calculate_rfm(customer_col, date_col, amount_col)` | R/F/M 계산 → 4분위 점수화 → 규칙 기반 5개 세그먼트 분류 |

**제약사항 준수**: OpenCV/Pillow, scikit-learn/NLTK, pandas-profiling/sweetviz는
사용하지 않았다. 이미지 통계는 전부 NumPy 배열 연산(`.mean()`, `.std()`,
`np.diff()`)으로 직접 구현했다.

## 6. 비즈니스 인사이트

> 분석 기준일: 2026-11-22 (거래 데이터 내 최종 주문일), 총 고객 800명, 세그먼트 5종

| 세그먼트 | 고객 수 | 비중 | 평균 Recency | 평균 Frequency | 평균 Monetary | 매출 기여도 |
|---|---|---|---|---|---|---|
| VIP | 218 | 27.3% | 16.6일 | 13.4회 | 3,311,639원 | **70.0%** |
| Loyal | 126 | 15.8% | 57.7일 | 6.7회 | 1,083,964원 | 13.2% |
| At Risk | 97 | 12.1% | 278.7일 | 5.9회 | 872,111원 | 8.2% |
| Churned | 204 | 25.5% | 174.0일 | 2.6회 | 296,476원 | 5.9% |
| New | 155 | 19.4% | 15.1일 | 1.5회 | 183,529원 | 2.8% |

**[인사이트 1] VIP 고객 27%가 전체 매출의 70%를 차지한다**
- (근거) VIP 세그먼트는 고객 수 기준 27.3%(218명)에 불과하지만, 1인당 평균 구매액
  (3,311,639원)이 가장 낮은 New 세그먼트(183,529원)의 18배에 달해 전체 매출의
  **70.0%**를 차지한다.
- (실행) VIP 대상 → 전담 등급(멤버십) 혜택·무료배송·우선 재입고 알림 제공 → 이탈 방지
  및 객단가 추가 상승 유도. VIP 1명 이탈 시 매출 손실이 크므로 CS 응대 우선순위를
  최상위로 배정한다.
- (검증) 현재 데이터만으로는 VIP의 "이탈 시 매출 손실 규모"를 시뮬레이션하기 어렵다.
  고객별 생애가치(LTV) 산출을 위해 상품 원가/마진 데이터가 추가로 필요하며, VIP 혜택이
  실제로 이탈을 낮추는지는 혜택 제공 전/후 A/B 테스트로 반증 가능하다.

**[인사이트 2] "At Risk"(12.1%, 97명) — 한때 우수했던 고객의 조용한 이탈**
- (근거) At Risk 세그먼트는 평균 Recency가 278.7일로 전 세그먼트 중 가장 길어 가장
  오랫동안 재구매가 없지만, 평균 Frequency는 5.9회로 완전히 이탈한 Churned(2.6회)보다
  오히려 높다. 즉 과거엔 활발히 구매하던 고객이 소리 없이 떠나고 있는 상태다.
- (실행) At Risk 대상 → 과거 구매 카테고리 기반 맞춤 할인 쿠폰 + "오랜만이에요" 리마인드
  메일 캠페인 → 재방문·재구매율 회복 기대. Churned(완전 이탈, 204명)보다 회수 가능성이
  높으므로 마케팅 예산을 At Risk에 우선 배분하는 것이 비용 대비 효율적이다.
- (검증) 이탈 원인(가격 인상/경쟁사 전환/단순 니즈 소멸)을 구분하려면 설문이나 CS
  문의 로그가 추가로 필요하다. 쿠폰 발송 후 재구매율이 오르지 않는다면 "조용한 이탈"
  가설은 반증된다.

**[인사이트 3] 가격 이상치의 97.5%가 전자제품·가구 두 카테고리에 집중**
- (근거) `price` 컬럼의 IQR(threshold=1.5) 이상치는 201건(전체 1,500건의 13.4%)인데,
  이 중 128건(전자제품)과 68건(가구)을 합치면 196건으로 전체 이상치의 **97.5%**를
  차지한다. 전체 카테고리를 묶어 하나의 IQR 기준을 적용했기 때문에 발생한 현상이다.
- (실행) 가격 이상치 모니터링/프로모션 가격 검수 프로세스를 **카테고리별 IQR 기준**으로
  전환해야 한다(전체 기준 적용 시 전자제품·가구의 정상 고가 상품이 이상치로 오탐되고,
  반대로 저가 카테고리 내부의 실제 이상치는 놓칠 수 있다).
- (검증) 이 201건이 실제 "입력 오류"인지 "정상 고가 상품"인지는 가격 데이터만으로는
  확정할 수 없다. 상품 등록 담당자의 수정 이력(가격 변경 로그)이 추가로 있어야
  오류 건과 정상 고가 상품을 구분할 수 있다.

## 7. 보너스 과제 수행 내역

- **이미지 고급 피처**: `np.diff()`를 이용해 인접 픽셀 간 차분(가로/세로)을 계산하고,
  그 절댓값 평균을 "엣지 강도(edge_strength)"로 정의해 NumPy만으로 구현했다
  (노트북 8절).
- **코호트 분석**: 고객의 첫 구매월을 코호트로 정의하고, 가입 후 경과월별 재구매율을
  히트맵으로 시각화했다 (노트북 9절).
- Plotly 인터랙티브 시각화(보너스 3)는 이번 제출에서는 생략했다.
