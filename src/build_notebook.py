"""notebooks/analysis_report.ipynb 를 프로그래밍 방식으로 생성하는 스크립트.

셀 내용을 여기서 한 곳에 모아 관리하고, nbformat으로 .ipynb 파일을 생성한다.
실행은 build_notebook.py가 아니라 `jupyter nbconvert --execute`로 수행해
실제 출력(그래프, 표)이 노트북에 임베드되도록 한다.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ------------------------------------------------------------------ #
md(
"""# 쇼핑몰 고객·상품 데이터 분석 — 단골(VIP) 고객 찾기

**미션:** 이커머스 플랫폼의 상품(수치+텍스트+범주+날짜+이미지) 데이터와 거래 데이터를 분석하여
멀티모달 전처리, 이상치 탐지, EDA, RFM 기반 고객 세분화를 수행한다.

- 분석 파이프라인: `src/pipeline.py`의 `DataAnalyzer` 클래스
- 데이터: `data/products.csv`, `data/product_images.npy`, `data/transactions.csv`
  (출처: 실제 Kaggle 이커머스 공개 데이터의 컬럼 구조를 참고해 재현한 합성 데이터.
  자세한 내용은 README.md의 "데이터 설명" 참고)
"""
)

code(
"""import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import DataAnalyzer

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)
"""
)

# ------------------------------------------------------------------ #
md("## 1. 데이터 로드 및 기본 탐색")

code(
"""da = DataAnalyzer(data_path="../data")
da.load_data()

print("products:", da.products.shape)
print("images  :", da.images.shape)
print("tx      :", da.transactions.shape)
"""
)

code('da.products.info()')

code('da.products.describe(include="all").T')

code('da.products.head()')

md(
"""**해석**
- 상품 데이터는 1,500건 x 7개 컬럼(수치형 3개: `price`, `stock_qty`, `rating` /
  범주형: `category` / 텍스트: `product_name` / 날짜형: `registered_date`)으로 구성되어
  있으며, 여기에 `product_images.npy`의 이미지 배열까지 더하면 4가지 데이터 유형이 모두 포함된다.
- `describe()` 결과 `price`의 평균(약 97,500원)이 중앙값(약 39,000원)보다 훨씬 커서
  분포가 오른쪽으로 크게 치우쳐 있음을 알 수 있다 → 고가 카테고리(전자제품/가구)가
  평균을 끌어올리는 구조이며, 이는 4장 이상치 탐지에서 다시 다룬다.
"""
)

code(
"""tx = da.transactions
print(tx.shape)
tx.head()
"""
)

md(
"""- 거래 데이터는 5,104건으로 `customer_id`, `order_date`, `amount`(=quantity*unit_price)를
  포함해 RFM 계산 조건을 만족한다. 분석 기준일(Recency 기준일)은 최종 거래일로 정의한다.
"""
)

# ------------------------------------------------------------------ #
md(
"""## 2. 멀티모달 피처 엔지니어링

- **이미지 → 수치**: `image_mean`(평균 밝기), `image_std`(명암 대비)를 NumPy 축 연산
  `images.mean(axis=(1,2,3))`으로 한 번에 계산한다 (반복문 없음).
- **텍스트 → 수치**: 상품명의 단어 수, 글자 길이를 파생 변수로 추가한다.
"""
)

code(
"""da.engineer_image_features()
da.engineer_text_features()
da.products[["product_name", "category", "image_mean", "image_std",
             "name_word_count", "name_length"]].head()
"""
)

code(
"""fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(da.images[i])
    axes[i].set_title(f"{da.products.loc[i,'category']}\\nmean={da.products.loc[i,'image_mean']:.1f}")
    axes[i].axis("off")
plt.suptitle("카테고리별 합성 상품 썸네일 예시")
plt.tight_layout()
plt.show()
"""
)

md(
"""**해석**
- `image_mean`(평균 밝기)은 카테고리마다 뚜렷이 다르게 설계되어 있다(도서 계열이 밝고,
  전자제품/가구 계열이 어둡게 생성). 뒤의 상관분석에서 `image_mean`과 `price`의 관계를
  정량적으로 확인한다.
"""
)

# ------------------------------------------------------------------ #
md("## 3. 결측치 처리 (그룹별 대치)")

code(
"""missing_before = da.products.isna().sum()
missing_before[missing_before > 0]
"""
)

code(
"""summary = da.handle_missing_values(strategy="group_mean", group_col="category",
                                    columns=["stock_qty", "rating"])
summary
"""
)

code('da.products.isna().sum().sum()  # 처리 후 전체 결측치 개수')

md(
"""**해석**
- `stock_qty` 90건, `rating` 60건의 결측을 **카테고리별 평균**으로 대치했다. 전체
  평균이 아닌 카테고리별 평균을 쓰는 이유는 예를 들어 `가구`와 `도서`의 평균 재고 수준이
  다르기 때문에(카테고리 특성 반영), 그룹 평균이 전체 평균보다 더 정확한 추정치이기
  때문이다.
- 처리 후 두 컬럼의 결측치는 0건으로, 이후 통계/시각화 분석에 전체 1,500건을 그대로
  사용할 수 있다.
"""
)

# ------------------------------------------------------------------ #
md("## 4. 이상치 탐지 (IQR) — 처리 전후 분포 비교")

code(
"""outliers = da.detect_outliers("price", threshold=1.5)
print(f"이상치 건수: {len(outliers)} / 전체 {len(da.products)} ({len(outliers)/len(da.products)*100:.1f}%)")
outliers["category"].value_counts()
"""
)

code('capped_col = da.cap_outliers("price", threshold=1.5)\ncapped_col')

code(
"""fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(y=da.products["price"], ax=axes[0], color="#ff8c69")
axes[0].set_title("처리 전: price (원본)")
axes[0].set_ylabel("price (원)")

sns.boxplot(y=da.products[capped_col], ax=axes[1], color="#69b3ff")
axes[1].set_title("처리 후: price_capped (IQR 클리핑)")
axes[1].set_ylabel("price (원)")
plt.suptitle("이상치 처리 전후 박스플롯 비교")
plt.tight_layout()
plt.show()
"""
)

md(
"""**해석**
- `price` 기준 IQR(threshold=1.5) 이상치는 201건(전체의 13.4%)으로 비교적 높은 비율인데,
  `value_counts()`로 보면 대부분 `전자제품`/`가구`처럼 원래 고가인 카테고리에 몰려 있다.
  즉 데이터 입력 오류라기보다 **카테고리별 가격대가 크게 다른 구조**가 원인이며,
  실무에서는 카테고리별로 IQR을 따로 적용하는 것이 더 적절하다는 시사점을 얻을 수 있다.
- 박스플롯 비교에서 클리핑(`price_capped`) 후 상단 수염(whisker) 위쪽의 극단치들이
  사라지고 박스 자체의 위치는 거의 그대로 유지되어, 중심 경향은 보존하면서 극단값만
  눌렀음을 확인할 수 있다.
"""
)

# ------------------------------------------------------------------ #
md("## 5. 통계 분석")

code(
"""num_cols = ["price", "rating", "stock_qty", "image_mean", "image_std", "name_length"]
da.summary_statistics(num_cols)
"""
)

code(
"""corr = da.correlation_matrix(num_cols)
corr
"""
)

code(
"""plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("수치형 변수 상관행렬 (Heatmap)")
plt.tight_layout()
plt.show()
"""
)

md(
"""**해석**
- `price`와 `image_std`의 상관계수는 약 **+0.58**로 뚜렷한 양의 상관을 보인다 → 고가
  상품일수록 썸네일의 명암 대비(디테일/노이즈)가 크게 생성된 카테고리(전자제품/가구)에
  속해 있음을 보여준다.
- `price`와 `image_mean`은 약 **-0.62**로 음의 상관을 보인다 → 저가 카테고리(도서/식품)가
  상대적으로 밝은 톤으로 생성되어, 이미지의 단순 통계치만으로도 가격대를 어느 정도
  구분할 수 있는 신호가 됨을 시사한다(실제 서비스라면 이미지 기반 가격대 추정 모델의
  가능성을 암시).
"""
)

code(
"""plt.figure()
sns.scatterplot(data=da.products, x="image_std", y="price", hue="category", alpha=0.6)
plt.title("상품 가격(price) vs 이미지 명암대비(image_std)")
plt.xlabel("image_std (명암 대비)")
plt.ylabel("price (원)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()
"""
)

code(
"""plt.figure()
sns.histplot(da.products["price"], bins=40, kde=True, color="#4c72b0")
plt.title("상품 가격(price) 분포 (Histogram)")
plt.xlabel("price (원)")
plt.ylabel("상품 수")
plt.tight_layout()
plt.show()
"""
)

code(
"""cat_price = da.products.groupby("category")["price"].mean().sort_values(ascending=False)
plt.figure()
sns.barplot(x=cat_price.values, y=cat_price.index, color="#55a868")
plt.title("카테고리별 평균 가격 (Bar Chart)")
plt.xlabel("평균 price (원)")
plt.ylabel("category")
plt.tight_layout()
plt.show()
"""
)

md(
"""**해석**
- 가격 히스토그램은 오른쪽으로 긴 꼬리를 가진(right-skewed) 분포로, 평균(약 97,500원)이
  중앙값(약 39,000원)보다 훨씬 큰 1절의 관찰과 일치한다.
- 카테고리별 평균가는 `전자제품` > `가구` > `스포츠` 순으로, 저가 카테고리(`도서`,`식품`)와
  10배 가까운 차이를 보인다. 이는 4장에서 `price`의 전체 IQR 이상치가 특정 카테고리에
  쏠렸던 이유를 바로 설명해 준다.
"""
)

# ------------------------------------------------------------------ #
md("## 6. 월별 매출 추이 (Line Chart)")

code(
"""tx["order_month"] = pd.to_datetime(tx["order_date"]).dt.to_period("M").dt.to_timestamp()
monthly_sales = tx.groupby("order_month")["amount"].sum()

plt.figure(figsize=(10, 5))
monthly_sales.plot(marker="o", color="#c44e52")
plt.title("월별 매출 추이 (Line Chart)")
plt.xlabel("월")
plt.ylabel("매출 합계 (원)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
)

md(
"""**해석**
- 월별 매출은 최근 달로 갈수록 거래 건수가 누적되며 전반적인 상승 추세를 보인다
  (신규 가입 고객 유입 + 기존 고객의 반복 구매 누적 효과). 특정 월에 급격한 하락이
  있다면 프로모션 종료/재고 이슈 등을 의심해볼 수 있는데, 이번 데이터에서는 뚜렷한
  이상 하락 구간은 관찰되지 않는다.
"""
)

# ------------------------------------------------------------------ #
md(
"""## 7. RFM 기반 고객 세분화

`calculate_rfm()`은 고객별 Recency(최근성)/Frequency(구매빈도)/Monetary(구매금액)를
계산한 뒤, 4분위 점수(R/F/M Score, 1~4)를 기반으로 5개 세그먼트
(`VIP`, `Loyal`, `New`, `Churned`, `At Risk`)로 분류한다.
"""
)

code(
"""rfm = da.calculate_rfm(customer_col="customer_id", date_col="order_date", amount_col="amount")
print("분석 기준일:", pd.to_datetime(tx['order_date']).max().date())
rfm.head()
"""
)

code(
"""seg_summary = rfm.groupby("segment").agg(
    고객수=("customer_id", "count"),
    평균_Recency=("recency", "mean"),
    평균_Frequency=("frequency", "mean"),
    평균_Monetary=("monetary", "mean"),
).round(1).sort_values("평균_Monetary", ascending=False)
seg_summary["고객비중(%)"] = (seg_summary["고객수"] / len(rfm) * 100).round(1)
seg_summary
"""
)

code(
"""order = seg_summary.index.tolist()
plt.figure()
sns.countplot(data=rfm, x="segment", order=order, palette="viridis")
plt.title("RFM 세그먼트별 고객 수 (Bar Chart)")
plt.xlabel("segment")
plt.ylabel("고객 수")
plt.tight_layout()
plt.show()
"""
)

code(
"""pivot = rfm.pivot_table(index="R_score", columns="F_score", values="customer_id", aggfunc="count", fill_value=0)
plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
plt.title("R_score x F_score 고객 분포 (Heatmap)")
plt.xlabel("F_score (구매 빈도)")
plt.ylabel("R_score (최근성, 4=가장 최근)")
plt.tight_layout()
plt.show()
"""
)

md(
"""**해석 (RFM 세그먼트별 특징)**
- **VIP** (고객 218명, 27.3%): 평균 재구매 간격(Recency) 약 16.6일, 평균 구매 13.4회,
  평균 누적 구매액 약 331만원으로 전 세그먼트 중 압도적 1위 → 전체 매출 기여도가 가장 큼.
- **Loyal** (126명, 15.8%): Recency 57.7일, Frequency 6.7회, Monetary 108만원 수준의
  꾸준한 중간 등급 고객.
- **New** (155명, 19.4%): Recency 15.1일로 매우 최근에 유입되었지만 Frequency 1.5회로
  아직 구매 이력이 짧은 신규 고객.
- **Churned** (204명, 25.5%): Recency 174.0일, Frequency 2.6회로 오래전에 소수 구매 후
  이탈한 고객.
- **At Risk** (97명, 12.1%): Recency가 278.7일로 전 세그먼트 중 가장 길지만(가장 오래
  구매가 없음) Frequency는 5.9회로 Churned보다 오히려 높다 → **한때 자주 구매했던
  우수 고객이 조용히 이탈하고 있는 상태**로, 선제적 대응이 가장 시급한 그룹이다.
- 비즈니스 전략 제안은 `README.md`의 "비즈니스 인사이트" 절에 상세히 정리했다.
"""
)

# ------------------------------------------------------------------ #
md(
"""## 8. 보너스 — 이미지 고급 피처 (히스토그램 / 엣지 특성)

OpenCV/Pillow 없이 NumPy만으로 (1) 밝기 히스토그램의 분산도, (2) 인접 픽셀 차분(gradient)
기반 엣지 강도를 계산한다.
"""
)

code(
"""gray = da.images.mean(axis=3)  # (N, H, W) 그레이스케일 근사

# 엣지 강도: 가로/세로 방향 인접 픽셀 차분의 절댓값 평균 (NumPy 슬라이싱, for-loop 없음)
dx = np.abs(np.diff(gray, axis=2))
dy = np.abs(np.diff(gray, axis=1))
edge_strength = dx.mean(axis=(1, 2)) * 0.5 + dy.mean(axis=(1, 2)) * 0.5

da.products["edge_strength"] = edge_strength
da.products.groupby("category")["edge_strength"].mean().sort_values(ascending=False)
"""
)

md(
"""- 카테고리별 `edge_strength`(엣지 강도)가 다르게 나타나는데, 이는 2장에서 카테고리마다
  부여한 노이즈 강도(noise)가 반영된 결과다. 노이즈가 큰 썸네일일수록 인접 픽셀 간
  차분이 커져 엣지 강도도 높게 측정된다.
"""
)

# ------------------------------------------------------------------ #
md(
"""## 9. 보너스 — 코호트 분석 (가입월별 재구매율)

고객의 첫 구매월을 "가입월(코호트)"로 정의하고, 이후 N개월 차에 재구매가 발생했는지를
히트맵으로 시각화한다.
"""
)

code(
"""tx2 = tx.copy()
tx2["order_month"] = pd.to_datetime(tx2["order_date"]).dt.to_period("M")
first_month = tx2.groupby("customer_id")["order_month"].min().rename("cohort_month")
tx2 = tx2.join(first_month, on="customer_id")
tx2["period_number"] = (tx2["order_month"] - tx2["cohort_month"]).apply(lambda x: x.n)

cohort_counts = tx2.groupby(["cohort_month", "period_number"])["customer_id"].nunique().unstack(fill_value=0)
cohort_size = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_size, axis=0).round(3)
retention.index = retention.index.astype(str)
retention.iloc[:, :9]
"""
)

code(
"""plt.figure(figsize=(11, 7))
sns.heatmap(retention.iloc[:, :9], annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1)
plt.title("가입월 코호트별 재구매율 (Retention Heatmap)")
plt.xlabel("가입 후 경과 월(Period)")
plt.ylabel("가입월(Cohort)")
plt.tight_layout()
plt.show()
"""
)

md(
"""- 0개월차(가입월)는 정의상 100%이며, 이후 월차로 갈수록 재구매율이 감소하는 전형적인
  코호트 패턴을 보인다. 초기 코호트일수록 관측 기간이 길어 장기 재구매율까지 확인할 수
  있다는 장점이 있다.
"""
)

md(
"""## 10. 결론

- 멀티모달(수치+텍스트+범주+날짜+이미지) 데이터를 하나의 `DataAnalyzer` 파이프라인으로
  통합 처리했고, IQR 이상치/그룹별 결측치 처리/6종 시각화/RFM 세분화까지 모든 필수
  요구사항을 충족했다.
- 가장 핵심적인 발견은 **"At Risk" 세그먼트(12.1%, 97명)** — 한때 평균 이상으로 자주
  구매했지만 현재 278일 넘게 재구매가 없는 고객군이다. 자세한 실행 전략은
  `README.md`를 참고한다.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

with open("notebooks/analysis_report.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("notebook written:", len(cells), "cells")
