"""
합성 이커머스 데이터셋 생성 스크립트.

실제 Kaggle/AI Hub 데이터 대신, 동일한 구조(수치/텍스트/범주/날짜/이미지 배열이
섞인 멀티모달 상품 데이터 + 고객 거래 데이터)를 갖는 합성 데이터를 생성한다.
고객 세그먼트(VIP/Loyal/New/Churned)를 의도적으로 재현 가능하게 설계해
RFM 분석 결과가 뚜렷한 패턴을 보이도록 한다.

실행: python src/generate_data.py
생성물:
  data/products.csv          (상품 마스터, 1,500건 x 8컬럼)
  data/product_images.npy    (상품 썸네일 배열, shape=(1500, 16, 16, 3), uint8)
  data/transactions.csv      (고객 거래 내역, 약 6,000건)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

ANALYSIS_DATE = datetime(2026, 11, 22)

CATEGORIES = ["전자제품", "의류", "식품", "뷰티", "가구", "도서", "스포츠", "완구"]

# 카테고리별 (가격 평균, 가격 표준편차, 평점 평균, 이미지 밝기 평균, 이미지 노이즈 강도)
CATEGORY_PROFILE = {
    "전자제품": (320000, 150000, 4.1, 120, 35),
    "의류":     (45000,   20000, 3.9, 160, 20),
    "식품":     (18000,    9000, 4.3, 180, 15),
    "뷰티":     (32000,   15000, 4.0, 200, 18),
    "가구":     (210000,  90000, 3.8, 110, 25),
    "도서":     (16000,    6000, 4.5, 230, 10),
    "스포츠":   (65000,   30000, 4.0, 140, 22),
    "완구":     (28000,   14000, 4.2, 190, 28),
}

ADJECTIVES = ["프리미엄", "베스트", "신상", "인기", "한정판", "가성비", "클래식", "미니멀", "프로", "슈퍼"]
NOUNS = {
    "전자제품": ["무선이어폰", "블루투스스피커", "노트북거치대", "보조배터리", "스마트워치", "공기청정기"],
    "의류":     ["면티셔츠", "후드집업", "데님팬츠", "경량패딩", "니트원피스", "트레이닝세트"],
    "식품":     ["견과류세트", "원두커피", "수제쿠키", "과일주스", "건강즙", "그래놀라"],
    "뷰티":     ["수분크림", "선크림", "립밤세트", "클렌징폼", "헤어에센스", "마스크팩"],
    "가구":     ["원목테이블", "수납선반", "접이식의자", "모던서랍장", "거실소파", "스탠드조명"],
    "도서":     ["자기계발서", "에세이집", "소설", "경제경영서", "그림책", "요리책"],
    "스포츠":   ["요가매트", "런닝화", "폼롤러", "자전거헬멧", "아령세트", "등산배낭"],
    "완구":     ["조립블록", "퍼즐세트", "캐릭터인형", "보드게임", "드론장난감", "교육완구"],
}


def make_products(n_products: int = 1500) -> pd.DataFrame:
    rows = []
    start = datetime(2024, 1, 1)
    for pid in range(1, n_products + 1):
        category = RNG.choice(CATEGORIES)
        price_mean, price_std, rating_mean, *_ = CATEGORY_PROFILE[category]

        name = f"{RNG.choice(ADJECTIVES)} {RNG.choice(NOUNS[category])}"
        price = max(1000, RNG.normal(price_mean, price_std))
        # 가격 이상치 5% 의도적으로 주입 (프로모션/프리미엄 특별가)
        if RNG.random() < 0.03:
            price *= RNG.choice([4.5, 5.0, 0.1])

        reg_date = start + timedelta(days=int(RNG.integers(0, 650)))
        rating = np.clip(RNG.normal(rating_mean, 0.4), 1.0, 5.0)
        stock_qty = RNG.integers(0, 500)

        rows.append(
            {
                "product_id": pid,
                "product_name": name,
                "category": category,
                "price": round(price, -2),
                "registered_date": reg_date.strftime("%Y-%m-%d"),
                "stock_qty": stock_qty,
                "rating": round(rating, 2),
            }
        )

    df = pd.DataFrame(rows)

    # 그룹(카테고리)별 결측치 대치 실습을 위해 stock_qty 일부를 의도적으로 결측 처리
    missing_idx = RNG.choice(df.index, size=int(len(df) * 0.06), replace=False)
    df.loc[missing_idx, "stock_qty"] = np.nan

    # rating 일부도 결측 처리 (다른 컬럼 기준 그룹 대치 실습용)
    missing_idx2 = RNG.choice(df.index, size=int(len(df) * 0.04), replace=False)
    df.loc[missing_idx2, "rating"] = np.nan

    return df


def make_product_images(df: pd.DataFrame, size: int = 16) -> np.ndarray:
    """카테고리별로 밝기/노이즈 특성이 다른 썸네일 이미지를 NumPy로 직접 생성한다."""
    n = len(df)
    images = np.zeros((n, size, size, 3), dtype=np.uint8)
    for i, category in enumerate(df["category"].values):
        _, _, _, brightness, noise = CATEGORY_PROFILE[category]
        base = RNG.normal(brightness, noise, size=(size, size, 3))
        img = np.clip(base, 0, 255).astype(np.uint8)
        images[i] = img
    return images


def make_transactions(df_products: pd.DataFrame, n_customers: int = 800) -> pd.DataFrame:
    """
    고객을 4가지 행동 아키타입(VIP/Loyal/New/Churned)으로 설계해 거래를 생성한다.
    실제 데이터에서는 이런 라벨이 주어지지 않으며, RFM 분석으로 '발견'하는 것이 목표다.
    """
    window_start = datetime(2025, 5, 1)
    window_end = ANALYSIS_DATE

    archetypes = RNG.choice(
        ["VIP", "Loyal", "New", "Churned", "Occasional"],
        size=n_customers,
        p=[0.08, 0.27, 0.20, 0.25, 0.20],
    )

    rows = []
    order_id = 1
    product_ids = df_products["product_id"].values
    product_prices = df_products.set_index("product_id")["price"]

    for cust_idx in range(1, n_customers + 1):
        archetype = archetypes[cust_idx - 1]

        if archetype == "VIP":
            n_orders = RNG.integers(15, 30)
            last_gap_days = RNG.integers(0, 15)
        elif archetype == "Loyal":
            n_orders = RNG.integers(6, 14)
            last_gap_days = RNG.integers(5, 45)
        elif archetype == "New":
            n_orders = RNG.integers(1, 3)
            last_gap_days = RNG.integers(0, 30)
        elif archetype == "Churned":
            n_orders = RNG.integers(2, 8)
            last_gap_days = RNG.integers(120, 400)
        else:  # Occasional
            n_orders = RNG.integers(1, 5)
            last_gap_days = RNG.integers(30, 200)

        last_order_date = window_end - timedelta(days=int(last_gap_days))
        if archetype == "New":
            span_days = RNG.integers(1, 20)
        else:
            span_days = RNG.integers(60, 500)
        first_order_date = max(window_start, last_order_date - timedelta(days=int(span_days)))

        if n_orders == 1:
            order_dates = [last_order_date]
        else:
            offsets = np.sort(RNG.integers(0, max((last_order_date - first_order_date).days, 1), size=n_orders - 1))
            order_dates = [first_order_date + timedelta(days=int(o)) for o in offsets]
            order_dates.append(last_order_date)

        for od in order_dates:
            pid = int(RNG.choice(product_ids))
            unit_price = float(product_prices.loc[pid])
            if archetype == "VIP":
                qty = RNG.integers(2, 6)
            else:
                qty = RNG.integers(1, 3)
            rows.append(
                {
                    "order_id": order_id,
                    "customer_id": f"C{cust_idx:04d}",
                    "product_id": pid,
                    "order_date": od.strftime("%Y-%m-%d"),
                    "quantity": int(qty),
                    "unit_price": unit_price,
                    "amount": round(unit_price * qty, 0),
                }
            )
            order_id += 1

    tx = pd.DataFrame(rows)
    tx = tx.sort_values("order_date").reset_index(drop=True)
    return tx


def main():
    products = make_products()
    images = make_product_images(products)
    transactions = make_transactions(products)

    products.to_csv("data/products.csv", index=False, encoding="utf-8-sig")
    np.save("data/product_images.npy", images)
    transactions.to_csv("data/transactions.csv", index=False, encoding="utf-8-sig")

    print("products:", products.shape)
    print("images:", images.shape)
    print("transactions:", transactions.shape)
    print("unique customers:", transactions["customer_id"].nunique())


if __name__ == "__main__":
    main()
