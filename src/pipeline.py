"""
이커머스 멀티모달 데이터 분석 파이프라인.

수치(price, stock_qty, rating) + 텍스트(product_name) + 범주(category) +
날짜(registered_date) + 이미지 배열(product_images.npy)을 하나의 Class로
통합 처리한다. 모든 배열 연산은 반복문(for loop) 대신 NumPy/Pandas의
벡터 연산을 사용한다.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd


class DataAnalyzer:
    """상품/거래 데이터를 로드하고 전처리, 피처 엔지니어링, 통계 분석을 수행하는 클래스."""

    def __init__(self, data_path: str):
        """
        Parameters
        ----------
        data_path : str
            products.csv, product_images.npy, transactions.csv 가 들어 있는 디렉터리 경로.
        """
        self.data_path = data_path
        self.products: Optional[pd.DataFrame] = None
        self.images: Optional[np.ndarray] = None
        self.transactions: Optional[pd.DataFrame] = None
        self.rfm_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # 1. 데이터 로드
    # ------------------------------------------------------------------ #
    def load_data(self) -> "DataAnalyzer":
        """products.csv / product_images.npy / transactions.csv 를 로드한다."""
        products_path = os.path.join(self.data_path, "products.csv")
        images_path = os.path.join(self.data_path, "product_images.npy")
        tx_path = os.path.join(self.data_path, "transactions.csv")

        self.products = pd.read_csv(products_path, parse_dates=["registered_date"])
        self.images = np.load(images_path)
        self.transactions = pd.read_csv(tx_path, parse_dates=["order_date"])

        if len(self.products) != len(self.images):
            raise ValueError("products와 image 배열의 행 수가 일치하지 않습니다.")

        return self

    # ------------------------------------------------------------------ #
    # 2. 멀티모달 피처 엔지니어링
    # ------------------------------------------------------------------ #
    def engineer_image_features(self) -> pd.DataFrame:
        """
        이미지 배열(N, H, W, C)에서 반복문 없이 축(axis) 연산만으로
        상품별 평균 밝기(image_mean)와 표준편차(image_std)를 계산한다.
        """
        if self.images is None:
            raise RuntimeError("load_data()를 먼저 호출하세요.")

        # axis=(1, 2, 3) -> 각 상품(N)에 대해 H, W, C 축을 동시에 축약 (벡터 연산, for-loop 없음)
        self.products["image_mean"] = self.images.mean(axis=(1, 2, 3))
        self.products["image_std"] = self.images.std(axis=(1, 2, 3))
        return self.products

    def engineer_text_features(self) -> pd.DataFrame:
        """상품명(텍스트)에서 단어 수와 글자 길이를 파생 변수로 추가한다."""
        name = self.products["product_name"].astype(str)
        self.products["name_word_count"] = name.str.split().str.len()
        self.products["name_length"] = name.str.len()
        return self.products

    # ------------------------------------------------------------------ #
    # 3. 결측치 처리
    # ------------------------------------------------------------------ #
    def handle_missing_values(
        self,
        strategy: str = "group_mean",
        group_col: Optional[str] = "category",
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        결측치를 처리한다.

        Parameters
        ----------
        strategy : {'group_mean', 'group_median', 'mean', 'median'}
            'group_*'  -> group_col 기준으로 그룹별 평균/중앙값 대치
            'mean'/'median' -> 전체 컬럼 평균/중앙값 대치
        group_col : strategy가 'group_*'일 때 기준이 되는 범주형 컬럼
        columns : 처리할 수치형 컬럼 목록 (None이면 결측이 있는 모든 수치 컬럼)

        Returns
        -------
        pd.DataFrame
            처리 전/후 결측치 개수를 담은 요약 테이블.
        """
        df = self.products
        if columns is None:
            columns = [c for c in df.select_dtypes(include=np.number).columns if df[c].isna().any()]

        before_counts = {c: int(df[c].isna().sum()) for c in columns}

        for col in columns:
            if strategy == "group_mean":
                fill_values = df.groupby(group_col)[col].transform("mean")
                df[col] = df[col].fillna(fill_values)
            elif strategy == "group_median":
                fill_values = df.groupby(group_col)[col].transform("median")
                df[col] = df[col].fillna(fill_values)
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            else:
                raise ValueError(f"알 수 없는 strategy: {strategy}")

            # 그룹 내 전체가 결측이라 group 대치로도 못 채운 잔여분은 전체 평균으로 보정
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        after_counts = {c: int(df[c].isna().sum()) for c in columns}

        summary = pd.DataFrame(
            {"column": columns, "missing_before": [before_counts[c] for c in columns],
             "missing_after": [after_counts[c] for c in columns]}
        )
        return summary

    # ------------------------------------------------------------------ #
    # 4. 이상치 탐지 (IQR)
    # ------------------------------------------------------------------ #
    def detect_outliers(self, column: str, threshold: float = 1.5) -> pd.DataFrame:
        """IQR 기반으로 column의 이상치 행을 반환한다."""
        q1 = self.products[column].quantile(0.25)
        q3 = self.products[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (self.products[column] < lower) | (self.products[column] > upper)
        return self.products[mask]

    def cap_outliers(self, column: str, threshold: float = 1.5, new_col: Optional[str] = None) -> str:
        """
        IQR 경계값으로 이상치를 클리핑(winsorize)한 새 컬럼을 추가하고,
        처리 전/후 분포를 비교할 수 있도록 새 컬럼명을 반환한다.
        """
        q1 = self.products[column].quantile(0.25)
        q3 = self.products[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        new_col = new_col or f"{column}_capped"
        self.products[new_col] = self.products[column].clip(lower=lower, upper=upper)
        return new_col

    # ------------------------------------------------------------------ #
    # 5. 통계 분석
    # ------------------------------------------------------------------ #
    def summary_statistics(self, columns: Sequence[str]) -> pd.DataFrame:
        """평균/중앙값/표준편차/사분위수를 포함한 요약 통계표를 반환한다."""
        stats = self.products[list(columns)].agg(
            ["mean", "median", "std", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
        )
        stats.index = ["mean", "median", "std", "q1", "q3"]
        return stats.T

    def correlation_matrix(self, columns: Sequence[str]) -> pd.DataFrame:
        """지정한 수치형 컬럼들의 상관계수 행렬을 반환한다."""
        return self.products[list(columns)].corr()

    # ------------------------------------------------------------------ #
    # 6. RFM 고객 세분화
    # ------------------------------------------------------------------ #
    def calculate_rfm(
        self,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "amount",
        analysis_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        거래 데이터로부터 고객별 Recency/Frequency/Monetary를 계산하고
        4분위 점수화(RFM Score) 후 규칙 기반으로 세그먼트를 부여한다.
        """
        tx = self.transactions.copy()
        tx[date_col] = pd.to_datetime(tx[date_col])

        if analysis_date is None:
            analysis_date = tx[date_col].max()
        analysis_date = pd.to_datetime(analysis_date)

        rfm = tx.groupby(customer_col).agg(
            recency=(date_col, lambda s: (analysis_date - s.max()).days),
            frequency=(date_col, "count"),
            monetary=(amount_col, "sum"),
        )

        # Recency는 작을수록 좋으므로(최근 구매) 역순으로 점수화
        rfm["R_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
        rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["rfm_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

        rfm["segment"] = rfm.apply(self._segment_customer, axis=1)

        self.rfm_ = rfm.reset_index().rename(columns={customer_col: "customer_id"})
        return self.rfm_

    @staticmethod
    def _segment_customer(row) -> str:
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 3 and f >= 3 and m >= 3:
            return "VIP"
        if (f >= 3 or m >= 3) and r >= 2:
            return "Loyal"
        if r >= 3 and f <= 2:
            return "New"
        if r <= 2 and f <= 2:
            return "Churned"
        # 남은 경우: 최근성(R)은 최하위 분위지만 과거 구매 빈도(F)는 높았던 고객
        # -> 한때 활발했지만 이탈 조짐을 보이는 "At Risk" 고객군
        return "At Risk"
