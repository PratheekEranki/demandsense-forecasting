FEATURE_COLS = [
    "store", "item", "dayofweek", "dayofmonth", "month", "quarter",
    "weekofyear", "year", "is_weekend", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "is_holiday", "is_day_before_holiday",
    "is_day_after_holiday", "trend",
    "lag_7", "lag_14", "lag_21", "lag_28", "lag_91", "lag_182", "lag_364",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_mean_91",
    "rolling_std_7", "rolling_std_14", "rolling_std_28", "rolling_std_91",
    "expanding_mean", "expanding_std",
]

TARGET_COL = "sales"
