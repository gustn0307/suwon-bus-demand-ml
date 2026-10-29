"""수원시 버스 일일 승차 수요의 검증·학습·산출물 생성 함수."""

from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "suwon_bus_boardings_202606_202608.csv"
RAW_FEATURES = ["정류소ID", "승하차일자"]
COLUMNS = ["승하차일자", "관할관청", "정류소ID", "정류소번호", "정류소명", "승차합계", "초승", "환승", "하차"]


def load_data():
    """원본을 읽기 전용으로 불러오고 품질과 예상 데이터 규모를 검증한다."""
    original_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    df = pd.read_csv(DATA, encoding="utf-8-sig", dtype={"정류소ID": "string", "정류소번호": "string"})
    if list(df.columns) != COLUMNS:
        raise ValueError(f"예상 9개 컬럼과 다릅니다: {list(df.columns)}")
    nulls = df.isna().sum().astype(int).to_dict()
    df["승하차일자"] = pd.to_datetime(df["승하차일자"], format="%Y-%m-%d", errors="raise")
    counts = df[["승차합계", "초승", "환승", "하차"]]
    if counts.isna().any().any() or not np.isfinite(counts.to_numpy()).all() or (counts < 0).any().any():
        raise ValueError("인원 값의 결측·음수·비유한 값을 확인하세요.")
    if any(nulls.values()) or df.duplicated(["정류소ID", "승하차일자"]).any():
        raise ValueError("결측치 또는 정류소·날짜 중복을 확인하세요. 임의 보정하지 않습니다.")
    if not df["관할관청"].eq("수원시").all():
        raise ValueError("수원시 이외의 관측행이 포함되어 있습니다.")
    if not df["승하차일자"].between("2026-06-01", "2026-08-31").all():
        raise ValueError("지정한 6~8월 이외의 관측일이 있습니다.")
    quality = {
        "rows": len(df), "columns": list(df.columns), "station_count": int(df["정류소ID"].nunique()),
        "date_count": int(df["승하차일자"].nunique()),
        "date_start": str(df["승하차일자"].min().date()), "date_end": str(df["승하차일자"].max().date()),
        "missing_by_column": nulls, "duplicate_station_dates": 0,
        "zero_boarding_rows": int(df["승차합계"].eq(0).sum()),
        "absent_station_dates": int(df["정류소ID"].nunique() * 92 - len(df)),
        "sha256": original_hash,
        "matches_expected": bool(len(df) == 129049 and df["정류소ID"].nunique() == 1435 and df["승하차일자"].nunique() == 92),
    }
    # 행이 없는 날은 실제 0명 관측과 다르므로 재색인·0 채우기를 하지 않는다.
    return df.sort_values(["승하차일자", "정류소ID"]).reset_index(drop=True), quality


def select_stations(df, top_n=100, min_days=55):
    """6~7월에 55일 이상 관측된 정류소 중 승차합계 상위 대상을 정한다."""
    # 8월 결과가 선정 순위나 UI의 정류소 메타데이터에 섞이지 않도록 먼저 자른다.
    past = df[df["승하차일자"].between("2026-06-01", "2026-07-31")]
    ranking = past.groupby("정류소ID").agg(observed_days=("승하차일자", "nunique"), total_boardings=("승차합계", "sum")).reset_index()
    ranking = ranking[ranking["observed_days"] >= min_days].sort_values(["total_boardings", "정류소ID"], ascending=[False, True]).head(top_n)
    metadata = past.sort_values("승하차일자").drop_duplicates("정류소ID", keep="last")[["정류소ID", "정류소번호", "정류소명"]]
    stations = ranking.merge(metadata, on="정류소ID", how="left")
    if stations.empty:
        raise ValueError("최소 관측일 조건을 만족하는 정류소가 없습니다.")
    return stations.reset_index(drop=True)


def make_features(raw):
    """예측 전에 알 수 있는 정류소 식별자와 날짜에서만 피처를 만든다."""
    # 허용 목록만 읽어 초승·환승·하차·당일 승차합계가 입력에 유입되는 것을 막는다.
    dates = pd.to_datetime(raw["승하차일자"], errors="raise")
    if raw["정류소ID"].isna().any() or dates.isna().any():
        raise ValueError("정류소ID와 날짜는 필수입니다.")
    return pd.DataFrame({
        "station": raw["정류소ID"].astype(str), "weekday": dates.dt.dayofweek.astype(str),
        "weekend": (dates.dt.dayofweek >= 5).astype(int),
        "month": dates.dt.month, "day": dates.dt.day,
    }, index=raw.index)


def build_pipeline(estimator):
    """피처 생성·범주형 인코딩·모델을 하나의 저장 가능한 파이프라인으로 묶는다."""
    # ID의 숫자 크기에는 거리 의미가 없고 요일에는 선형 순서가 없으므로 원핫 인코딩한다.
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["station", "weekday"]),
        ("numeric", StandardScaler(), ["weekend", "month", "day"]),
    ], remainder="drop")
    # 인코더와 스케일러를 Pipeline 내부에서 fit하여 검증·테스트 분포를 미리 학습하지 않는다.
    return Pipeline([("features", FunctionTransformer(make_features)), ("preprocessor", preprocessor), ("model", estimator)])


def predict_counts(model, raw):
    """앱과 평가에서 동일하게 음수 예측을 0명으로 제한하고 실수 예측을 반환한다."""
    # 인원은 음수가 될 수 없으므로 동일한 후처리를 평가와 서비스에 모두 적용한다.
    return np.maximum(model.predict(raw[RAW_FEATURES]), 0.0)


def regression_metrics(actual, predicted):
    """명 단위 MAE·RMSE와 결정계수 R²를 원래 목표 척도에서 계산한다."""
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def split_data(df, stations):
    """선정 정류소 관측행을 섞지 않고 6월·7월·8월로 분리한다."""
    selected = df[df["정류소ID"].isin(stations["정류소ID"])].copy()
    parts = {name: selected[selected["승하차일자"].dt.month == month].copy()
             for name, month in [("Train", 6), ("Validation", 7), ("Test", 8)]}
    if any(part.empty for part in parts.values()):
        raise ValueError("학습·검증·테스트 중 빈 구간이 있습니다.")
    assert parts["Train"]["승하차일자"].max() < parts["Validation"]["승하차일자"].min()
    assert parts["Validation"]["승하차일자"].max() < parts["Test"]["승하차일자"].min()
    return selected, parts


def compare_models(parts):
    """6월에만 학습한 세 후보를 7월 MAE로 비교하고 최종 후보를 고른다."""
    candidates = {"Dummy": DummyRegressor(strategy="mean"), "Ridge": Ridge(alpha=1.0),
                  # MVP의 학습 시간·파일 크기·과적합을 제한하기 위해 탐색 없이 고정한다.
                  "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=20, min_samples_leaf=3, random_state=42, n_jobs=2)}
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        start = time.perf_counter()
        pipeline = build_pipeline(estimator)
        pipeline.fit(parts["Train"][RAW_FEATURES], parts["Train"]["승차합계"])
        fitted[name] = pipeline
        for split in ["Train", "Validation"]:
            frame = parts[split]
            rows.append({"model": name, "split": split, "rows": len(frame),
                         **regression_metrics(frame["승차합계"], predict_counts(pipeline, frame)),
                         "fit_seconds": time.perf_counter() - start})
    comparison = pd.DataFrame(rows)
    # 테스트 데이터에 대한 후보별 비교는 수행하지 않아 8월로 모델을 선택하지 않는다.
    winner = comparison[comparison["split"] == "Validation"].sort_values(["MAE", "model"]).iloc[0]["model"]
    return fitted[winner], comparison, str(winner)


def fit_final_and_save(df, quality, stations, selected, parts, candidate, comparison, winner):
    """선정 모델을 6~7월에 재학습하고 8월을 한 번 평가하여 산출물을 저장한다."""
    for folder in ["models", "outputs/figures"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    trainval = selected[selected["승하차일자"] <= "2026-07-31"]
    final = clone(candidate)
    final.fit(trainval[RAW_FEATURES], trainval["승차합계"])
    final_train = regression_metrics(trainval["승차합계"], predict_counts(final, trainval))
    # 최종 모델 확정 뒤 테스트 예측을 한 번만 생성하고 표·지표·그림이 이를 공유한다.
    test = parts["Test"]
    predicted = predict_counts(final, test)
    test_metrics = regression_metrics(test["승차합계"], predicted)
    result = test[["승하차일자", "정류소ID", "승차합계"]].copy()
    result["prediction"] = predicted
    result.to_csv(ROOT / "outputs/test_predictions.csv", index=False, encoding="utf-8-sig")
    stations.to_csv(ROOT / "outputs/stations.csv", index=False, encoding="utf-8-sig")
    selected[["승하차일자", "정류소ID", "승차합계"]].to_csv(ROOT / "outputs/history.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(ROOT / "outputs/model_comparison.csv", index=False, encoding="utf-8-sig")
    model_path = ROOT / "models/demand_pipeline.joblib"
    joblib.dump(final, model_path, compress=3)
    metrics = {"selected_model": winner, "selection_metric": "Validation MAE (July)",
               "station_selection": {"period": "2026-06-01..2026-07-31", "min_days": 55, "top_n": len(stations), "ranking": "total_boardings descending; station ID ascending for ties"},
               "splits": {name: {"start": str(p["승하차일자"].min().date()), "end": str(p["승하차일자"].max().date()), "rows": len(p)} for name, p in parts.items()},
               "candidate_metrics": comparison.to_dict(orient="records"),
               "final_training": {"start": "2026-06-01", "end": "2026-07-31", "rows": len(trainval), **final_train},
               "test": test_metrics, "test_evaluation_count": 1,
               "prediction_postprocessing": "clip negative predictions to zero; no rounding for evaluation",
               "model_bytes": model_path.stat().st_size, "quality": quality,
               "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}}
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != quality["sha256"]:
        raise RuntimeError("원본 CSV 해시가 변경되었습니다.")
    (ROOT / "outputs/metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return final, metrics, result


def save_figures(selected, comparison, predictions):
    """월·요일별 관측 수요, 후보 성능, 최종 실제값과 예측값 그림을 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = ROOT / "outputs/figures"
    target.mkdir(parents=True, exist_ok=True)
    view = selected.assign(month=selected["승하차일자"].dt.month, weekday=selected["승하차일자"].dt.dayofweek)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, field, values, labels in [(axes[0], "month", [6, 7, 8], ["Jun", "Jul", "Aug"]),
                                      (axes[1], "weekday", list(range(7)), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])]:
        ax.boxplot([view.loc[view[field] == v, "승차합계"] for v in values], tick_labels=labels, showfliers=False)
        ax.set(title=f"Observed demand by {field}", ylabel="Daily boardings per station (people)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Selected top 100 stations | Jun-Aug 2026 | outliers hidden, retained in data")
    fig.savefig(target / "demand_patterns.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        comparison.pivot(index="model", columns="split", values=metric).plot.bar(ax=ax, rot=15)
        ax.set(title=f"Candidate {metric}", xlabel="", ylabel=metric)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fit: June | Validation: July | winner chosen by July MAE")
    fig.savefig(target / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].scatter(predictions["승차합계"], predictions["prediction"], alpha=.25, s=9)
    limit = max(predictions["승차합계"].max(), predictions["prediction"].max())
    axes[0].plot([0, limit], [0, limit], "r--")
    axes[0].set(xlabel="Actual daily boardings", ylabel="Predicted daily boardings", title="August: station-day observations")
    daily = predictions.groupby("승하차일자")[["승차합계", "prediction"]].sum()
    axes[1].plot(daily.index, daily["승차합계"], label="Actual")
    axes[1].plot(daily.index, daily["prediction"], label="Predicted")
    axes[1].set(title="August: total for selected observed stations", ylabel="Daily boardings (people)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    fig.savefig(target / "test_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
