"""선정 정류소의 일일 승차 수요를 조회하는 로컬 Streamlit 앱."""

from datetime import date
import json

import joblib
import pandas as pd
import streamlit as st

from train import ROOT, predict_counts


@st.cache_resource
def load_model():
    """학습된 전처리·예측 파이프라인을 한 번 읽어 캐시한다."""
    return joblib.load(ROOT / "models/demand_pipeline.joblib")


@st.cache_data
def load_artifacts():
    """정류소 표시 정보·과거 관측·실측 평가 지표를 읽는다."""
    stations = pd.read_csv(ROOT / "outputs/stations.csv", encoding="utf-8-sig", dtype={"정류소ID": str, "정류소번호": str})
    history = pd.read_csv(ROOT / "outputs/history.csv", encoding="utf-8-sig", dtype={"정류소ID": str}, parse_dates=["승하차일자"])
    metrics = json.loads((ROOT / "outputs/metrics.json").read_text(encoding="utf-8"))
    return stations, history, metrics


def main():
    """정류소·날짜 선택, 일일 예측, 과거 이용 패턴과 테스트 성능을 표시한다."""
    st.set_page_config(page_title="수원 버스 일일 승차 수요", page_icon="🚌", layout="wide")
    st.title("수원시 버스 정류장별 일일 승차 수요")
    st.caption("2026년 6~7월 이용량 기준 상위 정류소 MVP · 하루 승차 인원 예측")
    try:
        model = load_model()
        stations, history, metrics = load_artifacts()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        st.error(f"산출물을 읽지 못했습니다. requirements 설치 후 분석 노트북을 전체 실행하세요. 상세: {exc}")
        st.stop()

    labels = {row["정류소ID"]: f'{row["정류소명"]} | 번호 {row["정류소번호"]} | ID {row["정류소ID"]}' for _, row in stations.iterrows()}
    station = st.selectbox("정류소 선택", options=list(labels), format_func=labels.get)
    selected_date = st.date_input("예측 날짜", value=date(2026, 9, 1), min_value=date(2026, 6, 1), max_value=date(2027, 8, 31))
    training_end = date.fromisoformat(metrics["final_training"]["end"])
    if selected_date > training_end:
        st.warning(f"최종 학습일은 {training_end}입니다. 이 예측은 2026년 6~8월 과거 3개월 자료의 패턴을 참고하는 제한적 추정입니다. 모델 학습에는 6~7월만 사용했고 8월은 테스트로 보존했습니다. 날씨·공휴일·노선 변경은 반영하지 않습니다.")
    else:
        st.info("학습 기간 날짜의 예측입니다. 이미 학습에 사용된 날짜이므로 미래 예측 성능으로 해석하지 마세요.")
    if st.button("일일 승차 인원 예측", type="primary"):
        # 학습과 같은 두 컬럼만 전달하여 결과성 변수 입력과 전처리 불일치를 방지한다.
        request = pd.DataFrame({"정류소ID": [station], "승하차일자": [pd.Timestamp(selected_date)]})
        estimate = float(predict_counts(model, request)[0])
        st.metric("예상 일일 승차 인원", f"{estimate:,.0f} 명")
        st.caption("반올림한 추정치입니다. 실제 차량 혼잡도나 시간대별 탑승 인원이 아닙니다.")

    st.subheader("선택 정류소 과거 이용 패턴")
    observed = history[history["정류소ID"] == station].sort_values("승하차일자")
    st.line_chart(observed.set_index("승하차일자")[["승차합계"]], y_label="일일 승차 인원(명)")
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    pattern = observed.groupby(observed["승하차일자"].dt.dayofweek)["승차합계"].mean().reindex(range(7))
    pattern.index = weekdays
    st.bar_chart(pattern.rename("평균 승차 인원"), y_label="관측일 평균(명)")
    st.caption("차트는 6~8월 실제 관측을 설명합니다. 누락된 날짜를 0명으로 보정하지 않으며, 8월 관측은 모델 입력에 사용하지 않습니다.")

    st.subheader("최종 테스트 성능 · 2026년 8월")
    for column, metric, label in zip(st.columns(3), ["MAE", "RMSE", "R2"], ["MAE (명)", "RMSE (명)", "R²"]):
        column.metric(label, f'{metrics["test"][metric]:,.3f}')
    st.caption(f'선택 모델: {metrics["selected_model"]} · 대상 {len(stations)}개 정류소 · 7월 MAE로 선정 후 6~7월 재학습 · 8월 {metrics["splits"]["Test"]["rows"]:,}행 평가')
    st.info("2026년 여름의 제한된 관측 자료입니다. 수원시 전체 정류소 또는 장기·다른 계절 성능을 보장하지 않습니다.")


if __name__ == "__main__":
    main()
