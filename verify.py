"""원본 보존·누수 방지·저장 모델·Streamlit 동작을 실제 검증한다."""

from datetime import date
import hashlib
import importlib
import json
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

import joblib
import nbformat
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from train import DATA, ROOT, RAW_FEATURES, load_data, make_features, predict_counts, select_stations


def csv_sha256_lf(content):
    """파일을 쓰지 않고 CRLF만 LF로 정규화해 Git 원본과 비교할 해시를 구한다."""
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


class MVPTests(unittest.TestCase):
    """재학습 없이 저장된 MVP의 핵심 계약과 실제 앱 상호작용을 확인한다."""

    @classmethod
    def setUpClass(cls):
        """실제 데이터·학습 모델·측정 결과를 한 번 읽어 준비한다."""
        cls.df, cls.quality = load_data()
        cls.metrics = json.loads((ROOT / "outputs/metrics.json").read_text(encoding="utf-8"))
        cls.model = joblib.load(ROOT / "models/demand_pipeline.joblib")
        cls.stations = pd.read_csv(ROOT / "outputs/stations.csv", encoding="utf-8-sig", dtype={"정류소ID": str, "정류소번호": str})

    def test_original_and_observations(self):
        """원본 해시·0명 관측·원래의 행 수가 보존되었는지 확인한다."""
        original = DATA.read_bytes()
        expected = "67eab89c329abbfe75421c9b25e2aec0dadd9dfc699b2096dd2bf0da4ae4811b"
        self.assertEqual(self.quality["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(csv_sha256_lf(original), expected)
        # 동일한 CSV의 LF·CRLF 모두 기존 기대 해시를 만족해야 한다.
        lf = original.replace(b"\r\n", b"\n")
        for ending, content in [("LF", lf), ("CRLF", lf.replace(b"\n", b"\r\n"))]:
            with self.subTest(line_ending=ending):
                self.assertEqual(csv_sha256_lf(content), expected)
        self.assertTrue(self.quality["matches_expected"])
        self.assertEqual(self.quality["zero_boarding_rows"], 5481)
        self.assertTrue(self.stations["정류소번호"].str.startswith("0").any())
        history = pd.read_csv(ROOT / "outputs/history.csv", encoding="utf-8-sig")
        actual = self.df[self.df["정류소ID"].isin(self.stations["정류소ID"])]
        self.assertEqual(len(history), len(actual))

    def test_august_cannot_change_selection(self):
        """8월 값을 크게 변경해도 정류소 선정 결과가 동일한지 확인한다."""
        changed = self.df.copy()
        changed.loc[changed["승하차일자"].dt.month == 8, "승차합계"] = 100_000_000
        pd.testing.assert_frame_equal(select_stations(self.df), select_stations(changed))

    def test_outcome_variables_are_ignored(self):
        """결과성 컬럼을 변경해도 피처와 예측이 변하지 않는지 확인한다."""
        sample = self.df[self.df["정류소ID"].isin(self.stations["정류소ID"])].head(12).copy()
        changed = sample.copy()
        changed[["승차합계", "초승", "환승", "하차"]] = 100_000_000
        pd.testing.assert_frame_equal(make_features(sample), make_features(changed))
        np.testing.assert_allclose(self.model.predict(sample), self.model.predict(changed))
        self.assertEqual(list(self.model.feature_names_in_), RAW_FEATURES)

    def test_saved_pipeline_future_and_unknown(self):
        """전체 대상의 미래 날짜 및 인코더의 미지 범주 처리가 정상인지 확인한다."""
        request = pd.DataFrame({"정류소ID": self.stations["정류소ID"], "승하차일자": pd.Timestamp("2026-09-01")})
        prediction = predict_counts(self.model, request)
        self.assertEqual(len(prediction), 100)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())
        # 인코더의 기술적 견고성만 확인한다. 앱은 학습 대상 100개만 선택하도록 제한한다.
        unknown = pd.DataFrame({"정류소ID": ["unknown"], "승하차일자": [pd.Timestamp("2026-09-01")]})
        self.assertTrue(np.isfinite(predict_counts(self.model, unknown)).all())

    def test_notebook_and_time_splits(self):
        """노트북 전체 실행과 시간 구간·최종 선택의 일관성을 확인한다."""
        notebook = nbformat.read(ROOT / "suwon_bus_demand_ml.ipynb", as_version=4)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        self.assertTrue(all(c.execution_count is not None for c in cells))
        self.assertFalse(any(o.output_type == "error" for c in cells for o in c.outputs))
        self.assertEqual([self.metrics["splits"][key]["rows"] for key in ["Train", "Validation", "Test"]], [3000, 3100, 3100])
        candidates = [r for r in self.metrics["candidate_metrics"] if r["split"] == "Validation"]
        self.assertEqual(min(candidates, key=lambda r: r["MAE"])["model"], self.metrics["selected_model"])
        self.assertEqual(self.metrics["test_evaluation_count"], 1)
        self.assertEqual(len(pd.read_csv(ROOT / "outputs/test_predictions.csv")), 3100)

    def test_app_import_and_interactions(self):
        """앱 import·첫 화면·날짜 변경·정류소 변경·예측 버튼을 실행한다."""
        importlib.import_module("app")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox[0].options), 100)
        self.assertEqual(len(app.metric), 3)
        self.assertTrue(app.warning)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)
        self.assertIn("명", app.metric[0].value)
        app.selectbox[0].select(self.stations["정류소ID"].iloc[1]).run()
        app.date_input[0].set_value(date(2026, 6, 15)).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.warning)
        self.assertEqual(len(app.metric), 4)


def check_server():
    """실제 앱 서버의 로컬 HTTP 응답을 확인하고 해당 프로세스를 종료한다."""
    (ROOT / "tmp").mkdir(exist_ok=True)
    with (ROOT / "tmp/server.log").open("w", encoding="utf-8") as log:
        # 검증을 위해서만 서버를 띄우고 finally에서 직접 시작한 프로세스를 종료한다.
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.address=127.0.0.1", "--server.port=18517", "--browser.gatherUsageStats=false"], cwd=ROOT, stdout=log, stderr=log)
        status = None
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    raise RuntimeError("앱 서버가 조기 종료되었습니다. tmp/server.log 확인")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:18517/_stcore/health", timeout=1) as response:
                        if response.status == 200:
                            status = response.read().decode()
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            if status is None:
                raise RuntimeError("서버 health 응답을 받지 못했습니다.")
            with urllib.request.urlopen("http://127.0.0.1:18517", timeout=5) as response:
                assert response.status == 200
                assert "streamlit" in response.read().decode().lower()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    return {"health": status, "page_http_status": 200, "server_stopped": proc.poll() is not None}


def main():
    """검증을 실행하고 통과·실패 및 서버 종료 결과를 JSON으로 기록한다."""
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MVPTests))
    original = DATA.read_bytes()
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "success": result.wasSuccessful(),
               "csv_sha256_after": hashlib.sha256(original).hexdigest(),
               "csv_sha256_lf_normalized": csv_sha256_lf(original)}
    if result.wasSuccessful():
        summary["server"] = check_server()
    (ROOT / "outputs/validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
