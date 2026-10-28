"""Run from the repository root: python -m unittest discover -s tests."""
import contextlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "QNo 1 Final Term Global Temperature.ipynb"


class TemperatureUnitsTest(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(ROOT / "GlobalTemperatures.csv", parse_dates=["dt"])
        self.namespace = {"pd": pd, "data": self.raw.copy()}
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        with contextlib.redirect_stdout(io.StringIO()):
            for index in (7, 9):
                exec("".join(cells[index]["source"]), self.namespace)

    def test_measurements_used_by_plots_keep_their_original_units(self):
        plotted = self.namespace["data"]
        for column in self.raw.select_dtypes(include="number"):
            observed = self.raw[column].notna()
            pd.testing.assert_series_equal(
                plotted.loc[observed, column], self.raw.loc[observed, column]
            )
        mean_2015 = plotted.loc[
            plotted["dt"].dt.year.eq(2015), "LandAverageTemperature"
        ].mean()
        self.assertAlmostEqual(mean_2015, 9.831)

    def test_standardization_example_uses_a_separate_dataframe(self):
        normalized = self.namespace["normalized_data"].select_dtypes(include="number")
        for column in normalized:
            self.assertAlmostEqual(normalized[column].mean(), 0)
            self.assertAlmostEqual(normalized[column].std(), 1)


if __name__ == "__main__":
    unittest.main()
