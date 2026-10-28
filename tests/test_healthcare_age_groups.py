"""Run from the repository root: python -m unittest discover -s tests."""
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "QNo 1 Final Term Health Care.ipynb"


def assign_age_groups(frame):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {"pd": pd, "np": np, "df": frame.copy()}
    with contextlib.redirect_stdout(io.StringIO()):
        exec("".join(cells[26]["source"]), namespace)
    return namespace["df"]


class HealthcareAgeGroupsTest(unittest.TestCase):
    def test_age_boundaries_and_invalid_values(self):
        ages = [-1, 0, 1, 18, 19, 35, 36, 50, 51, 65, 66, 100, 102, 115, np.nan]
        frame = assign_age_groups(pd.DataFrame({"Age": ages, "No_show": "No"}))
        actual = frame["AgeGroup"].astype(object).where(frame["AgeGroup"].notna(), None)
        self.assertEqual(actual.tolist(), [
            None, "0-18", "0-18", "0-18", "19-35", "19-35", "36-50",
            "36-50", "51-65", "51-65", "66+", "66+", "66+", "66+", None
        ])

    def test_all_valid_appointments_contribute_to_age_statistics(self):
        frame = pd.read_csv(ROOT / "healthcare-2016.csv").rename(columns={"No-show": "No_show"})
        grouped = assign_age_groups(frame)
        valid = frame["Age"].ge(0)
        self.assertTrue(grouped.loc[valid, "AgeGroup"].notna().all())
        self.assertTrue(grouped.loc[~valid, "AgeGroup"].isna().all())
        counts = pd.crosstab(grouped["AgeGroup"], grouped["No_show"])
        self.assertEqual(int(counts.to_numpy().sum()), 110526)
        expected_children = frame.loc[frame["Age"].between(0, 18), "No_show"].value_counts()
        for attendance, expected in expected_children.items():
            self.assertEqual(counts.loc["0-18", attendance], expected)


if __name__ == "__main__":
    unittest.main()
