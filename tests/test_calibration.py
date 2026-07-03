import unittest

from esc_rag.calibration import base_prognosis_floor, high_risk_prognosis_rescue


class PrognosisCalibrationTests(unittest.TestCase):
    def test_complex_stable_case_can_be_prolonged_without_icu(self):
        snapshot = {
            "bp": {"sys": 130},
            "spo2": 95,
            "fio2": 0.21,
            "k": 5.9,
            "creatinine": 1.7,
            "lactate": 1.2,
            "gcs": 15,
        }
        self.assertEqual(base_prognosis_floor(snapshot), 5)

    def test_possible_intubation_promotes_to_seven(self):
        snapshot = {"bp": {"sys": 120}, "spo2": 94, "fio2": 0.21, "lactate": 1.0, "gcs": 15}
        self.assertEqual(base_prognosis_floor(snapshot, intubation_code=2), 7)

    def test_very_high_lactate_promotes_to_death_risk(self):
        snapshot = {"bp": {"sys": 110}, "lactate": 6.8, "creatinine": 1.8, "gcs": 15}
        response = {"excel_codes": {"departm_rag": 1}, "estimated_duration": {"hds_score_1_to_5": 5}}
        self.assertEqual(high_risk_prognosis_rescue(response, snapshot, 6), 8)

    def test_high_grade_av_block_with_very_low_rate_promotes_to_death_risk(self):
        snapshot = {"hr": 30, "bp": {"sys": 100}, "lactate": 1.7, "gcs": 15}
        response = {"clinical_judgment": "High-grade AV block with ventricular rate 30/min."}
        self.assertEqual(high_risk_prognosis_rescue(response, snapshot, 5), 8)


if __name__ == "__main__":
    unittest.main()
