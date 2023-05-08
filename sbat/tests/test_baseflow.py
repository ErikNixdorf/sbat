import numpy as np
import pandas as pd


class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [
            [0.62534071, 0.18391693, 0.29410677, 0.4445456, 0.17079335, 0.38419759],
            [0.58963001, 0.19907613, 0.33762889, 0.38273513, 0.16239809, 0.42430935],
            [0.71683781, 0.16463393, 0.22966692, 0.13804219, 0.05608744, 0.40630647],
            [0.71527995, 0.17794513, 0.24877689, 0.12534086, 0.04792516, 0.38235861],
        ]

        result = model_config1.gauges_meta[
            [
                "bfi_monthly_mean",
                "bfi_monthly_std",
                "bfi_monthly_cv",
                "bf_monthly_mean",
                "bf_monthly_std",
                "bf_monthly_cv",
            ]
        ].values
        np.testing.assert_almost_equal(
            result,
            expected,
        )

    def test_keys(self, model_config1):
        expected_keys = ["bf_daily", "bfi_monthly", "bf_attributes", "bf_monthly"]
        present_keys = model_config1.bf_output.keys()
        assert all(key in present_keys for key in expected_keys)

    def test_baseflow_daily(self, model_config1):
        expected = pd.read_csv(
            "data/example1/bf_daily.csv", index_col=0, parse_dates=True
        )
        result = model_config1.bf_output["bf_daily"]
        pd.testing.assert_frame_equal(expected, result)

    def test_baseflow_monthly(self, model_config1):
        expected = pd.read_csv(
            "data/example1/bf_monthly.csv", index_col=0, parse_dates=True
        )
        result = model_config1.bf_output["bf_monthly"]
        pd.testing.assert_frame_equal(expected, result)

    def test_baseflow_index_monthly(self, model_config1):
        expected = pd.read_csv(
            "data/example1/bfi_monthly.csv", index_col=0, parse_dates=True
        )
        result = model_config1.bf_output["bfi_monthly"]
        pd.testing.assert_frame_equal(expected, result)

    def test_baseflow_attributes(self, model_config1):
        expected = pd.read_csv("data/example1/bf_attributes.csv", index_col=0)
        result = model_config1.bf_output["bf_attributes"]
        pd.testing.assert_frame_equal(expected, result)


class TestConfig3:
    def test_updated_gauges_meta_mean(self, model_config3):
        expected = pd.Series(
            [0.783899, 0.135499, 0.187877, 2.759578, 1.341397, 0.550355],
            index=[
                "bfi_monthly_mean",
                "bfi_monthly_std",
                "bfi_monthly_cv",
                "bf_monthly_mean",
                "bf_monthly_std",
                "bf_monthly_cv",
            ],
        )

        result = model_config3.gauges_meta[
            [
                "bfi_monthly_mean",
                "bfi_monthly_std",
                "bfi_monthly_cv",
                "bf_monthly_mean",
                "bf_monthly_std",
                "bf_monthly_cv",
            ]
        ].mean()
        pd.testing.assert_series_equal(expected, result)

    def test_keys(self, model_config3):
        expected_keys = ["bf_daily", "bfi_monthly", "bf_attributes", "bf_monthly"]
        present_keys = model_config3.bf_output.keys()
        assert all(key in present_keys for key in expected_keys)

    def test_baseflow_daily(self, model_config3):
        expected = pd.read_csv(
            "data/example3/bf_daily.csv", index_col=0, parse_dates=True
        )
        result = model_config3.bf_output["bf_daily"]
        pd.testing.assert_frame_equal(expected, result)

    def test_baseflow_monthly(self, model_config3):
        expected = pd.read_csv(
            "data/example3/bf_monthly.csv", index_col=0, parse_dates=True
        )
        result = model_config3.bf_output["bf_monthly"]
        pd.testing.assert_frame_equal(expected, result)

    def test_baseflow_index_monthly(self, model_config3):
        expected = pd.read_csv(
            "data/example3/bfi_monthly.csv", index_col=0, parse_dates=True
        )
        result = model_config3.bf_output["bfi_monthly"]
        pd.testing.assert_frame_equal(expected, result)

    def test_baseflow_attributes(self, model_config3):
        expected = pd.read_csv("data/example3/bf_attributes.csv", index_col=0)
        result = model_config3.bf_output["bf_attributes"]
        pd.testing.assert_frame_equal(expected, result)
