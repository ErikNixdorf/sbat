import numpy as np
import pandas as pd
from pathlib import Path

class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [0.63759452, 0.1588524 , 0.2491433 , 0.46283348, 0.18505664,
       0.39983417, 0.72072093, 0.15494873, 0.21499131, 0.12810238,
       0.04967666, 0.38778873, 0.63759452, 0.1588524 , 0.2491433 ,
       0.46283348, 0.18505664, 0.39983417, 0.72072093, 0.15494873,
       0.21499131, 0.12810238, 0.04967666, 0.38778873]

        result = model_config1.gauges_meta[
            [
                "bfi_monthly_mean",
                "bfi_monthly_std",
                "bfi_monthly_cv",
                "bf_monthly_mean",
                "bf_monthly_std",
                "bf_monthly_cv",
            ]
        ].dropna().values.flatten()
        np.testing.assert_almost_equal(
            result,
            expected,
        )

    def test_keys(self, model_config1):
        expected_keys = ['bf_daily', 'bf_monthly', 'bfi_monthly']
        present_keys = model_config1.bf_output.keys()
        assert all(key in present_keys for key in expected_keys)

    def test_baseflow_monthly(self, model_config1):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example1/bf_monthly.csv"), index_col=0, parse_dates=True
        )
        result = model_config1.bf_output["bf_monthly"]
        pd.testing.assert_frame_equal(expected, result)

    """
    def test_baseflow_attributes(self, model_config1):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example1/bf_attributes.csv"), index_col=0)
        result = model_config1.bf_output["bf_attributes"]
        pd.testing.assert_frame_equal(expected, result)
    """

class TestConfig3:
    def test_updated_gauges_meta(self, model_config3):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example3/gauges_meta.csv"))
        expected['decade'] = expected['decade'].astype(str)        
        expected = expected.set_index(['gauge','decade'])[
            [
                "bfi_monthly_mean",
                "bfi_monthly_std",
                "bfi_monthly_cv",
                "bf_monthly_mean",
                "bf_monthly_std",
                "bf_monthly_cv",
            ]
            ]
        result = model_config3.gauges_meta[
            [
                "bfi_monthly_mean",
                "bfi_monthly_std",
                "bfi_monthly_cv",
                "bf_monthly_mean",
                "bf_monthly_std",
                "bf_monthly_cv",
            ]
            ]
        pd.testing.assert_frame_equal(expected, result)

    def test_keys(self, model_config3):
        expected_keys = ["bf_daily", "bfi_monthly", "bf_attributes", "bf_monthly"]
        present_keys = model_config3.bf_output.keys()
        assert all(key in present_keys for key in expected_keys)

    def test_baseflow_monthly(self, model_config3):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example3/bf_monthly.csv"), index_col=0, parse_dates=True
        )
        result = model_config3.bf_output["bf_monthly"]
        pd.testing.assert_frame_equal(expected, result)

    def test_baseflow_attributes(self, model_config3):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example3/bf_attributes.csv"), index_col=0)
        result = model_config3.bf_output["bf_attributes"]
        pd.testing.assert_frame_equal(expected, result)
