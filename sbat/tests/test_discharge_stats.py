import numpy as np
import pandas as pd


class TestConfig1:
    def test_daily_stats(self, model_config1):
        expected = [0.77467953, 1.06789572, 1.37849998, 0.77467953, 1.06789572,
                    1.37849998, 0.21449713, 0.29782467, 1.38847861, 0.21449713,
                    0.29782467, 1.38847861,
                    ]
        result = model_config1.gauges_meta[
            ['q_daily_mean', 'q_daily_std', 'q_daily_cv']
        ].sort_index().dropna().values.flatten()

        np.testing.assert_almost_equal(
            result,
            expected,
        )

    def test_monthly_stats(self, model_config1):
        expected = [0.77583502, 0.49909641, 0.64330225, 0.77583502, 0.49909641,
       0.64330225, 0.21478395, 0.13230531, 0.61599255, 0.21478395,
       0.13230531, 0.61599255]

        result = model_config1.gauges_meta[
            ["q_monthly_mean", "q_monthly_std", "q_monthly_cv"]
        ].sort_index().dropna().values.flatten()

        np.testing.assert_almost_equal(
            result,
            expected,
        )


class TestConfig3:
    def test_daily_stats(self, model_config3):
        expected = pd.Series(
            [0.49458833, 0.6828602, 1.3834893],
            index=["q_daily_mean", "q_daily_std", "q_daily_cv"],
        )
        result = model_config3.gauges_meta[
            ["q_daily_mean", "q_daily_std", "q_daily_cv"]
        ].mean()

        pd.testing.assert_series_equal(expected, result)

    def test_monthly_stats(self, model_config3):
        expected = pd.Series(
            [2.80302815, 1.4869131 , 0.59852174],
            index=["q_monthly_mean", "q_monthly_std", "q_monthly_cv"],
        )

        result = model_config3.gauges_meta[
            ["q_monthly_mean", "q_monthly_std", "q_monthly_cv"]
        ].mean()

        pd.testing.assert_series_equal(expected, result)
