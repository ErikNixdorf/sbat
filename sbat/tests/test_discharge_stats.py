import numpy as np


class TestConfig1:
    def test_daily_stats(self, model_config1):
        expected = [
            [0.83560016, 1.22026351, 1.46034379],
            [0.77467953, 1.06789572, 1.37849998],
            [0.21449713, 0.29782467, 1.38847861],
            [0.19540341, 0.23059803, 1.18011267],
        ]
        result = model_config1.gauges_meta[
            ["q_daily_mean", "q_daily_std", "q_daily_cv"]
        ].values

        np.testing.assert_almost_equal(
            result,
            expected,
        )

    def test_monthly_stats(self, model_config1):
        if model_config1.config["discharge"]["compute_monthly"]:
            expected = [
                [0.83616979, 0.55223783, 0.66043744],
                [0.77583502, 0.49909641, 0.64330225],
                [0.21478395, 0.13230531, 0.61599255],
                [0.19565859, 0.1093387, 0.55882393],
            ]

            result = model_config1.gauges_meta[
                ["q_monthly_mean", "q_monthly_std", "q_monthly_cv"]
            ].values

            np.testing.assert_almost_equal(
                result,
                expected,
            )


