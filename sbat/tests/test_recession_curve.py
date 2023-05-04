import numpy as np
import pandas as pd


class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [
            [
                1.20359665e00,
                2.07945157e-07,
                9.70259940e-01,
                6.15717780e01,
                1.10013415e03,
                6.89160790e03,
                1.94761960e-02,
                1.11183549e-04,
            ],
            [
                1.26502715e00,
                1.78419420e-07,
                9.26943897e-01,
                6.15717780e01,
                1.10013415e03,
                6.89160790e03,
                2.38577619e-02,
                1.16858258e-04,
            ],
            [
                3.58464150e-01,
                8.32761970e-08,
                9.75976503e-01,
                3.89889580e01,
                6.80578094e02,
                3.04186825e03,
                8.37692692e-02,
                7.50214496e-05,
            ],
            [
                3.43581651e-01,
                1.33496016e-07,
                9.73740485e-01,
                3.89889580e01,
                6.80578094e02,
                3.04186825e03,
                5.00865987e-02,
                7.19067542e-05,
            ],
        ]

        result = model_config1.gauges_meta[
            [
                "Q0_rec",
                "n0_rec",
                "pearson_r",
                "h_m",
                "dist_m",
                "network_length",
                "porosity_maillet",
                "transmissivity_maillet",
            ]
        ].values
        np.testing.assert_almost_equal(
            result,
            expected,
            decimal=5
        )

    def test_master_recession_curve(self, model_config1):
        expected = pd.read_csv(
            "data/example1/master_recession_curves.csv", index_col=0
        )
        result = model_config1.master_recession_curves
        pd.testing.assert_frame_equal(expected, result)

    def test_recession_limbs_ts(self, model_config1):
        expected = pd.read_csv(
            "data/example1/recession_limbs_ts.csv", index_col=0
        )
        result = model_config1.recession_limbs_ts
        pd.testing.assert_frame_equal(expected, result)