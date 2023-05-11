import numpy as np
import pandas as pd
from pathlib import Path

class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [
            [
                1.203596652,
                2.13E-07,
                0.971295972,
                61.57177796,
                1100.134147,
                6891.607899,
                0.019043828,
                0.000111184,

            ],
            [
                1.265027146,
                1.64E-07,
                0.920049772,
                61.57177796,
                1100.134147,
                6891.607899,
                0.025973382,
                0.000116858,
            ],
            [
                0.35846415,
                6.71E-08,
                0.971236857,
                38.988958,
                680.5780939,
                3041.86825,
                0.104019056,
                7.50E-05,
            ],
            [
                0.343581651,
                1.33E-07,
                0.974844334,
                38.988958,
                680.5780939,
                3041.86825,
                0.050409362,
                7.19E-05,
            ],
        ]

        result = model_config1.gauges_meta[
            [
                "rec_Q0",
                "rec_n",
                "pearson_r",
                "h_m",
                "dist_m",
                "network_length",
                "porosity_maillet",
                "transmissivity_maillet",
            ]
        ].values
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_master_recession_curve(self, model_config1):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example1/master_recession_curves.csv"), index_col=0)
        result = model_config1.master_recession_curves
        result["decade"]=result["decade"].astype(np.int64)
        pd.testing.assert_frame_equal(expected, result)


class TestConfig2:
    def test_updated_gauges_meta(self, model_config2):
        expected = [[0.05414452555816426,0.005409251536204091,0.9935334211464345]]

        result = model_config2.gauges_meta[
            [
                "rec_Q0",
                "rec_n",
                "pearson_r",
            ]
        ].values
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_master_recession_curve(self, model_config2):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example2/master_recession_curves.csv"), index_col=0)
        result = model_config2.master_recession_curves
        result["decade"] = result["decade"].astype(np.int64)
        pd.testing.assert_frame_equal(expected, result)
