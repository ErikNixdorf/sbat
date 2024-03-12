import numpy as np
import pandas as pd
from pathlib import Path

class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [[1.23998459e+00, 2.47240538e-07, 9.49764290e-01, 2.26612653e+01,
        1.10013415e+03, 6.89160790e+03, 4.58528276e-02, 1.14544924e-04],
       [1.23998459e+00, 2.47240538e-07, 9.49764290e-01, 2.26612653e+01,
        1.10013415e+03, 6.89160790e+03, 4.58528276e-02, 1.14544924e-04],
       [3.49116923e-01, 1.49034173e-07, 9.65811410e-01, 1.06095217e+01,
        6.80578094e+02, 3.04186825e+03, 1.67529288e-01, 7.30652080e-05],
       [3.49116923e-01, 1.49034173e-07, 9.65811410e-01, 1.06095217e+01,
        6.80578094e+02, 3.04186825e+03, 1.67529288e-01, 7.30652080e-05]
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
        ].dropna().values
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
