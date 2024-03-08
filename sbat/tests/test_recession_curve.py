import numpy as np
import pandas as pd
from pathlib import Path

class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [
            [1.16003900e+00, 1.98810690e-07, 9.50075240e-01, 2.26612653e+01,
            1.10013415e+03, 6.89160790e+03, 5.33460633e-02, 1.07159864e-04],
           [3.43422384e-01, 1.62018881e-07, 9.55822127e-01, 1.06095217e+01,
            6.80578094e+02, 3.04186825e+03, 1.51589344e-01, 7.18734218e-05],
           [1.16003900e+00, 1.98810690e-07, 9.50075240e-01, 2.26612653e+01,
            1.10013415e+03, 6.89160790e+03, 5.33460633e-02, 1.07159864e-04],
           [3.43422384e-01, 1.62018881e-07, 9.55822127e-01, 1.06095217e+01,
            6.80578094e+02, 3.04186825e+03, 1.51589344e-01, 7.18734218e-05]
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
