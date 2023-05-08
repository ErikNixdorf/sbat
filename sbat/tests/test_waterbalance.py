import numpy as np
import pandas as pd
import geopandas as gpd


class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [
            [0.44402187438793106],
            [0.3820896833759385],
            [0.13783595100685841],
            [0.12515986411384228],
        ]

        result = model_config1.gauges_meta[["balance"]].values
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_qdiff(self, model_config1):
        expected = pd.read_csv("data/example1/qdiff.csv", index_col=0)
        expected.columns.name = "downstream_point"
        result = model_config1.q_diff
        pd.testing.assert_frame_equal(expected, result)

    def test_sections_meta(self, model_config1):
        expected = pd.read_csv("data/example1/sections_meta.csv", index_col=0)
        result = model_config1.sections_meta
        result["decade"] = result["decade"].astype(int)
        pd.testing.assert_frame_equal(expected, result)