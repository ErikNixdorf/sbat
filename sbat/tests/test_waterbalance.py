import numpy as np
import pandas as pd
from pathlib import Path

class TestConfig1:
    def test_updated_gauges_meta(self, model_config1):
        expected = [0.46237469, 0.127907  , 0.46237469, 0.127907  ]
        result = model_config1.gauges_meta[['balance[m³/s]']].dropna().values
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_sections_meta(self, model_config1):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example1/sections_meta.csv"), index_col=0)
        result = model_config1.sections_meta
        result["decade"] = result["decade"].astype(np.int64)
        pd.testing.assert_frame_equal(expected, result)


class TestConfig3:
    def test_updated_gauges_meta(self, model_config3):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example3/gauges_meta.csv"))
        expected['decade'] = expected['decade'].astype(str)
        expected = expected.set_index(['gauge','decade'])['balance[m³/s]'].sort_index()
        result = model_config3.gauges_meta[['balance[m³/s]']].sort_index()['balance[m³/s]']
        pd.testing.assert_series_equal(expected, result)

    def test_updated_gauges_meta_nans(self, model_config3):
        expected = pd.Series(
            index=['doberburg',
                     'doberburg_muehle_up',
                     'friedland_panzerbruecke',
                     'goeritz_nr_195',
                     'goyatz_2',
                     'hammerstadt_1',
                     'hammerstadt_1',
                     'heinersbrueck',
                     'lieberose_wehr_op',
                     'merzdorf_2',
                     'moellen_1',
                     'neusalza_spremberg',
                     'neusalza_spremberg',
                     'niedergurig',
                     'pieskow',
                     'pretschen',
                     'radensdorf_1',
                     'radensdorf_2',
                     'reichwalde_3',
                     'reichwalde_3',
                     'schoenfeld',
                     'schoenfeld',
                     'schoeps'],
            name="balance",
        )
        expected.index.name = "gauge"
        single_indexed = model_config3.gauges_meta[['balance[m³/s]']].sort_index().reset_index(
                                level=1,
                                drop=True,
                                )
        result = single_indexed['balance[m³/s]'][single_indexed['balance[m³/s]'].isnull()]
        pd.testing.assert_series_equal(expected, result)

    def test_sections_meta(self, model_config3):
        expected = pd.read_csv(Path(Path(__file__).parents[0],"data/example3/sections_meta.csv"), index_col=[0])
        expected = expected.reset_index(drop=True).set_index(['date', 'sample_id', 'downstream_point', 'decade'])
        result = model_config3.sections_meta.set_index(['date', 'sample_id', 'downstream_point', 'decade'])
        pd.testing.assert_frame_equal(expected.sort_index(), result.sort_index())
