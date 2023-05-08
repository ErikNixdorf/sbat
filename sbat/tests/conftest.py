from pathlib import Path
import pytest
from sbat import sbat

# @pytest.fixture(scope="module")
# def config():
#    return sbat.Model.read_config(Path(Path(__file__).parents[2], "data/ex1_sbat.yml"))


@pytest.fixture(scope="module")
def model_config1():
    config = sbat.Model.read_config(
        Path(Path(__file__).parents[2], "data/examples/ex1_sbat.yml")
    )
    model = sbat.Model(conf=config, output=False)
    model.get_discharge_stats()
    model.get_baseflow()
    model.get_recession_curve()
    model.get_water_balance()
    return model


@pytest.fixture(scope="module")
def model_config2():
    config = sbat.Model.read_config(
        Path(Path(__file__).parents[2], "data/examples/ex2_sbat.yml")
    )
    model = sbat.Model(conf=config, output=False)
    model.get_recession_curve()
    return model
