from src.dataset_enum import DatasetEnum


def test_load_adult():
    data_source = DatasetEnum.adult.get_data_source()
    assert data_source
