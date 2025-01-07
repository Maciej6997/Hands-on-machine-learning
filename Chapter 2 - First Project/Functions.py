from pathlib import Path
import pandas as pd
import tarfile 
import urllib.request
import numpy as np
from zlib import crc32

#data loading
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents = True, exist_ok = True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path = 'datasets')
    return pd.read_csv(Path('datasets/housing/housing.csv'))


#train_test_split easy implementations
def shuffle_and_split(data, test_ratio):
    data_size = len(data)
    shuffled_indices = np.random.permutation(data_size)
    test_set_size = int(data_size * test_ratio)
    test_indices  = shuffled_indices[ : test_set_size]
    train_indeces = shuffled_indices[test_set_size : ]
    return data.iloc[train_indeces], data.iloc[test_indices]


def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : is_id_in_test_set(id_, test_ratio))
    return data.loc[-in_test_set], data.loc[in_test_set]


