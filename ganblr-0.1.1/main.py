import gc
import os
import time
import warnings

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ganblr import get_demo_data
from ganblr.models import GANBLR
from ganblr.models import RLiG
from ganblr.models import RLiG_Parallel
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# fetch dataset
def get_uci_data(name="adult"):
    if name == "adult":
        # dataset = fetch_ucirepo(id=2)
        dataset = pd.read_csv('../Datasets/discretizedata-main/adult-dm.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "intrusion":
        dataset = fetch_ucirepo(id=942)  # Maybe the wrong one?
    elif name == "pokerhand":
        # dataset = fetch_ucirepo(id=158)
        dataset = pd.read_csv('../Datasets/pokerhand_dm.csv')
        # dataset = dataset[dataset.iloc[:, -1].isin([0, 1])]
        # print(dataset)
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "har":
        # dataset = fetch_ucirepo(id=158)
        dataset = pd.read_csv('../Datasets/har70_a.csv')
        # dataset = dataset[dataset.iloc[:, -1].isin([0, 1])]
        # print(dataset)
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "shuttle":
        # dataset = fetch_ucirepo(id=148)
        dataset = pd.read_csv('../Datasets/discretizedata-main/shuttle.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "susytr":
        # dataset = fetch_ucirepo(id=148)
        dataset = pd.read_csv('../Datasets/susytr-dm10.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "loan":
        # dataset = fetch_ucirepo(id=148)
        dataset = pd.read_csv('../Datasets/discretizedata-main/loan-dm.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "connect":
        # dataset = fetch_ucirepo(id=151)
        dataset = pd.read_csv('../Datasets/discretizedata-main/connect-4.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "credit":
        dataset = pd.read_csv('../Datasets/discretizedata-main/creditcard-mod-dm-encode.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "NSLkdd":
        dataset = pd.read_csv('../Datasets/NSLkdd-dm.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "chess":  # sensus;credit
        # dataset = fetch_ucirepo(id=22)
        dataset = pd.read_csv('../Datasets/discretizedata-main/chess.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "letter":  # letter recognition; small classification dataset
        dataset = fetch_ucirepo(id=59)
    elif name == "magic":
        # dataset = fetch_ucirepo(id=159)
        dataset = pd.read_csv('../Datasets/discretizedata-main/magic.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "nursery":
        dataset = fetch_ucirepo(id=76)
    elif name == "satellite":
        # dataset = fetch_ucirepo(id=146)
        dataset = pd.read_csv('../Datasets/discretizedata-main/satellite.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "car":
        dataset = fetch_ucirepo(id=19)
    elif name == "letter-recog":
        dataset = pd.read_csv('../Datasets/letter-recog.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "room":
        dataset = pd.read_csv('../Datasets/datatest_roomoccupancy.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "health":
        dataset = pd.read_csv('../Datasets/Maternal Health Risk Data Set.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "localization-dm":
        dataset = pd.read_csv('../Datasets/localization-dm.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "covtype":
        # dataset = pd.read_csv('../Datasets/discretizedata-main/covtype_dm_encode.csv')
        dataset = pd.read_csv('../Datasets/covtype_mod_dm11.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    elif name == "sign":
        dataset = pd.read_csv('../Datasets/sign.csv')
        features = dataset.iloc[:, :-1]
        targets = dataset.iloc[:, -1]
        return features, targets
    else:
        raise Exception("Please Check Your Dataset Name")
    df = dataset.data.original.dropna(axis=0)
    features = dataset.data.features
    targets = dataset.data.targets
    # print(f"features: {x}")
    # print(f"targets: {y}")
    # df = df.drop(df.columns[0], axis=1)
    return features, targets
    # return df


def cdt_data_preparation(name="adult"):
    x, y = get_uci_data(name=name)  # Pandas.core.frame.Dataframe
    x = x.to_numpy()
    ordinal_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
    x_int = ordinal_encoder.fit_transform(x).astype(int)
    os.makedirs(f'../Baselines/SOTA/GraN-DAG/uci_data/{name}/', exist_ok=True)
    np.save(f'../Baselines/SOTA/GraN-DAG/uci_data/{name}/data1.npy', x_int)


def test_ganblr(name="adult", epoch=40, n=1, episodes=48, beta=0.9, beta_decay=0.85):
    x, y = get_uci_data(name=name)
    y = y.squeeze()
    # x: Dataset to fit the model.
    # y: Label of the dataset.

    import warnings
    import logging
    # from joblib import Parallel, delayed
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('pgmpy').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    # model = RLiG()
    model = RLiG_Parallel(beta=beta, beta_decay=beta_decay)
    print(x, y)

    start_time = time.time()
    model.fit(x, y, episodes=episodes, gan=1, k=0, epochs=epoch, n=n)
    # model.fit(x, y, k=1, epochs=50)
    end_time = time.time()

    model_graphviz = model.bayesian_network.to_graphviz()
    model_graphviz.draw(f"{name}.png", prog="dot")

    lr_result = model.evaluate(x, y, model='lr')
    mlp_result = model.evaluate(x, y, model='mlp')
    rf_result = model.evaluate(x, y, model='rf')

    results = {
        "LR": lr_result,
        "MLP": mlp_result,
        "RF": rf_result,  # not suitable
        "Score": model.best_score
    }
    print("Dataset:", name)
    print("Training time:", (end_time - start_time), "seconds")
    for model_name, result in results.items():
        print(f"{model_name}: {result}")

    file_path = "running_result.txt"
    with open(file_path, "a") as f:
        f.write(f"Dataset: {name}\n")
        for model_name, result in results.items():
            f.write(f"{model_name}: {result}\n")

    del model, model_graphviz
    gc.collect()

    return


if __name__ == '__main__':
    available_datasets = ["adult"]
    print("Testing the following datasets:", available_datasets)
    for dataset_name in available_datasets:
        print("Start test: ", dataset_name)
        test_ganblr(name=dataset_name, epoch=10, n=1, episodes=64, beta=0.9, beta_decay=1.0)
        # gc.collect()
        cdt_data_preparation(name=dataset_name)
