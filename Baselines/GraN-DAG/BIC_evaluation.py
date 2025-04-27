import gc
import os
import sys
import time
import warnings

import pandas as pd

sys.path.append(os.path.abspath("/home/jifeng/PycharmProjects/unrestricted-GANBLR/ganblr-0.1.1"))

from pgmpy.estimators import BicScore
from pgmpy.metrics import structure_score
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ganblr.models import RLiG


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
    elif name == "shuttle":
        # dataset = fetch_ucirepo(id=148)
        dataset = pd.read_csv('../Datasets/discretizedata-main/shuttle.csv')
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


def test_ganblr(name="adult"):
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

    model = RLiG()

    start_time = time.time()
    model.fit(x, y, episodes=60, gan=1, k=0, epochs=15, n=1)
    # model.fit(x, y, k=1, epochs=50)
    end_time = time.time()

    model_graphviz = model.bayesian_network.to_graphviz()
    model_graphviz.draw(f"{name}.png", prog="dot")

    lr_result = model.evaluate(x, y, model='lr')
    mlp_result = model.evaluate(x, y, model='mlp')
    rf_result = model.evaluate(x, y, model='rf')

    # 用自己的
    # 单开一个pipeline,onehot
    results = {
        "Logistic Regression": lr_result,
        "MLP": mlp_result,
        "Random Forest": rf_result,  # not suitable
        "Best Score": model.best_score
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

    del model
    gc.collect()

    return


def test_ganblr(name="adult", method="gran"):
    if method == "gran":
        MODEL_PATH = f"./results_{method}/{name}/to-dag/DAG.npy"
    elif method == "notears":
        MODEL_PATH = f"./results_{method}/{name}/DAG_NOTEARS.npy"
    elif method == "dag_gnn":
        MODEL_PATH = f"./results_{method}/{name}/DAG_DAG_GNN.npy"
    else:
        raise TypeError("Unmatched Method")

    x, y = get_uci_data(name=name)
    y = y.squeeze()

    import warnings
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('pgmpy').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    # load model from DAG
    adj_matrix = np.load(MODEL_PATH)

    model = RLiG()
    model.fit(x, y, input_model=adj_matrix, episodes=1, gan=1, k=0, epochs=0, n=0)

    # Illustrate
    model_graphviz = model.bayesian_network.to_graphviz()
    model_graphviz.draw(f"{method}_{name}.png", prog="dot")

    # TSTR
    lr_result = model.evaluate(x, y, model='lr')
    mlp_result = model.evaluate(x, y, model='mlp')
    rf_result = model.evaluate(x, y, model='rf')

    # Results
    results = {
        "Best Score": model.best_score,
        "Logistic Regression": lr_result,
        "MLP": mlp_result,
        "Random Forest": rf_result,  # not suitable
    }

    print("Dataset:", name)
    for model_name, result in results.items():
        print(f"{model_name}: {result}")

    file_path = "running_result.txt"
    with open(file_path, "a") as f:
        f.write(f"Dataset: {name}\n")
        for model_name, result in results.items():
            f.write(f"{model_name}: {result}\n")

    return


if __name__ == '__main__':
    # available_datasets = ["magic", "satellite", "loan", "connect", "credit", "localization-dm"]
    # available_datasets = ["adult", "car", "chess", "connect", "credit", "letter", "loan", "magic", "nursery",
    #                       "pokerhand"]
    available_datasets = ["satellite"]
    # methods = ["gran", "notears", "dag_gnn"]
    methods = ["notears"]

    for dataset_name in available_datasets:
        for method in methods:
            test_ganblr(name=dataset_name, method=method)
