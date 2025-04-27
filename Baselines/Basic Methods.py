#This is the file used for creating the baseline for A*, Hill Climbing Search, Tree Search, and Mmhc Estimator
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch,BicScore,PC
from pgmpy.metrics import structure_score
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
import warnings

from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')

def get_uci_data(name="adult"):
    if name == "adult":
        # dataset = fetch_ucirepo(id=2)
        dataset = pd.read_csv('../Datasets/discretizedata-main/adult-dm.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "intrusion":
        dataset = fetch_ucirepo(id=942)  # Maybe the wrong one?
        dataset = dataset.data.original.dropna(axis=0)
    elif name == "pokerhand":
        # dataset = fetch_ucirepo(id=158)
        dataset = pd.read_csv('../Datasets/pokerhand_dm.csv')
        # dataset = dataset[dataset.iloc[:, -1].isin([0, 1])]
        # print(dataset)
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "shuttle":
        # dataset = fetch_ucirepo(id=148)
        dataset = pd.read_csv('../Datasets/discretizedata-main/shuttle.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "loan":
        # dataset = fetch_ucirepo(id=148)
        dataset = pd.read_csv('../Datasets/discretizedata-main/loan-dm.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "connect":
        # dataset = fetch_ucirepo(id=151)
        dataset = pd.read_csv('../Datasets/discretizedata-main/connect-4.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "credit":
        dataset = pd.read_csv('../Datasets/discretizedata-main/creditcard-mod-dm-encode.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "chess":  # sensus;credit
        # dataset = fetch_ucirepo(id=22)
        dataset = pd.read_csv('../Datasets/discretizedata-main/chess.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "letter":  # letter recognition; small classification dataset
        dataset = fetch_ucirepo(id=59)
        dataset = dataset.data.original.dropna(axis=0)
    elif name == "magic":
        # dataset = fetch_ucirepo(id=159)
        dataset = pd.read_csv('../Datasets/discretizedata-main/magic.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "nursery":
        dataset = fetch_ucirepo(id=76)
        dataset = dataset.data.original.dropna(axis=0)
    elif name == "satellite":
        # dataset = fetch_ucirepo(id=146)
        dataset = pd.read_csv('../Datasets/discretizedata-main/satellite.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "car":
        dataset = fetch_ucirepo(id=19)
        dataset = dataset.data.original.dropna(axis=0)
    elif name == "letter-recog":
        dataset = pd.read_csv('../Datasets/letter-recog.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "localization-dm":
        dataset = pd.read_csv('../Datasets/localization-dm.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "covtype":
        dataset = pd.read_csv('../Datasets/discretizedata-main/covtype_dm_encode.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    elif name == "sign":
        dataset = pd.read_csv('../Datasets/sign.csv')
        # features = dataset.iloc[:, :-1]
        # targets = dataset.iloc[:, -1]
        # return features, targets
    else:
        raise Exception("Please Check Your Dataset Name")

    # df = dataset.data.original.dropna(axis=0)
    return dataset
    # features = dataset.data.features
    # targets = dataset.data.targets
    # return features, targets


if __name__ == '__main__':

    available_datasets = ["satellite","localization-dm","shuttle"]
    print("Testing the following datasets:", available_datasets)

    for name in available_datasets:
        # data = pd.read_csv('water_samples.csv', delimiter=',', dtype='category')
        data = get_uci_data(name=name)


        hc_est = HillClimbSearch(data)
        hc_best_model = hc_est.estimate(scoring_method=BicScore(data))

        # print("Hill Climbing Nodes:", len(hc_best_model.nodes()))
        # print("Edges:", len(hc_best_model.edges()))
        print(name, "Hill Climbing Score:", structure_score(hc_best_model, data, scoring_method="bic"))

        # PC_est = PC(data)
        # PC_best_model = PC_est.estimate(scoring_method=BicScore(data))
        # # print("PC Nodes:",sorted(PC_best_model.nodes()))
        # print("PC Nodes:", len(PC_best_model.nodes()))
        # # print("PC Edges:", PC_best_model.edges())
        # print("PC Edges:", len(PC_best_model.edges()))
        # print("PC Score:", structure_score(PC_best_model,data,scoring_method="bic"))


