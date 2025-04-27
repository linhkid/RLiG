import gc
import time
import warnings

from ganblr import get_demo_data
from ganblr.models import GANBLR
from ganblr.models import RLiG
from ucimlrepo import fetch_ucirepo
from pgmpy.estimators import HillClimbSearch, BicScore

# # this is a discrete version of adult since GANBLR requires discrete data.
# df = get_demo_data('adult')


# fetch dataset
def get_uci_data(name="adult"):
    if name == "adult":
        dataset = fetch_ucirepo(id=2)
    elif name == "intrusion":
        dataset = fetch_ucirepo(id=942) #Maybe the wrong one?
    elif name == "pokerhand":
        dataset = fetch_ucirepo(id=158)
    elif name == "shuttle":
        dataset = fetch_ucirepo(id=148)
    elif name == "connect":
        dataset = fetch_ucirepo(id=151)
    elif name == "chess":  #sensus;credit
        dataset = fetch_ucirepo(id=22)
    elif name == "letter": #letter recognition; small classification dataset
        dataset = fetch_ucirepo(id=59)
    elif name == "magic":
        dataset = fetch_ucirepo(id=159)
    elif name == "nursery":
        dataset = fetch_ucirepo(id=76)
    elif name == "satellite":
        dataset = fetch_ucirepo(id=146)
    elif name == "car":
        dataset = fetch_ucirepo(id=19)
    else:
        raise Exception("Please Check Your Dataset Name")
    df = dataset.data.original.dropna(axis=0)
    # df = df.drop(df.columns[0], axis=1)
    return df

def test_ganblr(name="adult"):
    df = get_uci_data(name=name)

    # x, y = df.values[:,:-1], df.values[:,-1]
    x,y = df.iloc[:,:-1], df.iloc[:, -1]
    # x: Dataset to fit the model.
    # y: Label of the dataset.



    est = HillClimbSearch(df)
    model = est.estimate(scoring_method=BicScore(df))


    print("Dataset:",name)
    print(model.edges())
    del model, df
    gc.collect()
    return

if __name__ == '__main__':
    available_datasets = ["connect","chess","magic","nursery","satellite","car"]
    #"pokerhand", "letter", "intrusion", "shuttle",
    print("Testing the following datasets:",available_datasets)
    for dataset_name in available_datasets:
    #     dataset_name="adult"
        print("Start test: ", dataset_name)
        # try:
        test_ganblr(name=dataset_name)
        # except e:
        #     continue
