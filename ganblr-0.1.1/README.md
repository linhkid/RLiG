# GANBLR Toolbox

GANBLR Toolbox contains GANBLR models proposed by `Tulip Lab` for tabular data generation, which can sample fully artificial data from real data.

Currently, this package contains following GANBLR models:

- GANBLR
- GANBLR++

For a quick start, you can check out this usage example in Google Colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w7A26JRkrXPeeA9q1Kbi_CRjbptkr8Ls?usp=sharing]

# Install

We recommend you to install ganblr through pip:

```bash
pip install ganblr
```

Alternatively, you can also clone the repository and install it from sources.

```bash
git clone git@github.com:tulip-lab/ganblr.git
cd ganblr
python setup.py install
```

# Usage Example

In this example we load the [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)* which is a built-in demo dataset. We use `GANBLR` to learn from the real data and then generate some synthetic data.

```python3
from ganblr import get_demo_data
from ganblr.models import GANBLR

# this is a discrete version of adult since GANBLR requires discrete data.
df = get_demo_data('adult')
x, y = df.values[:,:-1], df.values[:,-1]

model = GANBLR()
model.fit(x, y, epochs = 10)

#generate synthetic data
synthetic_data = model.sample(1000)
```

The steps to generate synthetic data using `GANBLR++` are similar to `GANBLR`, but require an additional parameter `numerical_columns` to tell the model the index of the numerical columns.

```python3
from ganblr import get_demo_data
from ganblr.models import GANBLRPP
import numpy as np

# raw adult
df = get_demo_data('adult-raw')
x, y = df.values[:,:-1], df.values[:,-1]

def is_numerical(dtype):
    return dtype.kind in 'iuf'

column_is_numerical = df.dtypes.apply(is_numerical).values
numerical_columns = np.argwhere(column_is_numerical).ravel()

model = GANBLRPP(numerical_columns)
model.fit(x, y, epochs = 10)

#generate synthetic data
synthetic_data = model.sample(1000)
```

# Documentation

You can check the documentation at [https://ganblr-docs.readthedocs.io/en/latest/](https://ganblr-docs.readthedocs.io/en/latest/).
# Model Evaluation

We provide comprehensive evaluation scripts for both Train on Real, Test on Real (TRTR) and Train on Synthetic, Test on Real (TSTR) methodologies.

## Evaluation Scripts

- `compare_models_real_data.py`: Evaluates models on real data (TRTR)
- `eval_tstr_final.py`: Implements TSTR evaluation for generative models

## TSTR Leaderboard

Here we show the results of the TSTR (Train on Synthetic, Test on Real) evaluation on various datasets based on the experiments in our paper.

TRTR (Train on Real, Test on Real) is used as the baseline for comparison.

### Adult Dataset Results

|          | LR     | MLP    | RF     | AVG    |
|----------|--------|--------|--------|--------|
| TRTR     | 0.8741 | 0.8561 | 0.8379 | 0.8560 |
| RLiG     | 0.8250 | 0.8050 | 0.8214 | 0.8171 |
| GANBLR++ | 0.8244 | 0.8219 | 0.8036 | 0.8166 |
| GANBLR   | 0.7856 | 0.7852 | 0.7544 | 0.7751 |
| CTGAN    | 0.8258 | 0.8254 | 0.8257 | 0.8256 |
| NB       | 0.7607 | 0.7607 | 0.7607 | 0.7607 |

# Citation
If you use GANBLR, please cite the following work:

*Y. Zhang, N. A. Zaidi, J. Zhou and G. Li*, "GANBLR: A Tabular Data Generation Model," 2021 IEEE International Conference on Data Mining (ICDM), 2021, pp. 181-190, doi: 10.1109/ICDM51629.2021.00103.

```LaTeX
@inproceedings{ganblr,
    author={Zhang, Yishuo and Zaidi, Nayyar A. and Zhou, Jiahui and Li, Gang},  
    booktitle={2021 IEEE International Conference on Data Mining (ICDM)},   
    title={GANBLR: A Tabular Data Generation Model},   
    year={2021},  
    pages={181-190},  
    doi={10.1109/ICDM51629.2021.00103}
}
@inbook{ganblrpp,
    author = {Yishuo Zhang and Nayyar Zaidi and Jiahui Zhou and Gang Li},
    title = {<bold>GANBLR++</bold>: Incorporating Capacity to Generate Numeric Attributes and Leveraging Unrestricted Bayesian Networks},
    booktitle = {Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
    pages = {298-306},
    doi = {10.1137/1.9781611977172.34},
}
```