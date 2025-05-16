# Model Evaluation Results Summary

## Overview

- Total datasets evaluated: 15
- Total models compared: 8
- Overall average performance: 0.6903
- n_bins: 7
- quantile

## Model Rankings

| Rank | Model      | Average Score |
|------|------------|---------------|
| 1 | GANBLR     | 0.8927 |
| 2 | GANBLR++   | 0.8919 |
| 3 | RLiG       | 0.8895 |
| 4 | CTABGAN    | 0.8385 |
| 5 | DIST_SAMPL | 0.6098 |
| 6 | CTGAN      | 0.5797 |
| 7 | NB         | 0.5778 |
| 8 | GREAT      | 0.2424 |

## Top Performing Model-Classifier Combinations

| Rank | Model-Classifier | Average Score |
|------|-----------------|---------------|
| 1 | GANBLR-LR | 0.9048 |
| 2 | GANBLR++-LR | 0.9008 |
| 3 | RLiG-LR | 0.8967 |
| 4 | GANBLR++-XGB | 0.8956 |
| 5 | GANBLR-MLP | 0.8936 |
| 6 | GANBLR-XGB | 0.8910 |
| 7 | GANBLR++-MLP | 0.8873 |
| 8 | RLiG-RF | 0.8859 |
| 9 | RLiG-MLP | 0.8857 |
| 10 | GANBLR++-RF | 0.8837 |

## Dataset Performance

Best performing model for each dataset:

| Dataset | Best Model | Score |
|---------|------------|-------|
| Adult | GANBLR | 0.9069 |
| Chess | RLiG | 0.9108 |
| Connect-4 | GANBLR | 0.9069 |
| Credit | GANBLR | 0.9069 |
| NSL-KDD | GANBLR++ | 0.9101 |
| letter_rocog | GANBLR++ | 0.9105 |
| Loan | GANBLR++ | 0.9101 |
| Magic | GANBLR++ | 0.9105 |
| Maternal Health | RLiG | 0.9076 |
| Nursery | GANBLR++ | 0.9101 |
| PokerHand | GANBLR++ | 0.9105 |
| Rice | GANBLR++ | 0.9101 |
| Car | RLiG | 0.7054 |
| Room Occupancy | RLiG | 0.9144 |
| TicTacToe | GANBLR | 0.9069 |


## Efficiency Metrics

| Rank | Model | Average Score | Average Time (s) | Efficiency (Score/Time) |
|------|-------|---------------|------------------|-------------------------|
| 1 | GANBLR++ | 0.8919 | 1.92 | 0.465501 |
| 2 | GANBLR | 0.8927 | 1.97 | 0.453148 |
| 3 | CTABGAN | 0.8385 | 2.10 | 0.400112 |
| 4 | TABSYN | 0.6098 | 3.13 | 0.195010 |
| 5 | CTGAN | 0.5797 | 3.20 | 0.180902 |
| 6 | RLiG | 0.8895 | 9.37 | 0.094961 |
| 7 | GREAT | 0.2424 | 3.64 | 0.066670 |
| 8 | NB | 0.5778 | 33.94 | 0.017022 |
