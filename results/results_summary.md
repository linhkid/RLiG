# Model Evaluation Results Summary

## Overview

- Total datasets evaluated: 11
- Total models compared: 7
- Overall average performance: 0.6585

## Model Rankings

| Rank | Model | Average Score |
|------|-------|---------------|
| 1 | RLiG | 0.8426 |
| 2 | GANBLR | 0.8079 |
| 3 | GANBLR++ | 0.8007 |
| 4 | NB | 0.5938 |
| 5 | CTGAN | 0.5840 |
| 6 | TABSYN | 0.5004 |
| 7 | GREAT | 0.4798 |

## Top Performing Model-Classifier Combinations

| Rank | Model-Classifier | Average Score |
|------|-----------------|---------------|
| 1 | RLiG-MLP | 0.8514 |
| 2 | RLiG-RF | 0.8495 |
| 3 | RLiG-LR | 0.8270 |
| 4 | GANBLR-LR | 0.8240 |
| 5 | GANBLR++-MLP | 0.8161 |
| 6 | GANBLR-MLP | 0.8133 |
| 7 | GANBLR++-LR | 0.8055 |
| 8 | GANBLR-XGB | 0.8003 |
| 9 | GANBLR-RF | 0.7938 |
| 10 | GANBLR++-RF | 0.7930 |

## Dataset Performance

Best performing model for each dataset:

| Dataset | Best Model | Score |
|---------|------------|-------|
| Magic | RLiG | 0.8740 |
| Nursery | RLiG | 0.7778 |
| Adult | RLiG | 0.8746 |
| PokerHand | RLiG | 0.8702 |
| TicTacToe | RLiG | 0.8787 |
| Chess | RLiG | 0.7442 |
| letter_rocog | RLiG | 0.8898 |
| Connect-4 | RLiG | 0.8796 |
| Rice | RLiG | 0.8775 |
| Car | RLiG | 0.7280 |
| Room Occupancy | RLiG | 0.8745 |


## Efficiency Metrics

| Rank | Model | Average Score | Average Time (s) | Efficiency (Score/Time) |
|------|-------|---------------|------------------|-------------------------|
| 1 | GANBLR | 0.8079 | 1.52 | 0.529803 |
| 2 | GANBLR++ | 0.8007 | 1.61 | 0.496262 |
| 3 | GREAT | 0.4798 | 1.72 | 0.279283 |
| 4 | CTGAN | 0.5840 | 2.42 | 0.241091 |
| 5 | TABSYN | 0.5004 | 2.33 | 0.214613 |
| 6 | RLiG | 0.8426 | 8.22 | 0.102556 |
| 7 | NB | 0.5938 | 22.39 | 0.026523 |
