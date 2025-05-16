# Model Evaluation Results Summary

## Overview

- Total datasets evaluated: 16
- Total models compared: 8
- Overall average performance: 0.6763

## Model Rankings

| Rank | Model | Average Score |
|------|-------|---------------|
| 1 | RLiG | 0.8575 |
| 2 | CTABGAN | 0.8452 |
| 3 | GANBLR | 0.8345 |
| 4 | GANBLR++ | 0.8264 |
| 5 | NB | 0.5821 |
| 6 | GREAT | 0.5197 |
| 7 | TABSYN | 0.4793 |
| 8 | CTGAN | 0.4654 |

## Top Performing Model-Classifier Combinations

| Rank | Model-Classifier | Average Score |
|------|-----------------|---------------|
| 1 | RLiG-MLP | 0.8614 |
| 2 | RLiG-RF | 0.8591 |
| 3 | CTABGAN-LR | 0.8533 |
| 4 | CTABGAN-MLP | 0.8524 |
| 5 | RLiG-LR | 0.8520 |
| 6 | GANBLR-MLP | 0.8466 |
| 7 | GANBLR-LR | 0.8433 |
| 8 | CTABGAN-XGB | 0.8420 |
| 9 | GANBLR++-MLP | 0.8405 |
| 10 | CTABGAN-RF | 0.8332 |

## Dataset Performance

Best performing model for each dataset:

| Dataset | Best Model | Score |
|---------|------------|-------|
| Adult | RLiG | 0.8768 |
| Car | RLiG | 0.8765 |
| Chess | CTABGAN | 0.8780 |
| Connect-4 | RLiG | 0.8749 |
| Credit | CTABGAN | 0.8804 |
| letter_rocog | CTABGAN | 0.8815 |
| Loan | CTABGAN | 0.8771 |
| Magic | CTABGAN | 0.8785 |
| 'Maternal | RLiG | 0.8844 |
| Health' | RLiG | 0.7946 |
| Nursery | CTABGAN | 0.8764 |
| PokerHand | CTABGAN | 0.8773 |
| Rice | CTABGAN | 0.8723 |
| 'Room | RLiG | 0.8917 |
| Occupancy' | RLiG | 0.7031 |
| TicTacToe | CTABGAN | 0.8810 |


## Efficiency Metrics

| Rank | Model | Average Score | Average Time (s) | Efficiency (Score/Time) |
|------|-------|---------------|------------------|-------------------------|
| 1 | GANBLR | 0.8345 | 1.61 | 0.517175 |
| 2 | CTABGAN | 0.8452 | 1.72 | 0.492412 |
| 3 | GANBLR++ | 0.8264 | 1.70 | 0.486674 |
| 4 | GREAT | 0.5197 | 1.78 | 0.291946 |
| 5 | TABSYN | 0.4793 | 2.62 | 0.182949 |
| 6 | CTGAN | 0.4654 | 2.63 | 0.176907 |
| 7 | RLiG | 0.8575 | 7.70 | 0.111299 |
| 8 | NB | 0.5821 | 27.75 | 0.020977 |
