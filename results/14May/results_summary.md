# Model Evaluation Results Summary

## Overview

- Total datasets evaluated: 14
- Total models compared: 7
- Overall average performance: 0.6484

## Model Rankings

| Rank | Model | Average Score |
|------|-------|---------------|
| 1 | RLiG | 0.8557 |
| 2 | GANBLR | 0.8440 |
| 3 | GANBLR++ | 0.8430 |
| 4 | TABSYN | 0.6070 |
| 5 | NB | 0.5947 |
| 6 | CTGAN | 0.5450 |
| 7 | GREAT | 0.2493 |

## Top Performing Model-Classifier Combinations

| Rank | Model-Classifier | Average Score |
|------|-----------------|---------------|
| 1 | GANBLR-LR | 0.8636 |
| 2 | RLiG-MLP | 0.8586 |
| 3 | RLiG-RF | 0.8583 |
| 4 | GANBLR++-LR | 0.8578 |
| 5 | RLiG-LR | 0.8501 |
| 6 | GANBLR-MLP | 0.8478 |
| 7 | GANBLR++-MLP | 0.8432 |
| 8 | GANBLR++-XGB | 0.8370 |
| 9 | GANBLR-XGB | 0.8347 |
| 10 | GANBLR++-RF | 0.8341 |

## Dataset Performance

Best performing model for each dataset:

| Dataset | Best Model | Score |
|---------|------------|-------|
| Rice | GANBLR++ | 0.9101 |
| Car | RLiG | 0.7454 |
| PokerHand | GANBLR++ | 0.9105 |
| Adult | RLiG | 0.9096 |
| TicTacToe | GANBLR++ | 0.9101 |
| Chess | RLiG | 0.7454 |
| letter_rocog | GANBLR++ | 0.9101 |
| Magic | GANBLR++ | 0.9101 |
| Nursery | RLiG | 0.7407 |
| Room Occupancy | GANBLR++ | 0.9101 |
| Maternal Health | RLiG | 0.9119 |
| Loan | RLiG | 0.7483 |
| Credit | GANBLR | 0.9069 |
| Connect-4 | GANBLR++ | 0.9105 |


## Efficiency Metrics

| Rank | Model | Average Score | Average Time (s) | Efficiency (Score/Time) |
|------|-------|---------------|------------------|-------------------------|
| 1 | GANBLR++ | 0.8430 | 1.67 | 0.504238 |
| 2 | GANBLR | 0.8440 | 1.71 | 0.492998 |
| 3 | TABSYN | 0.6070 | 2.53 | 0.239519 |
| 4 | CTGAN | 0.5450 | 2.53 | 0.215323 |
| 5 | RLiG | 0.8557 | 10.00 | 0.085535 |
| 6 | GREAT | 0.2493 | 3.05 | 0.081743 |
| 7 | NB | 0.5947 | 25.81 | 0.023045 |
