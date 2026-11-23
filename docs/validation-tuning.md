# Validation tuning and final configuration recommendations

This stage used only the training and validation partitions defined in the [pilot](validation-pilot.md). The random split keeps its original seed 42 and holdout membership. Chronological training and validation retain the September 13 and October 20 boundaries. Target labels were used by the random splitter to allocate stratified partitions. Each test partition was then discarded, and its labels did not enter model training, metrics, or diagnostics.

Run from the repository root with the C++ Release executable built:

```powershell
python -m steel_load.tune --knn-executable build/Release/steel_knn.exe --output docs/validation-tuning.json
```

The [complete results](validation-tuning.json) record each candidate's validation accuracy, macro F1, recall in Light / Medium / Maximum order, timing, and diagnostic details.

## Why chronological calendar k-NN jumped at k=19

A schedule baseline predicts the most common training label for each weekday and time-of-day pair, falling back to the time-only majority and then the overall training majority if necessary. It reached **84.96% random validation accuracy** (macro F1 0.8173) and **93.69% chronological validation accuracy** (macro F1 0.9356). Chronological recall was 0.8871 / 1.0000 / 1.0000.

All 3,552 chronological queries had **36–37 exact training matches** in the encoded calendar space. Across 672 training weekday/time combinations, 392 had conflicting labels. At k=3, 11, and 19, an independent vote over the first k exact training matches reproduced every C++ prediction. Removing min-max scaling from the sine and cosine columns changed zero predictions at k=3, 11, 19, and 51.

| k | Native training order accuracy | Reversed training order accuracy | Native predictions different from schedule baseline |
| ---: | ---: | ---: | ---: |
| 3 | 68.47% | 85.81% | 1,024 |
| 11 | 68.47% | 85.81% | 1,024 |
| 19 | 91.89% | 93.69% | 80 |
| 51 | 93.69% | 93.69% | 0 |

The smaller k values favor the earliest tied rows. At Thursday 17:15 on September 13, the first three exact neighbors all vote Maximum. The first 19 vote Light 1, Medium 11, Maximum 7, and the observed label is Medium. The earliest training period contains a different recurring schedule. The 46 days of that profile occur in January and February. Training also contains 170 days of the more common March–September profile and 39 all-Light days. Validation contains 28 common-profile days and nine all-Light days. It contains zero examples of the January–February profile or of a new profile.

Training class counts were 12,384 Light, 6,912 Medium, and 5,184 Maximum. Validation counts were 1,984, 896, and 672. The stronger chronological calendar scores are consistent with the shifted profile mix and the availability of repeated calendar keys.

Two additional calendar checks used only earlier months within the original training period. Each fit used all rows before the start of the check month. March accuracy was 66.40% for the schedule baseline and 65.83% for k-NN at k=51. August accuracy was 88.71% and 87.97%, respectively. The March window demonstrates the cost of a profile change. These sanity checks were excluded from the tuning criterion.

## Search and feature availability

- k-NN searched all odd k from 1 through 51. Selection maximizes mean validation macro F1 across native training order and two fixed training-row permutations (seeds 7 and 101). Smaller k wins exact selection ties. The permutations preserve all reference rows and measure tie-order sensitivity.
- Logistic regression screened learning rates 0.1 and 0.3 with L2 coefficients 0, 0.001, and 0.01. The penalty is applied to weights only. Validation cross-entropy excludes the penalty.
- The neural network screened widths 16 and 32, learning rates 0.05 and 0.1, and ReLU and tanh activations. Both Python models used momentum 0.9 and validation-loss patience of 30 epochs. The 240-epoch screen used seed 42. The two lowest-loss candidates per model, split, and feature set advanced to the 600-epoch cap with initialization seeds 42, 7, and 101. Their mean checkpoint validation loss selected the winner, and seed 42 provides the fixed reference configuration. These seed checks hold partition membership fixed.
- Calendar-only uses advance-available time inputs. Electrical-only uses four same-interval reactive energy and power-factor values. Combined uses those four plus calendar signals and corresponds to the existing CLI feature key `electrical`. The prior pilot separately examined active usage and CO₂ in `interval`. Those two inputs were excluded here to isolate the contribution of the four electrical readings. Same-interval measurements require confirmed delivery before the decision cutoff. The calendar experiment establishes schedule-label predictability within observed profiles. The source labeling rule and production workload meaning remain unverified.


## Selected configurations by feature set

Scores and recall use the fixed reference configuration. Stability is the macro-F1 range across initialization seeds or k-NN reference orders. Python timings include validation loss at every epoch. k-NN has no separate fit, so its times are validation queries.

| Split | Features | Model | Settings | Accuracy | Macro F1 | Recall (Light / Medium / Maximum) | Macro-F1 stability | Seconds |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: |
| random | calendar | logistic | LR 0.3, L2 0 | 74.97% | 0.6912 | 0.926 / 0.511 / 0.630 | 0.6912?0.6912 | 1.11 |
| random | calendar | neural | 32 relu, LR 0.1 | 79.94% | 0.7532 | 0.923 / 0.613 / 0.740 | 0.7494?0.7574 | 14.29 |
| random | calendar | knn | k=31 | 84.96% | 0.8173 | 0.919 / 0.808 / 0.733 | 0.8173?0.8173 | 1.76 query |
| random | electrical-only | logistic | LR 0.3, L2 0 | 72.29% | 0.6485 | 0.915 / 0.344 / 0.750 | 0.6485?0.6490 | 2.70 |
| random | electrical-only | neural | 32 relu, LR 0.1 | 72.43% | 0.6519 | 0.921 / 0.377 / 0.697 | 0.6505?0.6568 | 15.39 |
| random | electrical-only | knn | k=47 | 75.11% | 0.6871 | 0.930 / 0.537 / 0.593 | 0.6871?0.6871 | 0.73 query |
| random | electrical | logistic | LR 0.3, L2 0 | 79.02% | 0.7459 | 0.932 / 0.547 / 0.761 | 0.7459?0.7459 | 3.81 |
| random | electrical | neural | 32 relu, LR 0.1 | 85.99% | 0.8250 | 0.956 / 0.753 / 0.763 | 0.8250?0.8299 | 12.72 |
| random | electrical | knn | k=7 | 87.53% | 0.8427 | 0.962 / 0.800 / 0.761 | 0.8427?0.8427 | 1.74 query |
| chronological | calendar | logistic | LR 0.3, L2 0 | 81.70% | 0.7906 | 0.854 / 0.674 / 0.899 | 0.7906?0.7906 | 2.67 |
| chronological | calendar | neural | 16 relu, LR 0.1 | 89.78% | 0.8826 | 0.890 / 0.839 / 1.000 | 0.8617?0.8826 | 7.55 |
| chronological | calendar | knn | k=25 | 93.69% | 0.9356 | 0.887 / 1.000 / 1.000 | 0.9356?0.9356 | 1.63 query |
| chronological | electrical-only | logistic | LR 0.3, L2 0 | 76.21% | 0.6415 | 0.975 / 0.199 / 0.884 | 0.6415?0.6421 | 2.44 |
| chronological | electrical-only | neural | 16 tanh, LR 0.1 | 75.87% | 0.6565 | 0.975 / 0.280 / 0.757 | 0.6525?0.6591 | 9.48 |
| chronological | electrical-only | knn | k=17 | 71.09% | 0.6546 | 0.816 / 0.398 / 0.818 | 0.6546?0.6549 | 0.75 query |
| chronological | electrical | logistic | LR 0.3, L2 0 | 86.29% | 0.8393 | 0.927 / 0.634 / 0.979 | 0.8393?0.8393 | 2.81 |
| chronological | electrical | neural | 16 relu, LR 0.1 | 93.24% | 0.9266 | 0.945 / 0.873 / 0.976 | 0.9206?0.9316 | 8.27 |
| chronological | electrical | knn | k=15 | 91.02% | 0.9074 | 0.903 / 0.890 / 0.960 | 0.9074?0.9074 | 1.64 query |

## Recommendations for final evaluation

The [selected configurations](selected-configurations.json) freeze the split seed, dataset hash, feature keys, training settings, Python checkpoints, and k-NN reference order. The same-interval recommendations use the highest mean validation macro F1 across the three stability runs among the selected electrical-only and combined models.

- **Advance prediction on either split.** Use the weekday/time schedule baseline. It matches the tuned calendar k-NN scores on both splits and removes sensitivity to equal-distance row order. For a learned comparison, use calendar k-NN at k=31 for random and k=25 for chronological. The March check limits confidence when the schedule changes.
- **Random same-interval classification.** Use combined k-NN at k=7. Validation accuracy is **87.53%** and macro F1 is **0.8427**. The selected score was unchanged across the three tested reference orders.
- **Chronological same-interval classification.** Use the combined 16-unit ReLU network, learning rate 0.1, seed 42, and checkpoint 600. Validation accuracy is **93.24%** and macro F1 is **0.9266**, with a macro-F1 range of **0.9206–0.9316**. The advance-available schedule baseline scored slightly higher on this window. Include it when comparing the added value of meter inputs.

Adding calendar features to the four electrical readings substantially improved every model's validation macro F1 on both splits. The strongest electrical-only settings reached 0.6871 on random validation and 0.6565 on chronological validation at the reference seed or order. This reinforces how much task performance depends on schedule information. These validation windows and fit seeds cannot establish performance on new schedule regimes.

## Convergence and cost

Learning rate 0.3 and L2=0 won all six logistic selections by mean minimum validation loss within the tested grid. Random calendar logistic stopped at epoch 180 and returned epoch 150. The remaining reference logistic selections reached epoch 600. Their validation loss changed by at most 0.00086 over the last 25 epochs. Across both finalist configurations and all seeds, nine of 36 logistic fits stopped on patience.

One of 36 neural finalist fits stopped on patience. Each reference neural winner reached epoch 600, with validation loss still improving by 0.00074–0.00293 over the last 25 epochs. The capped configurations are reproducible, and further training might change their ordering.

The search took **13 minutes 22 seconds** with one BLAS thread. Every fit was capped at 600 epochs. Twelve correctness tests passed, including the C++ integration check, schedule fallbacks, electrical-only features, early stopping, and the L2 weight update. This stage left final-test scoring pending.
