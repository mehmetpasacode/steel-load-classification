# Final holdout benchmark

The benchmark scored each held-out partition once after verifying the [extended neural freeze](encoding-extended-frozen.json), the [original model freezes](selected-configurations.json), and the dataset fingerprint. Every learned model used the combined calendar and electrical feature group. It includes cyclic time and four reactive-energy/power-factor readings. The model inputs themselves contain no active energy or CO2.

- The neural network used encoding C with 12 inputs, seed 42, ReLU, learning rate 0.1, momentum 0.9, and patience 30. The random and chronological widths were 32 and 16. Both restored the frozen epoch-2,000 checkpoint. Neural encoding A remains a [validation-only ablation](encoding-extended.md).
- Logistic regression used the original 15 columns, seed 42, learning rate 0.3, L2=0, momentum 0.9, and the frozen 600-epoch checkpoint on both splits.
- C++ k-NN used the original 15 columns and native training order, with k=7 for random and k=15 for chronological. The training partition was the entire reference pool. Held-out rows entered as unlabeled queries.
- Majority used the training class mode (Light). The calendar schedule predicted the training majority by weekday and time, with time-only and overall fallbacks.

All three partitions within each split were disjoint. The scaler's row count and bounds and the one-hot vocabularies were verified directly against the training frame. Both Python models reproduced their validation scores and checkpoint epochs. k-NN and the schedule baseline reproduced their frozen validation confusion matrices. Both sets of checks completed before holdout scoring. The random splitter used labels to allocate stratified partitions, and test labels were used only for final metrics after predictions were made.

## Held-out metrics

Each test split contains 7,008 rows. Recall follows Light / Medium / Maximum. Times are wall seconds for model fit and held-out prediction, with shared preprocessing excluded. Python fitting includes validation monitoring. k-NN has no separate fit and its prediction time includes serialization and process startup. Schedule time includes the training lookup and prediction.

| Split | Model | Accuracy | Macro F1 | Recall | Fit / predict (s) |
| --- | --- | ---: | ---: | --- | ---: |
| Random | Neural network | 87.17% | 0.8386 | 0.961 / 0.749 / 0.812 | 49.5276 / 0.0033 |
| Random | Logistic regression | 79.68% | 0.7513 | 0.938 / 0.548 / 0.778 | 3.1244 / 0.0010 |
| Random | k-NN | 87.47% | 0.8415 | 0.961 / 0.793 / 0.769 | 0.0000 / 2.8041 |
| Random | Majority baseline | 51.58% | 0.2269 | 1.000 / 0.000 / 0.000 | 0.0000 / 0.0000 |
| Random | Calendar schedule | 84.66% | 0.8150 | 0.913 / 0.795 / 0.750 | 0.0499 combined |
| Chronological | Neural network | 74.80% | 0.6561 | 0.960 / 0.549 / 0.459 | 27.5855 / 0.0020 |
| Chronological | Logistic regression | 68.32% | 0.5960 | 0.896 / 0.410 / 0.490 | 2.8301 / 0.0009 |
| Chronological | k-NN | 73.53% | 0.6496 | 0.930 / 0.561 / 0.459 | 0.0000 / 2.7647 |
| Chronological | Majority baseline | 52.85% | 0.2305 | 1.000 / 0.000 / 0.000 | 0.0000 / 0.0000 |
| Chronological | Calendar schedule | 71.80% | 0.6315 | 0.909 / 0.559 / 0.429 | 0.0528 combined |

Confusion matrices below use actual rows and predicted columns, with Light / Medium / Maximum on both axes.

- **Random.** Neural network `[[3475, 129, 11], [117, 1453, 369], [28, 245, 1181]]`, Logistic regression `[[3390, 223, 2], [384, 1063, 492], [51, 272, 1131]]`, k-NN `[[3474, 128, 13], [100, 1538, 301], [42, 294, 1118]]`, Majority baseline `[[3615, 0, 0], [1939, 0, 0], [1454, 0, 0]]`, Calendar schedule `[[3301, 188, 126], [102, 1541, 296], [73, 290, 1091]]`.
- **Chronological.** Neural network `[[3555, 144, 5], [80, 1037, 771], [84, 682, 650]]`, Logistic regression `[[3320, 379, 5], [215, 774, 899], [99, 623, 694]]`, k-NN `[[3443, 254, 7], [70, 1060, 758], [128, 638, 650]]`, Majority baseline `[[3704, 0, 0], [1888, 0, 0], [1416, 0, 0]]`, Calendar schedule `[[3368, 192, 144], [96, 1056, 736], [72, 736, 608]]`.

## Historical context and limitations

| Model | Course report accuracy | Corrected random accuracy | Corrected chronological accuracy |
| --- | ---: | ---: | ---: |
| Neural network | 89.07% | 87.17% | 74.80% |
| Logistic regression | 81.12% | 79.68% | 68.32% |
| k-NN | approximately 86% | 87.47% | 73.53% |

The [historical archive](../historical/README.md) documents the earlier protocol. It used different preprocessing and candidate selection. The original k-NN searched other held-out rows as potential neighbors and normalized across the full dataset. The original neural sweep exposed the test set across candidates, and an original confusion matrix swapped the Medium and Maximum display labels. The historical accuracy figures remain useful context. Differences also reflect split, feature, and training choices, so they cannot isolate the impact of any one correction.

The combined k-NN led the random split, while the frozen neural network led chronological evaluation. The neural network's chronological macro F1 fell from 0.9298 on validation to 0.6561 on test, and the schedule baseline fell from 0.9356 to 0.6315. The drop across both calendar and same-interval approaches limits confidence in generalization to later periods. In the chronological neural confusion matrix, 771 Medium rows were called Maximum and 682 Maximum rows were called Medium. No model or threshold was revised after scoring.

The combined models need measurements from the classified interval. The calendar schedule uses inputs available in advance, but the label-generation rule and measurement delivery times remain unverified. Random stratification can place adjacent intervals in different partitions. The two evaluations share a single plant and year, so they offer limited evidence for other plants or schedule regimes.

## Reproducibility and readiness

The benchmark used the C++ Release executable, one BLAS thread, and the local Windows environment. It completed in 92.8 seconds including validation replay and holdout scoring. The [results JSON](final-benchmark.json) records environment versions, data and configuration hashes, per-split runtimes, per-class metrics, and checkpoint validation checks. The entry point is `steel_load.final_benchmark`. It reserves its output before scoring and rejects reuse of that output path. The existing 12 correctness checks passed, including the C++ integration test. The technical benchmark artifacts are ready for review. The project does not yet declare a source-code license for public distribution.
