# Validation-only encoding experiment

Run from the repository root with `python -m steel_load.encoding_experiment --output docs/encoding-experiment.json`. Use a fresh output path for a repeat. The runner uses the [frozen validation settings](selected-configurations.json) and saves every seed, fit, checkpoint, and loss trace in [the results JSON](encoding-experiment.json).

The experiment covered calendar-only and combined calendar/electrical inputs (CLI key `electrical`) for the two Python models. It used the original split seed 42 and initialization seeds 42, 7, and 101. Random split labels were used only for stratified allocation before the test partition was discarded. Validation used 3,504 random or 3,552 chronological rows. All encodings shared the original train-fitted scaling and category vocabulary. B and C are exact column subsets of A. No production preprocessor or k-NN encoding was changed.

A retains seven day and two WeekStatus dummies. B drops the two WeekStatus dummies. C also drops Monday, represented by six zeros. With the bias included, calendar designs had ranks A 9/12, B 9/10, C 9/9. Combined designs had ranks A 13/16, B 13/14, C 13/13. Both partitions produced those same ranks.

Each split, feature set, and model used the same settings across all arms. Logistic used learning rate 0.3 and L2=0. Neural used learning rate 0.1 and ReLU with 32 hidden units on random and 16 on chronological. Momentum was 0.9, patience was 30, and the epoch cap was 600. Identical seeds control the random-number streams. Changing input dimension also changes which draws are assigned to the weights, so starting logits can differ. Run order rotated across seeds. All A runs reproduced the prior checkpoint loss and confusion matrix.

## Results

Metrics and best loss are means over the three restored validation checkpoints. Recall lists Light / Medium / Maximum. Checkpoints and epochs run list seeds 42 / 7 / 101. Seconds is median training wall time, including per-epoch validation. Speed is the median first epoch at or below a common loss target, with the fraction of seeds that reached it. Each comparison uses the median A checkpoint loss plus 0.01 as its target. Curves are available in the [plot](encoding-loss-curves.png). Thin lines represent individual seeds, thick lines show their median through the last common epoch, and the dotted line is the common target.

| Split | Features | Model | Arm | Accuracy | Macro F1 | Recall | Loss | Epoch to target | Checkpoints | Epochs run | Seconds |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | ---: |
| random | calendar | logistic | A | 74.97% | 0.6912 | 0.926 / 0.511 / 0.630 | 0.6062 | 40 (3/3) | 150/150/150 | 180/180/180 | 0.92 |
| random | calendar | logistic | B | 75.06% | 0.6924 | 0.926 / 0.507 / 0.640 | 0.6062 | 45 (3/3) | 183/183/183 | 213/213/213 | 1.03 |
| random | calendar | logistic | C | 74.74% | 0.6903 | 0.919 / 0.508 / 0.641 | 0.6063 | 54 (3/3) | 457/457/457 | 487/487/487 | 2.32 |
| random | calendar | neural | A | 79.97% | 0.7533 | 0.922 / 0.620 / 0.736 | 0.4910 | 431 (3/3) | 600/600/600 | 600/600/600 | 12.50 |
| random | calendar | neural | B | 77.28% | 0.7234 | 0.906 / 0.549 / 0.740 | 0.5401 | — (0/3) | 600/600/600 | 600/600/600 | 12.27 |
| random | calendar | neural | C | 79.24% | 0.7440 | 0.924 / 0.591 / 0.735 | 0.5115 | — (0/3) | 600/600/600 | 600/600/600 | 12.19 |
| random | electrical | logistic | A | 79.02% | 0.7459 | 0.932 / 0.547 / 0.761 | 0.4889 | 288 (3/3) | 600/600/600 | 600/600/600 | 2.92 |
| random | electrical | logistic | B | 79.02% | 0.7461 | 0.932 / 0.548 / 0.761 | 0.4905 | 317 (3/3) | 600/600/600 | 600/600/600 | 2.94 |
| random | electrical | logistic | C | 79.08% | 0.7470 | 0.931 / 0.552 / 0.761 | 0.4909 | 317 (3/3) | 600/600/600 | 600/600/600 | 2.89 |
| random | electrical | neural | A | 86.19% | 0.8272 | 0.958 / 0.760 / 0.760 | 0.3548 | 519 (3/3) | 600/600/600 | 600/600/600 | 14.78 |
| random | electrical | neural | B | 86.42% | 0.8297 | 0.958 / 0.765 / 0.762 | 0.3516 | 494 (3/3) | 600/600/600 | 600/600/600 | 13.98 |
| random | electrical | neural | C | 86.36% | 0.8287 | 0.959 / 0.762 / 0.761 | 0.3542 | 500 (3/3) | 600/600/600 | 600/600/600 | 15.84 |
| chronological | calendar | logistic | A | 81.70% | 0.7906 | 0.854 / 0.674 / 0.899 | 0.5370 | 45 (3/3) | 600/600/600 | 600/600/600 | 3.07 |
| chronological | calendar | logistic | B | 81.70% | 0.7906 | 0.854 / 0.674 / 0.899 | 0.5370 | 55 (3/3) | 600/600/600 | 600/600/600 | 2.93 |
| chronological | calendar | logistic | C | 81.22% | 0.7843 | 0.855 / 0.670 / 0.875 | 0.5347 | 54 (3/3) | 194/194/194 | 224/224/224 | 1.05 |
| chronological | calendar | neural | A | 89.15% | 0.8736 | 0.890 / 0.835 / 0.972 | 0.4450 | 372 (3/3) | 600/600/600 | 600/600/600 | 7.44 |
| chronological | calendar | neural | B | 87.14% | 0.8519 | 0.875 / 0.778 / 0.984 | 0.4602 | 398.5 (2/3) | 600/243/600 | 600/273/600 | 7.32 |
| chronological | calendar | neural | C | 84.67% | 0.8264 | 0.848 / 0.729 / 0.998 | 0.4712 | — (0/3) | 600/600/600 | 600/600/600 | 7.26 |
| chronological | electrical | logistic | A | 86.29% | 0.8393 | 0.927 / 0.634 / 0.979 | 0.3630 | 383 (3/3) | 600/600/600 | 600/600/600 | 2.78 |
| chronological | electrical | logistic | B | 86.23% | 0.8387 | 0.926 / 0.633 / 0.979 | 0.3639 | 398 (3/3) | 600/600/600 | 600/600/600 | 2.73 |
| chronological | electrical | logistic | C | 86.23% | 0.8387 | 0.926 / 0.633 / 0.979 | 0.3645 | 404 (3/3) | 600/600/600 | 600/600/600 | 2.68 |
| chronological | electrical | neural | A | 93.33% | 0.9262 | 0.949 / 0.865 / 0.978 | 0.2317 | 517 (3/3) | 600/600/600 | 600/600/600 | 7.78 |
| chronological | electrical | neural | B | 92.21% | 0.9099 | 0.950 / 0.818 / 0.979 | 0.2500 | — (0/3) | 600/600/600 | 600/600/600 | 7.90 |
| chronological | electrical | neural | C | 92.63% | 0.9158 | 0.950 / 0.838 / 0.974 | 0.2415 | 568 (1/3) | 600/600/600 | 600/600/600 | 7.65 |

## Paired macro-F1 differences

Differences are in percentage points relative to A. The range is the smallest and largest paired seed difference.

| Split | Features | Model | B minus A, mean (range) | C minus A, mean (range) |
| --- | --- | --- | ---: | ---: |
| random | calendar | logistic | +0.13 (+0.13 to +0.13) | -0.09 (-0.09 to -0.09) |
| random | calendar | neural | -2.99 (-4.40 to -1.82) | -0.93 (-3.29 to +0.43) |
| random | electrical | logistic | +0.02 (+0.02 to +0.02) | +0.11 (+0.11 to +0.11) |
| random | electrical | neural | +0.25 (+0.14 to +0.32) | +0.15 (-0.13 to +0.30) |
| chronological | calendar | logistic | +0.00 (+0.00 to +0.00) | -0.63 (-0.63 to -0.63) |
| chronological | calendar | neural | -2.17 (-5.05 to -0.42) | -4.72 (-6.56 to -3.20) |
| chronological | electrical | logistic | -0.06 (-0.06 to -0.06) | -0.06 (-0.06 to -0.06) |
| chronological | electrical | neural | -1.63 (-2.14 to -1.04) | -1.05 (-2.63 to -0.25) |

## Recommendation

Keep **A as the default** for the frozen models. C removed exact collinearity, but it did not produce a consistent validation or convergence benefit at the frozen settings.

- Logistic regression stayed within 0.63 macro-F1 points of A across all comparisons. C reached the common loss later in all four comparisons. Its chronological calendar fit ended earlier because of patience (epoch 224 versus the 600-epoch cap), even though it reached the common loss later (54 versus 45 epochs). C is an option for a separate logistic model where identifiable calendar contrasts matter. The evidence does not support a speed improvement.
- For calendar-only neural fits, neither reduced encoding reached the random-split common target at any seed. On chronological validation, C also missed the target at all seeds and lost 4.72 macro-F1 points on average. Medium recall fell from 0.835 to 0.729.
- Both reduced encodings helped slightly on the random combined neural fits, with B reaching the common loss 25 epochs earlier and improving mean macro F1 by 0.25 points. On the chronological combined network, both reduced encodings lost macro F1 at all three seeds. A reached the common loss at all seeds, compared with zero for B and one for C.

Treat these as comparisons of encoding under the frozen optimizer and initialization scheme. Changing encoding changes gradient geometry and weight initialization, and the three fit seeds share the same validation rows. If reference coding is needed, evaluate a model-specific configuration with validation-only learning-rate retuning and, to isolate optimizer effects, equivalent initial logits.

The experiment completed **72 fits in 7 minutes 52 seconds**. The existing 12 correctness tests passed, including the compiled k-NN integration test. Final test sets were left unscored.
