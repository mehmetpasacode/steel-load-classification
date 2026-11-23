# Validation-only pilot

The pilot used the bundled 35,040-row dataset, seed 42, and both existing splits. The random stratified split supplied 24,528 training and 3,504 validation rows. The chronological split supplied 24,480 training and 3,552 validation rows. Chronological validation ran from September 13 through October 19, 2018. The runner discarded both test partitions immediately after splitting. The random splitter used labels solely to allocate stratified partitions. No test labels entered encoding, preprocessing, fitting, metrics, diagnostics, or model selection.

```powershell
python -m steel_load.verify --knn-executable build/Release/steel_knn.exe --output docs/validation-pilot.json
```

[Machine-readable results](validation-pilot.json) include fitted feature names, per-class recall, confusion matrices, configuration, timing, and both Python loss histories.

## Implementation checks

- All three models receive the same matrices from the shared preprocessing function. The timestamp supplies cyclic time of day and calendar categories. The three feature sets have 11, 15, and 17 encoded columns. A training-fitted min-max scaler handles numeric columns, and a training-fitted one-hot encoder handles calendar categories. Validation values can fall outside the training scale and unknown categories encode as all zeros. Squared Euclidean distance in C++ uses those same numeric and one-hot values.
- Label IDs 0, 1, and 2 represent Light, Medium, and Maximum. NumPy one-hot targets, C++ votes, confusion matrix axes, and recall use that order. The majority baseline picks the most frequent training label, with the same order for ties. Both splits selected Light.
- The Python/C++ protocol sends only training labels and training feature rows as the reference pool. Validation arrives as separate unlabeled queries. In a spot check, the compiled binary agreed with an independent stable-sort NumPy implementation on 54 real validation queries across both splits and all three feature sets, for all three k values.
- Nine automated checks passed, including the compiled C++ integration test, split boundaries, held-out category and scale handling, matrix ordering, the majority baseline, and checkpoint restoration.

## Results

Logistic regression and the neural network used full-batch momentum 0.9, learning rate 0.1, cross-entropy, a cap of 300 epochs, and patience of 20 epochs. The neural network used 16 ReLU units and a softmax output. k-NN evaluated k = 3, 11, and 19 in one pass per feature set. Its displayed result uses the k with highest validation accuracy. These k-NN scores reflect selection on this validation partition.

Recall is listed in Light / Medium / Maximum order. Python times include per-epoch validation loss calculation. The k-NN timing is end-to-end validation query time, including serialization and process startup, with no separate fitting step. Shared preprocessing took 0.04–0.05 s per configuration.

| Split | Features | Model | Accuracy | Macro F1 | Recall | Fit s | k-NN query s |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Random | — | Majority | 51.57% | 0.2268 | 1.0000 / 0 / 0 | <0.001 | — |
| Random | Calendar | Logistic | 74.97% | 0.6887 | 0.9325 / 0.5186 / 0.6039 | 1.43 | — |
| Random | Calendar | Neural | 78.51% | 0.7344 | 0.9247 / 0.5670 / 0.7290 | 4.16 | — |
| Random | Calendar | k-NN, k=19 | 84.62% | 0.8133 | 0.9175 / 0.8041 / 0.7249 | 0 | 1.13 |
| Random | Electrical | Logistic | 79.45% | 0.7511 | 0.9375 / 0.5567 / 0.7565 | 1.58 | — |
| Random | Electrical | Neural | 83.90% | 0.8006 | 0.9458 / 0.6990 / 0.7607 | 3.94 | — |
| Random | Electrical | k-NN, k=11 | 87.21% | 0.8385 | 0.9629 / 0.7866 / 0.7607 | 0 | 1.52 |
| Random | Interval | Logistic | 78.97% | 0.7450 | 0.9286 / 0.5567 / 0.7552 | 1.48 | — |
| Random | Interval | Neural | 80.17% | 0.7583 | 0.9314 / 0.5825 / 0.7717 | 4.51 | — |
| Random | Interval | k-NN, k=11 | 87.39% | 0.8420 | 0.9613 / 0.7897 / 0.7689 | 0 | 1.77 |
| Chronological | — | Majority | 55.86% | 0.2389 | 1.0000 / 0 / 0 | <0.001 | — |
| Chronological | Calendar | Logistic | 80.83% | 0.7775 | 0.8564 / 0.6741 / 0.8452 | 1.27 | — |
| Chronological | Calendar | Neural | 88.74% | 0.8684 | 0.8911 / 0.7946 / 1.0000 | 4.89 | — |
| Chronological | Calendar | k-NN, k=19 | 91.89% | 0.9101 | 0.8871 / 1.0000 / 0.9048 | 0 | 1.36 |
| Chronological | Electrical | Logistic | 86.18% | 0.8416 | 0.9103 / 0.6663 / 0.9792 | 1.66 | — |
| Chronological | Electrical | Neural | 90.48% | 0.8930 | 0.9269 / 0.8002 / 0.9792 | 4.06 | — |
| Chronological | Electrical | k-NN, k=19 | 90.85% | 0.9061 | 0.8982 / 0.8951 / 0.9568 | 0 | 1.80 |
| Chronological | Interval | Logistic | 86.43% | 0.8429 | 0.9158 / 0.6641 / 0.9792 | 1.52 | — |
| Chronological | Interval | Neural | 88.06% | 0.8580 | 0.9299 / 0.6975 / 0.9792 | 3.88 | — |
| Chronological | Interval | k-NN, k=19 | 90.26% | 0.8984 | 0.8952 / 0.8750 / 0.9613 | 0 | 1.79 |

## Convergence and early stopping

Training and validation cross-entropy fell in every Python run. Logistic validation loss fell from 1.075–1.080 to 0.404–0.607. Neural validation loss fell from 1.093–1.094 to 0.286–0.511. All runs used the full 300 epochs and achieved their best validation loss at epoch 300. Recomputing the loss from the returned checkpoint matched each recorded minimum. In a controlled fixture with conflicting training and validation labels, loss improved on training and rose on validation. Both trainers used four epochs with patience set to three, then restored epoch one. The real-data runs show learning within the pilot budget. Their asymptotic convergence is still unknown.

## Leakage and interpretation

Same-interval classification can use metering and emissions readings when those readings have already arrived. Before-interval prediction at the tested horizon can use the calendar inputs. Including the target interval's electrical or usage data in a forecast would use measurements from the future relative to the forecast cutoff.

The [training-period audit](feature-audit.md) found substantial overlap in kWh across classes, a 0.9866 usage/CO₂ correlation, and three daily label profiles. Chronological validation contained 28 days of the most common training profile and nine all-Light days. It contained zero days of the third training profile and zero days with previously unseen profiles. This restricted profile mix limits what the chronological scores establish.

Calendar-only k-NN reached 91.89% on chronological validation at k=19 and 68.47% at both k=3 and k=11. In chronological training, 392 of 672 weekday/time-slot combinations occurred with multiple labels. Identical calendar vectors therefore have conflicting labels, and k-NN resolves equal distances by training row order. This result needs sensitivity checks across k, tie policies, and date windows before drawing conclusions about future performance.

Adding active usage and CO₂ to the electrical feature set produced mixed changes in this pilot. Electrical inputs improved both Python models over calendar alone on both splits. The results support calendar predictability and possible meter proxies, while the original label definition and measurement publication times remain unverified. A true workload forecast would require a defined lead time, inputs timestamped before that cutoff, and confirmation that the label represents the intended workload outcome.

## Computational cost

The whole pilot took **44.5 seconds** on an Intel Core i7-10750H with one BLAS thread and a CMake/MSVC Release build for k-NN. Across both splits and all feature sets, Python fits took 8.9 s for logistic and 25.4 s for the neural network. C++ validation queries took 9.4 s.

- At a 14,000-epoch cap and similar throughput, one logistic fit would take about **1.0–1.3 minutes**, and one 16-unit neural fit about **3.0–3.8 minutes**. Earlier patience stops would lower these times.
- For the current three-rate logistic grid across both splits and all feature sets, the same projection gives about **21 minutes**. The nine-candidate neural grid gives about **3 hours** if candidates cost the same as width 16, or **about 7 hours** if cost scales linearly with widths 16, 32, and 64. These scenarios are rough projections. Activation costs, wider matrix efficiency, and early stopping were unmeasured.
- k-NN computed about **86–87 million distances** for each validation configuration. If later authorized, roughly 172 million additional distances per configuration for final queries would put the combined six-configuration k-NN run on the order of **30 seconds** at the observed rate. This estimate uses partition sizes and costs from validation only. Heap size and fixed overhead affect scaling.

