# Extended neural-network encoding ablation

This run compares the original 15-input encoding A with the 12-input encoding C on combined calendar and electrical features. A includes seven weekday indicators and two WeekStatus indicators. C removes both WeekStatus columns and uses Monday as the reference day. Electrical readings and cyclic time features are the same. The train-fitted preprocessor supplies both matrices, with C selected from the original columns after fitting. The original k-NN encoding is untouched.

The original random stratified and chronological partitions were fixed. The random splitter used target labels only to allocate its stratified partitions. Neither holdout entered training, checkpoint selection, or evaluation. The model fits restart from initialization seeds 42, 7, and 101. The paired seeds use the same initialization procedure, and the two input widths can assign random draws differently to weights.

Both arms used ReLU, learning rate 0.1, momentum 0.9, unweighted cross-entropy, and patience 30. Hidden width was 32 for the random split and 16 for the chronological split, matching the prior freeze. The sole training-budget change was the 2,000-epoch cap. No hyperparameters were searched.

Reproduce from the repository root using a fresh output path.

```powershell
python -m steel_load.encoding_experiment --extended --output docs/encoding-extended-repeat.json
```

The [raw results](encoding-extended.json) include every loss trace and validation metric. The [loss curves](encoding-extended-loss.png) show individual seeds as faint lines and the median through the last common epoch as a solid line. The dotted vertical marker is the original 600-epoch cap.

## Results

Accuracy, macro F1, and minimum validation cross-entropy are means across the three seeds. Fit time is the median, including per-epoch validation. Best epochs are listed in seed order 42 / 7 / 101.

| Split | Encoding | Accuracy | Macro F1 | Minimum loss | Best epochs | Stops on patience | Fit time |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Random stratified | A | 87.16% | 0.8386 | 0.31105 | 2000 / 2000 / 2000 | 0/3 | 44.68 s |
| Random stratified | C | 87.12% | 0.8374 | 0.31039 | 2000 / 2000 / 2000 | 0/3 | 41.20 s |
| Chronological | A | 94.07% | 0.9318 | 0.19679 | 1570 / 1534 / 1924 | 3/3 | 21.61 s |
| Chronological | C | 94.12% | 0.9326 | 0.20258 | 2000 / 2000 / 2000 | 0/3 | 25.39 s |

All 12 extended runs reproduced their previously recorded validation loss curves through epoch 600. At that cap, C led A by 0.15 macro-F1 percentage points on random validation and trailed A by 1.05 points on chronological validation. After longer training, C trailed by 0.12 points on random validation and led by 0.08 points on chronological validation. The earlier chronological gap closed, and the ranking reversed on both splits. C won one of the three paired seeds on each split. The paired C-minus-A macro-F1 differences ranged from -0.33 to +0.03 points on random and from -1.01 to +1.29 points on chronological.

Chronological A met the operational early-stopping criterion on all seeds. Both random arms and chronological C were still setting validation loss minima at epoch 2,000. In the final 200 epochs, new minima improved by 0.00084-0.00252 for random A, 0.00145-0.00242 for random C, and 0.00208-0.00382 for chronological C. These runs showed continued progress within the budget. Patience stops establish a validation checkpoint under the chosen rule, and exact optimizer convergence remains unverified.

## Frozen encoding

**Freeze C for this combined-input neural ablation**, using the higher mean chronological validation macro F1 as the decision rule. Its lead is small relative to the paired seed variation. A had lower chronological validation loss, stronger seed-to-seed consistency, and a shorter chronological median fit. The result supports keeping C as a full-rank, smaller-input configuration under the stated metric, with no reliable performance gain established.

The [frozen configuration](encoding-extended-frozen.json) records the dataset and result hashes, ordered feature columns, split seed, optimizer settings, and seed-42 checkpoint for each split. Seed 42 remains the reference seed rather than selecting the best-performing seed. Both reference checkpoints are at epoch 2,000. Production encoding remains unchanged.

The 12 fits took 6 minutes 44 seconds on one BLAS thread. All 12 existing correctness tests passed. The final test partitions and historical results were left untouched.
