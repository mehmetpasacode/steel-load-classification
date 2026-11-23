# Historical course results

These figures come from the original EEE-485 project. They belong to its experimental protocol and carry no estimate of performance under the new held-out evaluation.

| Model | Historical accuracy |
| --- | ---: |
| Neural network | 89.07% |
| Logistic regression | 81.12% |
| k-nearest neighbors | Approximately 86% |

[The archived neural-network sweep](nn_27_experiment_results.csv) is copied verbatim from the original project. It contains per-candidate test scores, which exposed the test set during experimentation. The neural-network script encoded target columns in alphabetical order and manually labeled the chart in load-severity order, swapping the Medium and Maximum labels on both axes.

The [original normalizer](cpp/normalize.cpp) and [original classifier](cpp/knn.cpp) were recovered from Appendix B, pages 17–29 of the course report in the sibling `Project` directory. Extraction removed page boundaries and adjusted wrapping and indentation. The old normalizer learned feature ranges and categories across all rows. Its one-hot buffer also left some entries uninitialized. The old classifier swept k on the selected test rows and could pick other test rows as neighbors. It excluded every zero-distance match, including distinct rows with identical features.

The active C++ source and shared preprocessing live in `../cpp` and `../steel_load`. The historical programs expect their original text inputs and belong to the archive.

