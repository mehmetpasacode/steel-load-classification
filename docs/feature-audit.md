# Feature availability and leakage audit

The descriptive checks below use the **24,480 rows before September 13, 2018**, corresponding to the chronological training period. Run `python -m steel_load.audit` to reproduce them.

## Target and measurement proxies

| Label | Rows | Minimum kWh | Median kWh | Maximum kWh |
| --- | ---: | ---: | ---: | ---: |
| Light | 12,384 | 2.48 | 3.28 | 140.29 |
| Medium | 6,912 | 2.59 | 31.88 | 153.14 |
| Maximum | 5,184 | 3.02 | 58.14 | 149.65 |

The class ranges overlap extensively, ruling out a simple deterministic threshold rule on the recorded `Usage_kWh` alone. Pearson correlation between usage and CO₂ is **0.9866**. Grouping by exact usage values and picking each group's most frequent label describes **74.21%** of its own training rows. The equivalent figure is **66.12%** for CO₂, compared with an unconditional majority-class fraction of **50.59%**. These are in-sample descriptions. Exact-value grouping can benefit from rare values, and the figures do not establish forecast performance or how the target was generated.

Reactive energy and power factors are contemporaneous interval quantities and may also proxy active energy. `electrical` omits direct energy and CO₂ to support an ablation, while `calendar` omits all same-interval readings. The documented dataset variables do not establish the target-generation rule, sensor publication times, or whether CO₂ was independently measured. [UCI variable metadata](https://archive.ics.uci.edu/dataset/851/steel%2Bindustry%2Benergy%2Bconsumption)

## Recurring label profiles

All 255 training days have one of three ordered label sequences. Each entry below gives the number of consecutive 15-minute slots, starting at midnight.

| Training days | Runs in order |
| ---: | --- |
| 170 | Light 37, Medium 4, Maximum 8, Medium 4, Maximum 16, Medium 24, Light 3 |
| 46 | Light 37, Medium 4, Maximum 8, Medium 20, Maximum 12, Medium 8, Maximum 4, Light 3 |
| 39 | Light 96 |

This pattern suggests a schedule-based label definition or a process strongly aligned with schedules. Calendar inputs are available ahead of the interval, but performance may measure recovery of schedule categories. The original labeling procedure is needed to determine whether those categories correspond to independently measured production workload. A comparison against a calendar-only model under a fixed chronological split can measure the incremental value of electrical inputs if those inputs are available for the intended use.

