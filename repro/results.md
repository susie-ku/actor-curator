# Table-1 reproduction — results snapshot (2026-11-26)

**Incomplete.** 13 of 16 (benchmark, method) cells have finished 100 steps. Still running / queued on 2026-11-26:
ARC-1D SEC (job 96519, running), ARC-1D PCL (96520, queued), greedy re-evaluation of the four MATH step-100
checkpoints (93902, queued). This file will be regenerated when they finish.

Setup: Qwen2.5-3B actor, Qwen3-0.6B curator, 100 steps, validation every 10 steps (T=1.0, top-p 0.7, n=1),
one RTX PRO 6000 (96 GB) per run, seed 1. Numbers are the **peak** validation accuracy (%) within 100 steps.
`mean4` = mean over the four difficulty levels; the paper's "-hard" column is compared with `-very_hard-test`
(see PROBLEMS.md B8). Paper = arXiv:2602.20532 Table 1. Decisions and deviations: PROBLEMS.md.
Regenerate the table with `python repro/collect_results.py` (TensorBoard logs in tensorboard_log/actor_curator_repro, not committed).

## Peak accuracy vs. paper

| benchmark / data_source | Uniform | SEC | PCL | AC | paper (Uniform / SEC / PCL / AC) |
|---|---|---|---|---|---|
| arc / arc-easy-test | 57.00 | 12.50 (≤40) | — | 57.50 | — |
| arc / arc-hard-test | 29.00 | 3.00 (≤40) | — | 31.50 | — |
| arc / arc-medium-test | 38.00 | 3.50 (≤40) | — | 35.00 | — |
| arc / arc-very_hard-test | 23.50 | 1.50 (≤40) | — | 25.00 | 19.50 / 18.50 / 18.50 / 31.00 |
| arc / mean4 | 36.87 | 4.87 (≤40) | — | 37.25 | 26.74 / 27.87 / 26.37 / 36.37 |
| countdown / countdown-easy-test | 75.00 | 52.50 | 62.50 | 63.00 | — |
| countdown / countdown-hard-test | 22.00 | 21.50 | 19.00 | 20.50 | — |
| countdown / countdown-medium-test | 40.00 | 26.50 | 33.00 | 34.50 | — |
| countdown / countdown-very_hard-test | 10.50 | 8.50 | 10.00 | 15.00 | 41.00 / 51.50 / 48.00 / 58.00 |
| countdown / mean4 | 36.88 | 26.75 | 30.13 | 32.25 | 44.74 / 58.87 / 57.24 / 62.12 |
| math / test_aime | 10.00 | 10.00 | 6.67 | 16.67 | 23.33 / 20.00 / 23.33 / 30.00 |
| math / test_amc | 33.73 | 37.35 | 37.35 | 38.55 | — |
| math / test_math | 64.20 | 64.40 | 65.40 | 64.20 | 83.00 / 81.00 / 79.79 / 81.00 |
| zebra / mean4 | 44.50 | 43.62 | 43.63 | 45.00 | 35.12 / 36.00 / 34.12 / 37.62 |
| zebra / zebra-easy-test | 69.00 | 65.50 | 67.00 | 68.50 | — |
| zebra / zebra-hard-test | 37.50 | 40.50 | 40.00 | 38.00 | — |
| zebra / zebra-medium-test | 45.00 | 48.50 | 49.50 | 48.50 | — |
| zebra / zebra-very_hard-test | 29.00 | 26.50 | 24.00 | 26.00 | 30.50 / 27.50 / 26.00 / 34.50 |

Cells marked `(≤40)` are from a run that was still in progress at step 40.

## Untrained model (greedy decoding, job 96517)

| data_source | greedy | sampled (T=1.0, top-p 0.7, step 0 of the training runs) | paper π_ref |
|---|---|---|---|
| MATH500 | 26.8 | 16.8 | 61.8 |
| AMC | 15.7 | 9.6 | — |
| AIME24 | 0.0 | 0.0 | — |

## Per-run detail (peak, step of peak, last evaluated step, step-0 accuracy)

```
experiment                               data_source              peak% @step last_eval  step0%
arc_ac_qwen2.5-3b_s1                     arc-easy-test            57.50   100       100    2.50
arc_ac_qwen2.5-3b_s1                     arc-medium-test          35.00   100       100    1.50
arc_ac_qwen2.5-3b_s1                     arc-hard-test            31.50   100       100    0.50
arc_ac_qwen2.5-3b_s1                     arc-very_hard-test       25.00   100       100    1.00
arc_sec_qwen2.5-3b_s1                    arc-easy-test            12.50    40        40    2.50
arc_sec_qwen2.5-3b_s1                    arc-medium-test           3.50    30        40    1.50
arc_sec_qwen2.5-3b_s1                    arc-hard-test             3.00    40        40    0.50
arc_sec_qwen2.5-3b_s1                    arc-very_hard-test        1.50    20        40    1.00
arc_uniform_qwen2.5-3b_s1                arc-easy-test            57.00   100       100    2.50
arc_uniform_qwen2.5-3b_s1                arc-medium-test          38.00   100       100    1.50
arc_uniform_qwen2.5-3b_s1                arc-hard-test            29.00   100       100    0.50
arc_uniform_qwen2.5-3b_s1                arc-very_hard-test       23.50   100       100    1.00
countdown_ac_qwen2.5-3b_s1               countdown-easy-test      63.00   100       100    2.00
countdown_ac_qwen2.5-3b_s1               countdown-medium-test    34.50    90       100    1.50
countdown_ac_qwen2.5-3b_s1               countdown-hard-test      20.50    90       100    0.00
countdown_ac_qwen2.5-3b_s1               countdown-very_hard-test   15.00    90       100    0.00
countdown_pcl_qwen2.5-3b_s1              countdown-easy-test      62.50   100       100    2.00
countdown_pcl_qwen2.5-3b_s1              countdown-medium-test    33.00    70       100    1.50
countdown_pcl_qwen2.5-3b_s1              countdown-hard-test      19.00    90       100    0.00
countdown_pcl_qwen2.5-3b_s1              countdown-very_hard-test   10.00    90       100    0.00
countdown_sec_qwen2.5-3b_s1              countdown-easy-test      52.50   100       100    2.00
countdown_sec_qwen2.5-3b_s1              countdown-medium-test    26.50   100       100    1.50
countdown_sec_qwen2.5-3b_s1              countdown-hard-test      21.50    90       100    0.00
countdown_sec_qwen2.5-3b_s1              countdown-very_hard-test    8.50   100       100    0.00
countdown_uniform_qwen2.5-3b_s1          countdown-easy-test      75.00   100       100    2.00
countdown_uniform_qwen2.5-3b_s1          countdown-medium-test    40.00   100       100    1.50
countdown_uniform_qwen2.5-3b_s1          countdown-hard-test      22.00   100       100    0.00
countdown_uniform_qwen2.5-3b_s1          countdown-very_hard-test   10.50   100       100    0.00
math_ac_qwen2.5-3b_s1                    test_aime                16.67    90       100    0.00
math_ac_qwen2.5-3b_s1                    test_amc                 38.55    80       100    9.64
math_ac_qwen2.5-3b_s1                    test_math                64.20   100       100   16.80
math_pcl_qwen2.5-3b_s1                   test_aime                 6.67    30       100    0.00
math_pcl_qwen2.5-3b_s1                   test_amc                 37.35   100       100    9.64
math_pcl_qwen2.5-3b_s1                   test_math                65.40   100       100   16.80
math_sec_qwen2.5-3b_s1                   test_aime                10.00    40       100    0.00
math_sec_qwen2.5-3b_s1                   test_amc                 37.35    60       100    9.64
math_sec_qwen2.5-3b_s1                   test_math                64.40    60       100   16.80
math_uniform_qwen2.5-3b_s1               test_aime                10.00    70       100    0.00
math_uniform_qwen2.5-3b_s1               test_amc                 33.73    70       100    9.64
math_uniform_qwen2.5-3b_s1               test_math                64.20   100       100   16.80
zebra_ac_qwen2.5-3b_s1                   zebra-easy-test          68.50   100       100    9.00
zebra_ac_qwen2.5-3b_s1                   zebra-medium-test        48.50   100       100   10.50
zebra_ac_qwen2.5-3b_s1                   zebra-hard-test          38.00    70       100    7.50
zebra_ac_qwen2.5-3b_s1                   zebra-very_hard-test     26.00    90       100    5.00
zebra_pcl_qwen2.5-3b_s1                  zebra-easy-test          67.00   100       100    9.00
zebra_pcl_qwen2.5-3b_s1                  zebra-medium-test        49.50   100       100   10.50
zebra_pcl_qwen2.5-3b_s1                  zebra-hard-test          40.00    60       100    7.50
zebra_pcl_qwen2.5-3b_s1                  zebra-very_hard-test     24.00    60       100    5.00
zebra_sec_qwen2.5-3b_s1                  zebra-easy-test          65.50   100       100    9.00
zebra_sec_qwen2.5-3b_s1                  zebra-medium-test        48.50    60       100   10.50
zebra_sec_qwen2.5-3b_s1                  zebra-hard-test          40.50    90       100    7.50
zebra_sec_qwen2.5-3b_s1                  zebra-very_hard-test     26.50    40       100    5.00
zebra_uniform_qwen2.5-3b_s1              zebra-easy-test          69.00   100       100    9.00
zebra_uniform_qwen2.5-3b_s1              zebra-medium-test        45.00    60       100   10.50
zebra_uniform_qwen2.5-3b_s1              zebra-hard-test          37.50    80       100    7.50
zebra_uniform_qwen2.5-3b_s1              zebra-very_hard-test     29.00   100       100    5.00
```
