# Pseudo-Label Discovery Diagnostics

- Input workbook: `/home/justin/projects/saem_hackathon/data/raw/Hackathon_Data_Release_1_SHARE.xlsx`
- RAPIDS enabled: `True`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` on `cuda`
- Expanded nested columns: triage.labs, ed_course.vitals_timeseries, ed_course.labs_timeseries, ed_course.interventions
- Stage-1 festival-like count: `183` / `261`
- Stage-1 threshold: `0.4000`

## Stability Summary
| metric                          |      value |
|:--------------------------------|-----------:|
| n_total                         | 261        |
| n_festival_like                 | 183        |
| n_no_festival                   |  78        |
| stage1_threshold                |   0.4      |
| stage1_anchor_balanced_accuracy |   0.991935 |
| stage2_pairwise_ari_mean        |   0.556992 |
| stage2_pairwise_ari_min         |   0.307283 |
| stage2_pairwise_ari_max         |   1        |
| stage2_consensus_conf_mean      |   0.864481 |

## Leakage / Sensitivity Checks
| check                                           |       value |
|:------------------------------------------------|------------:|
| stage1_label_flip_rate_with_clinical_course     | 0           |
| stage1_mean_abs_prob_shift_with_clinical_course | 5.18853e-07 |
| stage2_flip_rate_on_baseline_festival_subset    | 0           |
| stage2_ari_on_baseline_festival_subset          | 1           |

## Disposition Balance
| drug_target_pseudo   | encounter_disposition_label   |   count |   row_pct |
|:---------------------|:------------------------------|--------:|----------:|
| Drug_1               | Discharge                     |      91 | 0.928571  |
| Drug_1               | Floor                         |       7 | 0.0714286 |
| Drug_1               | ICU                           |       0 | 0         |
| Drug_2               | Discharge                     |       8 | 0.148148  |
| Drug_2               | Floor                         |      40 | 0.740741  |
| Drug_2               | ICU                           |       6 | 0.111111  |
| Drug_3               | Discharge                     |       1 | 0.0322581 |
| Drug_3               | Floor                         |       1 | 0.0322581 |
| Drug_3               | ICU                           |      29 | 0.935484  |
| No_Festival          | Discharge                     |      71 | 0.910256  |
| No_Festival          | Floor                         |       4 | 0.0512821 |
| No_Festival          | ICU                           |       3 | 0.0384615 |