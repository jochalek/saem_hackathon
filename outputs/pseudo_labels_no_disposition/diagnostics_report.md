# Pseudo-Label Discovery Diagnostics (No Disposition / Leakage-Reduced)

- Input workbook: `/home/justin/projects/saem_hackathon/data/raw/Hackathon_Data_Release_1_SHARE.xlsx`
- RAPIDS enabled: `True`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` on `cuda`
- Modeling inputs: `Triage_Data` + restricted HPI text from `Four_Hour_Data`
- Excluded Four_Hour_Data columns (non-HPI): `35`
- Explicitly dropped leakage columns from modeling frame: `None`
- Expanded nested columns: `triage.labs`
- Stage-1 festival-like count: `183` / `261`
- Stage-1 threshold: `0.3700`

## Stability Summary
| metric                          |      value |
|:--------------------------------|-----------:|
| n_total                         | 261        |
| n_festival_like                 | 183        |
| n_no_festival                   |  78        |
| stage1_threshold                |   0.37     |
| stage1_anchor_balanced_accuracy |   0.991935 |
| stage2_pairwise_ari_mean        |   0.80097  |
| stage2_pairwise_ari_min         |   0.583057 |
| stage2_pairwise_ari_max         |   0.960954 |
| stage2_consensus_conf_mean      |   0.965027 |

## Text Leakage Scan
| column                               | pattern            |   count |
|:-------------------------------------|:-------------------|--------:|
| triage_chief_complaint               | mentions_discharge |       0 |
| triage_chief_complaint               | mentions_floor     |       0 |
| triage_chief_complaint               | mentions_icu       |       0 |
| triage_chief_complaint               | mentions_admit     |       0 |
| triage_brief_note                    | mentions_discharge |       0 |
| triage_brief_note                    | mentions_floor     |       1 |
| triage_brief_note                    | mentions_icu       |       0 |
| triage_brief_note                    | mentions_admit     |       0 |
| narrative_notes_structured_brief_hpi | mentions_discharge |       0 |
| narrative_notes_structured_brief_hpi | mentions_floor     |       0 |
| narrative_notes_structured_brief_hpi | mentions_icu       |       0 |
| narrative_notes_structured_brief_hpi | mentions_admit     |       0 |
| narrative_notes_structured_hpi       | mentions_discharge |       1 |
| narrative_notes_structured_hpi       | mentions_floor     |       1 |
| narrative_notes_structured_hpi       | mentions_icu       |       0 |
| narrative_notes_structured_hpi       | mentions_admit     |       0 |

## Disposition Balance (diagnostic only; not used for modeling)
| drug_target_pseudo   | encounter_disposition_label   |   count |   row_pct |
|:---------------------|:------------------------------|--------:|----------:|
| Drug_1               | Discharge                     |      58 | 0.568627  |
| Drug_1               | Floor                         |      27 | 0.264706  |
| Drug_1               | ICU                           |      17 | 0.166667  |
| Drug_2               | Discharge                     |      31 | 0.525424  |
| Drug_2               | Floor                         |      16 | 0.271186  |
| Drug_2               | ICU                           |      12 | 0.20339   |
| Drug_3               | Discharge                     |      11 | 0.5       |
| Drug_3               | Floor                         |       5 | 0.227273  |
| Drug_3               | ICU                           |       6 | 0.272727  |
| No_Festival          | Discharge                     |      71 | 0.910256  |
| No_Festival          | Floor                         |       4 | 0.0512821 |
| No_Festival          | ICU                           |       3 | 0.0384615 |