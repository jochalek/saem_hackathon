# Symptom Constellation Pseudo-Label Diagnostics

- Input workbook: `/home/justin/projects/saem_hackathon/data/raw/Hackathon_Data_Release_1_SHARE.xlsx`
- Model: `emilyalsentzer/Bio_ClinicalBERT` on `cuda`
- Rows: `261`

## Stability Summary
| metric                    |      value |
|:--------------------------|-----------:|
| n_total                   | 261        |
| n_no_party_drug           | 190        |
| n_drug_1                  |   0        |
| n_drug_2                  |  27        |
| n_drug_3                  |  44        |
| mean_confidence           |   0.28453  |
| mean_symptom_evidence     |   1.32184  |
| mean_prototype_similarity |   0.293171 |
| n_residual_clustered      |  51        |

## Disposition Balance
| drug_target_pseudo   | encounter_disposition_label   |   count |   row_pct |
|:---------------------|:------------------------------|--------:|----------:|
| Drug_2               | Discharge                     |      11 | 0.407407  |
| Drug_2               | Floor                         |       6 | 0.222222  |
| Drug_2               | ICU                           |      10 | 0.37037   |
| Drug_3               | Discharge                     |      27 | 0.613636  |
| Drug_3               | Floor                         |      15 | 0.340909  |
| Drug_3               | ICU                           |       2 | 0.0454545 |
| No_Party_Drug        | Discharge                     |     133 | 0.7       |
| No_Party_Drug        | Floor                         |      31 | 0.163158  |
| No_Party_Drug        | ICU                           |      26 | 0.136842  |