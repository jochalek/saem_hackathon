# LLM Structured Extraction Pseudo-Label Diagnostics

- Input workbook: `/home/justin/projects/saem_hackathon/data/raw/Hackathon_Data_Release_1_SHARE.xlsx`
- OpenAI model: `gpt-5.5`
- Rows processed: `3`
- Extraction errors: `0`
- Clusters requested: `4`
- Clusters used: `3`

## Stability Summary
| metric                  | value   |
|:------------------------|:--------|
| n_total                 | 3       |
| n_extraction_errors     | 0       |
| mean_cluster_confidence | 1.0     |
| n_clusters_requested    | 4       |
| n_clusters_used         | 3       |
| openai_model            | gpt-5.5 |

## Sample extracted named entities

- encounter_id=E000143
  symptoms=['abdominal pain', 'anxiety/panic', 'back pain', 'chest pain', 'chills', 'diarrhea', 'dryness symptoms', 'feeling unsteady', 'fever', 'joint swelling', 'leg swelling', 'self-injury', 'shortness of breath', 'suicidal thoughts', 'sustained hypoxia prior to ed arrival', 'time-distortion sensation', 'weakness', 'witnessed seizure activity']
  time_prior_to_arrival_minutes=268
  context=arrival_mode=walk_in, origin=offsite
- encounter_id=E001622
  symptoms=['awareness of rapid heartbeat', 'generalized tiredness', 'nausea/vomiting', 'persistent unilateral neurologic deficit', 'restlessness', 'ringing in ears', 'sense of internal unease']
  time_prior_to_arrival_minutes=107
  context=arrival_mode=festival_tent_transfer, origin=medical_tent