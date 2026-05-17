# Pseudo-labeled Brief HPI Examples

- Total pseudo-labeled rows: `261`
- Categories shown: `Drug_2, Drug_3, No_Party_Drug`
- Examples per category: `3`
- Sampling: `top`
- Seed: `42`

## Drug_2

Disposition counts in category:

| encounter_disposition_label   |   count |
|:------------------------------|--------:|
| Discharge                     |      11 |
| ICU                           |      10 |
| Floor                         |       6 |

### Example 1
- encounter_id: `E000286`
- drug_target_confidence: `0.7154`
- encounter_disposition_label: `ICU`
- vitals_shorthand: `HR 148 | RR 43 | BP 116/67 | SpO2 92% | Temp 39.4C | GCS 13 | SuppO2 0`
- narrative_notes_structured_brief_hpi: 52-year-old male arrived via ems from festival medical tent with shortness of breath and feeling unsteady, palpitations, chills, and exercise intolerance sensation approximately 128 minutes prior to arrival. No sustained hypoxia prior to ED arrival; No witnessed seizure activity.

### Example 2
- encounter_id: `E001170`
- drug_target_confidence: `0.6870`
- encounter_disposition_label: `ICU`
- vitals_shorthand: `HR 172 | RR 46 | BP 126/56 | SpO2 89% | Temp 39.2C | GCS 11 | SuppO2 1`
- narrative_notes_structured_brief_hpi: 55-year-old female arrived via ems from campground with nausea/vomiting and chills, marked restlessness, diffuse head discomfort, and shortness of breath approximately 235 minutes prior to arrival. No sustained hypoxia prior to ED arrival; No witnessed seizure activity.

### Example 3
- encounter_id: `E000130`
- drug_target_confidence: `0.6870`
- encounter_disposition_label: `ICU`
- vitals_shorthand: `HR 144 | RR 40 | BP 130/70 | SpO2 90% | Temp 39.0C | GCS 11 | SuppO2 1`
- narrative_notes_structured_brief_hpi: 14-year-old male arrived via festival medical tent transfer from festival medical tent with agitation and stomach upset, near-faint sensation, behavioral dysregulation, and diffuse head discomfort approximately 219 minutes prior to arrival. No sustained hypoxia prior to ED arrival; No witnessed seizure activity.

## Drug_3

Disposition counts in category:

| encounter_disposition_label   |   count |
|:------------------------------|--------:|
| Discharge                     |      27 |
| Floor                         |      15 |
| ICU                           |       2 |

### Example 1
- encounter_id: `E000267`
- drug_target_confidence: `0.5755`
- encounter_disposition_label: `Discharge`
- vitals_shorthand: `HR 104 | RR 20 | BP 135/88 | SpO2 97% | Temp 37.8C | GCS 14 | SuppO2 0`
- narrative_notes_structured_brief_hpi: 43-year-old non-binary arrived via walk-in from shopping area with agitation and air hunger sensation, diffuse head discomfort, lightheadedness, and dryness symptoms approximately 31 minutes prior to arrival. No major traumatic mechanism identified; No persistent unilateral neurologic deficit.

### Example 2
- encounter_id: `E001568`
- drug_target_confidence: `0.5755`
- encounter_disposition_label: `Discharge`
- vitals_shorthand: `HR 113 | RR 31 | BP 118/78 | SpO2 96% | Temp 38.0C | GCS 11 | SuppO2 1`
- narrative_notes_structured_brief_hpi: 61-year-old male arrived via ems from beach shuttle stop with chest pain and heart racing sensation, persistent disequilibrium, oral dryness, and chills approximately 194 minutes prior to arrival. Collateral context suggested symptoms began after possible festival-related substance exposure of unclear composition. No sustained hypoxia prior to ED arrival; No recurrent syncope observed in ED.

### Example 3
- encounter_id: `E001007`
- drug_target_confidence: `0.5755`
- encounter_disposition_label: `Floor`
- vitals_shorthand: `HR 110 | RR 31 | BP 117/59 | SpO2 95% | Temp 37.7C | GCS 12 | SuppO2 0`
- narrative_notes_structured_brief_hpi: 40-year-old male arrived via walk-in from main stage with headache and non-radiating chest pressure, palpitations, chills, and feeling unsteady approximately 67 minutes prior to arrival. No recurrent syncope observed in ED; No major traumatic mechanism identified.

## No_Party_Drug

Disposition counts in category:

| encounter_disposition_label   |   count |
|:------------------------------|--------:|
| Discharge                     |     133 |
| Floor                         |      31 |
| ICU                           |      26 |

### Example 1
- encounter_id: `E000880`
- drug_target_confidence: `0.4884`
- encounter_disposition_label: `Discharge`
- vitals_shorthand: `HR 79 | RR 18 | BP 118/85 | SpO2 97% | Temp 37.2C | GCS 15 | SuppO2 0`
- narrative_notes_structured_brief_hpi: 54-year-old non-binary arrived via ems from street with nausea/vomiting and awareness of rapid heartbeat, behavioral dysregulation, oral dryness, and ringing in ears approximately 192 minutes prior to arrival. No recurrent syncope observed in ED; No persistent unilateral neurologic deficit.

### Example 2
- encounter_id: `E000424`
- drug_target_confidence: `0.4456`
- encounter_disposition_label: `Floor`
- vitals_shorthand: `HR 125 | RR 23 | BP 135/74 | SpO2 93% | Temp 38.3C | GCS 9 | SuppO2 1`
- narrative_notes_structured_brief_hpi: 71-year-old female arrived via police from campground with altered mental status and nausea, shortness of breath, head pressure, and heat-stress symptoms approximately 193 minutes prior to arrival. No major traumatic mechanism identified; No sustained hypoxia prior to ED arrival.

### Example 3
- encounter_id: `E000818`
- drug_target_confidence: `0.4456`
- encounter_disposition_label: `Discharge`
- vitals_shorthand: `HR 99 | RR 18 | BP 117/80 | SpO2 97% | Temp 37.1C | GCS 13 | SuppO2 0`
- narrative_notes_structured_brief_hpi: 31-year-old female arrived via walk-in from campground with agitation and dry mouth, air hunger sensation, head pressure, and psychomotor slowing approximately 127 minutes prior to arrival. No persistent unilateral neurologic deficit; No recurrent syncope observed in ED.
