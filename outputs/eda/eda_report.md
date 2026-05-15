# EDA Report: Hackathon Data Release 1

- Workbook: `/home/justin/projects/saem_hackathon/data/raw/Hackathon_Data_Release_1_SHARE.xlsx`
- Codebook: `/home/justin/projects/saem_hackathon/data/raw/Hackathon_Codebook_Release_1_SHARE.docx`

## Sheet Overview
| sheet          |   n_rows |   n_cols |   numeric_cols |   categorical_cols | id_candidates                                        |
|:---------------|---------:|---------:|---------------:|-------------------:|:-----------------------------------------------------|
| Triage_Data    |      261 |       24 |             16 |                  8 | encounter_id, encounter_arrival_date                 |
| Four_Hour_Data |      261 |       38 |             27 |                 11 | encounter_id, ed_course.reassessment_4h.encounter_id |
| Disposition    |      261 |        2 |              0 |                  2 | encounter_id, encounter_disposition_label            |

## Disposition Label Distribution
| label     |   count |      pct |
|:----------|--------:|---------:|
| Discharge |     171 | 0.655172 |
| Floor     |      52 | 0.199234 |
| ICU       |      38 | 0.145594 |

## Notes
- Use `Triage_Data` only for drug-identification model features.
- Use `Four_Hour_Data` (+ triage data) for disposition model features.
- Exclude `narrative_notes_structured_clinical_course` in final hackathon-compatible model.

## ID Overlap Checks
| sheet_a        | sheet_b        | id_col       |   unique_a |   unique_b |   intersection |   coverage_of_a |   coverage_of_b |
|:---------------|:---------------|:-------------|-----------:|-----------:|---------------:|----------------:|----------------:|
| Triage_Data    | Four_Hour_Data | encounter_id |        261 |        261 |            261 |               1 |               1 |
| Triage_Data    | Disposition    | encounter_id |        261 |        261 |            261 |               1 |               1 |
| Four_Hour_Data | Disposition    | encounter_id |        261 |        261 |            261 |               1 |               1 |

## Nested Column Length Checks
### nested_triage_labs_len
|   len |   count |
|------:|--------:|
|     1 |     261 |

### nested_ed_course_vitals_timeseries_len
|   len |   count |
|------:|--------:|
|     6 |      69 |
|     7 |      59 |
|     8 |      74 |
|     9 |      59 |

### nested_ed_course_labs_timeseries_len
|   len |   count |
|------:|--------:|
|     0 |      47 |
|     1 |      84 |
|     2 |      74 |
|     3 |      56 |

### nested_ed_course_interventions_len
|   len |   count |
|------:|--------:|
|     0 |      29 |
|     1 |      87 |
|     2 |      75 |
|     3 |      53 |
|     4 |      11 |
|     5 |       6 |

### intervention_event_counts
| event_name                 |   count |
|:---------------------------|--------:|
| IVF                        |     151 |
| Medication-Benzodiazepine  |     133 |
| Medication-Antipyretic     |      81 |
| Medication-Reversal        |      46 |
| Procedure-PositivePressure |      25 |
| Procedure-Intubation       |      22 |
| Procedure-CVC              |      12 |

## Potential Label Leakage in `narrative_notes_structured_clinical_course`
| pattern             |   count |
|:--------------------|--------:|
| mentions_discharge  |     171 |
| mentions_floor      |      52 |
| mentions_icu        |      39 |
| mentions_final_plan |     261 |

## Sex vs HPI Text Consistency
| triage_sex_gender   | hpi_gender   | sex_matches_hpi   |   count |
|:--------------------|:-------------|:------------------|--------:|
| Female              | Female       | True              |     114 |
| Female              | Male         | False             |       3 |
| Male                | Male         | True              |     138 |
| Non-binary          | Non-binary   | True              |       6 |

## Top Missingness (First 10 Columns Per Sheet)
### Triage_Data
| column                            |   missing_count |   missing_pct |
|:----------------------------------|----------------:|--------------:|
| encounter_id                      |               0 |             0 |
| encounter_arrival_date            |               0 |             0 |
| triage_heart_rate                 |               0 |             0 |
| triage_respiratory_rate           |               0 |             0 |
| triage_snapshot.systolic_bp       |               0 |             0 |
| triage_snapshot.diastolic_bp      |               0 |             0 |
| triage_snapshot.oxygen_saturation |               0 |             0 |
| triage_supplemental_oxygen        |               0 |             0 |
| triage_temperature_c              |               0 |             0 |
| triage_gcs                        |               0 |             0 |

### Four_Hour_Data
| column                                  |   missing_count |   missing_pct |
|:----------------------------------------|----------------:|--------------:|
| ed_course_reassessment_4h.lactate_4h    |             128 |      0.490421 |
| ed_course_reassessment_4h.cpk_4h        |             128 |      0.490421 |
| ed_course_reassessment_4h.vbg_ph_4h     |             128 |      0.490421 |
| ed_course_reassessment_4h.troponin_4h   |             128 |      0.490421 |
| encounter_id                            |               0 |      0        |
| ed_course.vitals_timeseries             |               0 |      0        |
| ed_course.labs_timeseries               |               0 |      0        |
| ed_course.interventions                 |               0 |      0        |
| ed_course.reassessment_4h.encounter_id  |               0 |      0        |
| ed_course_reassessment_4h.heart_rate_4h |               0 |      0        |

### Disposition
| column                      |   missing_count |   missing_pct |
|:----------------------------|----------------:|--------------:|
| encounter_id                |               0 |             0 |
| encounter_disposition_label |               0 |             0 |

## Codebook Variable Rows (Heuristic Extract)
| variable                                            | description                                                                                                                                                                                                                    |
|:----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| encounter_id                                        | Unique patient encounter identifier                                                                                                                                                                                            |
| encounter_disposition_label                         | Simplified discharge disposition                                                                                                                                                                                               |
| encounter_arrival_date                              | Date patient arrived at ED                                                                                                                                                                                                     |
| triage_heart_rate                                   | Heart rate at triage (bpm)                                                                                                                                                                                                     |
| triage_respiratory_rate                             | Respiratory rate at triage (breaths/min)                                                                                                                                                                                       |
| triage_snapshot.systolic_bp                         | Systolic blood pressure at triage (mmHg)                                                                                                                                                                                       |
| triage_snapshot.diastolic_bp                        | Diastolic blood pressure at triage (mmHg)                                                                                                                                                                                      |
| triage_snapshot.oxygen_saturation                   | Oxygen saturation at triage (%)                                                                                                                                                                                                |
| triage_supplemental_oxygen                          | Supplemental oxygen given at triage                                                                                                                                                                                            |
| triage_temperature_c                                | Body temperature at triage (°C)                                                                                                                                                                                                |
| triage_gcs                                          | Glasgow Coma Scale score at triage                                                                                                                                                                                             |
| triage_age                                          | Patient age at encounter (years)                                                                                                                                                                                               |
| triage_sex_gender                                   | Patient self-reported sex/gender                                                                                                                                                                                               |
| triage_race_ethnicity                               | Patient self-reported race/ethnicity                                                                                                                                                                                           |
| triage_chief_complaint                              | Primary presenting complaint at triage                                                                                                                                                                                         |
| triage_brief_note                                   | Free-text brief clinical note by triage clinician                                                                                                                                                                              |
| triage_esi                                          | Emergency Severity Index (1 = most acute, 5 = least acute)                                                                                                                                                                     |
| triage_pain_scale                                   | Self-reported pain score at triage                                                                                                                                                                                             |
| triage_mode_of_arrival                              | How the patient arrived at the ED                                                                                                                                                                                              |
| triage_mh_psych                                     | History of psychiatric condition                                                                                                                                                                                               |
| triage_mh_cardiac                                   | History of cardiac condition                                                                                                                                                                                                   |
| triage_mh_pulm                                      | History of pulmonary condition                                                                                                                                                                                                 |
| triage_mh_renal                                     | History of renal condition                                                                                                                                                                                                     |
| triage_mh_substance_use                             | History of substance use disorder                                                                                                                                                                                              |
| triage.labs                                         | Point-of-care iStat labs at triage. Fields: encounter_id, minute, fingerstick_glucose (mg/dL), ph, sodium (mEq/L), potassium (mEq/L), hemoglobin (g/dL), anion_gap                                                             |
| ed_course.vitals_timeseries                         | Serial vital signs during ED course. Fields: minute, heart_rate, respiratory_rate, systolic_bp, diastolic_bp, oxygen_saturation, supplemental_oxygen, temperature_c, gcs, end_tidal_co2                                        |
| ed_course.labs_timeseries                           | Serial lab results during ED course. Fields: minute, cbc_wbc, bmp_sodium, bmp_potassium, bmp_bicarb, lft_ast, vbg_ph, hcg_positive, ua_abnormal, troponin, lactate, cpk, esr, crp, poct_glucose, serum_tox_positive, serum_osm |
| ed_course.interventions                             | Timestamped interventions during ED course. Fields: minute, event_name                                                                                                                                                         |
| ed_course.reassessment_4h.encounter_id              | Encounter ID repeated in 4-hour reassessment block (join verification)                                                                                                                                                         |
| ed_course_reassessment_4h.heart_rate_4h             | Heart rate at 4-hour reassessment (bpm)                                                                                                                                                                                        |
| ed_course_reassessment_4h.respiratory_rate_4h       | Respiratory rate at 4-hour reassessment (breaths/min)                                                                                                                                                                          |
| ed_course_reassessment_4h.systolic_bp_4h            | Systolic BP at 4-hour reassessment (mmHg)                                                                                                                                                                                      |
| ed_course_reassessment_4h.diastolic_bp_4h           | Diastolic BP at 4-hour reassessment (mmHg)                                                                                                                                                                                     |
| ed_course_reassessment_4h.oxygen_saturation_4h      | Oxygen saturation at 4-hour reassessment (%)                                                                                                                                                                                   |
| ed_course_reassessment_4h.supplemental_oxygen_4h    | Supplemental oxygen in use at 4-hour reassessment                                                                                                                                                                              |
| ed_course_reassessment_4h.temperature_c_4h          | Temperature at 4-hour reassessment (°C)                                                                                                                                                                                        |
| ed_course_reassessment_4h.gcs_4h                    | Glasgow Coma Scale at 4-hour reassessment                                                                                                                                                                                      |
| ed_course_reassessment_4h.end_tidal_co2_4h          | End-tidal CO2 at 4-hour reassessment (mmHg)                                                                                                                                                                                    |
| ed_course_reassessment_4h.delta_hr                  | Change in heart rate from triage to 4h (bpm)                                                                                                                                                                                   |
| ed_course_reassessment_4h.delta_temp                | Change in temperature from triage to 4h (°C)                                                                                                                                                                                   |
| ed_course_reassessment_4h.delta_gcs                 | Change in GCS from triage to 4h                                                                                                                                                                                                |
| ed_course_reassessment_4h.lactate_4h                | Serum lactate at 4-hour reassessment (mmol/L)                                                                                                                                                                                  |
| ed_course_reassessment_4h.cpk_4h                    | Creatine phosphokinase at 4-hour reassessment (U/L)                                                                                                                                                                            |
| ed_course_reassessment_4h.vbg_ph_4h                 | Venous blood gas pH at 4-hour reassessment                                                                                                                                                                                     |
| ed_course_reassessment_4h.troponin_4h               | Troponin at 4-hour reassessment (ng/mL)                                                                                                                                                                                        |
| ed_course_reassessment_4h.ivf_count_0_4h            | IV fluid administration within first 4 hours                                                                                                                                                                                   |
| ed_course_reassessment_4h.benzodiazepine_count_0_4h | Benzodiazepine administered within first 4 hours                                                                                                                                                                               |
| ed_course_reassessment_4h.reversal_count_0_4h       | Reversal agent administered within first 4 hours                                                                                                                                                                               |
| ed_course_reassessment_4h.intubated_0_4h            | Patient intubated within first 4 hours                                                                                                                                                                                         |
| ed_course_reassessment_4h.positive_pressure_0_4h    | Non-invasive positive pressure ventilation within first 4 hours                                                                                                                                                                |
