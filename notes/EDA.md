### Environment/setup used

 - Ran:
     - mise use python@3.12
     - mise use uv
 - Created a scripts project with uv and dependencies, then ran with:
     - cd scripts && uv sync
     - cd scripts && uv run python eda_hackathon_data.py

 ### Files created

 - scripts/pyproject.toml
 - scripts/uv.lock
 - scripts/eda_hackathon_data.py

 ### EDA outputs

 All outputs are in:
 - outputs/eda/

 Key files:
 - outputs/eda/eda_report.md
 - outputs/eda/sheet_overview.csv
 - outputs/eda/disposition_distribution.csv
 - outputs/eda/clinical_course_leakage_scan.csv
 - outputs/eda/sex_hpi_mismatch_summary.csv
 - plus per-sheet missingness/dtypes/numeric/categorical summaries

 ### Main findings

 - 3 sheets, all with 261 encounters, perfectly joinable on encounter_id.
 - Disposition class balance:
     - Discharge: 171 (65.5%)
     - Floor: 52 (19.9%)
     - ICU: 38 (14.6%)
 - Major missingness in 4-hour reassessment labs:
     - lactate/cpk/vbg_ph/troponin each missing ~49%.
 - Strong leakage risk confirmed:
     - narrative_notes_structured_clinical_course contains final plan/disposition language (counts align with disposition labels), so should be excluded for
 hackathon-compatible modeling.
 - Minor documentation inconsistency detected:
     - 3 rows where triage sex/gender vs HPI text gender mismatch.
 - Important: I did not find an explicit drug-label column in any worksheet (or codebook variable list).
