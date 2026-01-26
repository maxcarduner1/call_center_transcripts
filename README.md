# Call Center Analytics Demos

## What is included

### Notebooks  

**Make sure to edit the `catalog` and `schema` variable names in the .env folder**
  
**01_next_best_action: Identify Engagement Propensity & Next Best Action**
- `call_center_demo_setup.py`: set up the initial datasets for the demo
- `call_center_ml_model.py`: leverage XGBoost and LightGBM models on the structured data to build models that help identify cohorts of engagement and what the next best action for the type of cohort
- `ml_concepts_explained.md`: explanation of why Classic ML models perform better for this

**02_transcriptions_generation: Generate transcriptions for a sample of calls, score it, for reverse-etl into Lakebase**
- transcriptions

**03_app_infra_setup**
- some setup scripts to get lakebase provisioned and sync our scored transcripts with lakebase
- see App in another repo: https://github.com/maxcarduner1/call-center-analytics-app
