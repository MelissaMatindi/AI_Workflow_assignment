# Hospital 30-Day Readmission Risk Prediction  
**AI Development Workflow Assignment** – AI for Software Engineering  
  
**Tech Stack**: Python, pandas, LightGBM, SHAP, scikit-learn, FastAPI, Docker-ready  

## Project Overview  
End-to-end AI workflow for predicting unplanned 30-day hospital readmissions using structured EHR data.  
Deliberately optimized for **high recall (≥ 70%)** while preserving clinical interpretability and HIPAA-readiness.

### Key Deliverables  
- `AI_Readmission_Prediction_Notebook.ipynb` – fully reproducible notebook (all 6 steps + plots)  
- Synthetic balanced dataset (2000 patients, 12% readmission rate)  
- Production-grade LightGBM model with **monotonic constraints** on clinically known directions  
- SHAP global & local explanations (summary + waterfall)  
- FastAPI inference endpoint (Docker-ready)  
- Scaler + feature order persistence for zero-downtime deployment  

## Quick Start (5 seconds)

```bash
git clone https://github.com/yourusername/readmission-risk-prediction.git
cd readmission-risk-prediction

# Run the notebook (Google Colab / Jupyter / VS Code)
# Or spin up the API instantly:
uvicorn app:app --reload
# → POST http://127.0.0.1:8000/predict
