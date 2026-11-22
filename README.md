# California Housing Price Predictor — Task 2

Extension of **Maincrafts AI/ML Internship Task 1**. Compares original Linear Regression with the cross-validated tuned Decision Tree trained on the California Housing dataset.

## 🚀 Live Demo

Try the interactive Model Optimization & Comparison application:

https://california-housing-model-optimization.streamlit.app/

This application compares predictions from Linear Regression
and an optimized Decision Tree Regressor using the
California Housing dataset.

## Files

- `AI_ML_Task2_Model_Comparison.ipynb` — executed training, tuning, and evaluation notebook
- `AI_ML_Task2_Model_Comparison_Report.pdf` — mandatory two-page report
- `task2_best_model.joblib` — tuned Decision Tree pipeline (optional artifact)
- `house_price_model.joblib` — original Linear Regression pipeline from Task 1
- `app.py` — optional Streamlit model-comparison application
- `requirements.txt` — pinned runtime dependencies

## Test results

| Model | Test RMSE ($100k units) | Test R² |
|---|---:|---:|
| Linear Regression | 0.7456 | 0.5758 |
| Ridge Regression | 0.7456 | 0.5758 |
| Decision Tree (max_depth=5) | 0.7242 | 0.5997 |
| Tuned Decision Tree | **0.6022** | **0.7233** |

The tuned Decision Tree was selected based on 5-fold cross-validation RMSE on the training data. Test-set metrics are held-out results. Models predict **median census block-group house values**, not individual current-market prices.

## Run locally

Use **Python 3.12** and then run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Do not download arbitrary untrusted `.joblib` files; joblib unpickling can execute code. Models were trained with scikit-learn 1.6.1.

## Deploy

Create a separate public repository (for example `California-Housing-Task2-Model-Comparison`), upload these six files, then deploy `app.py` with Streamlit Community Cloud. Use Python 3.12 and let `requirements.txt` install the compatible scikit-learn version. Keeping this separate avoids accidentally disrupting the existing Task 1 deployment.
