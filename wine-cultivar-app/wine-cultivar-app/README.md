# 🍷 Wine Cultivar Predictor (Streamlit ML Web App)

An interactive web app that wraps a trained scikit-learn model. Move the sliders to enter a wine's
chemical measurements and get a live prediction of its grape cultivar with class probabilities.

**Live app:** _paste your Streamlit Community Cloud link here_

## Features
- Streamlit interface with 6 sliders (ranges and defaults come from the training data)
- Model loaded once with `joblib` and `@st.cache_resource`
- Prediction shown with `st.metric` plus a probability bar chart
- Friendly messages for a missing model file or invalid input (no raw error traces)
- Educational caption explaining the model and its data

## Model
Random Forest in a scikit-learn Pipeline (StandardScaler + RandomForestClassifier) trained on the
UCI Wine dataset bundled with scikit-learn (178 samples, 3 cultivars). Features, in order:
`alcohol, malic_acid, total_phenols, flavanoids, color_intensity, proline`.
Test accuracy: 100% on a stratified 25% split (small dataset, so treat this as a learning exercise).

## Run locally
```bash
pip install -r requirements.txt
python train_model.py        # creates model.joblib and model_meta.json
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository (include `model.joblib` and `model_meta.json`).
2. Go to https://streamlit.io/cloud, sign in with GitHub and click **Create app**.
3. Select the repo, branch `main`, main file `app.py`, then **Deploy**.
4. Open the public link on your phone to test it.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `train_model.py` | Trains and saves the model |
| `model.joblib` | Saved model |
| `model_meta.json` | Feature order, slider ranges, class names, accuracy |
| `requirements.txt` | Dependencies (scikit-learn pinned to the training version) |

_Educational example only; not advice._
