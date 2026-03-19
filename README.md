# 🪨 3D Geological Classification Model

Predicting subsurface lithology from borehole data using machine learning.

**Live demo:** [https://geo-ml.streamlit.app](https://geo-ml.streamlit.app)

## Problem

Understanding underground geology typically requires expensive drilling campaigns.
This project uses existing borehole data to train a neural network
that can predict rock type at any 3D point in the study area.

## Pipeline

```
CSV data → Elevation calc → Point cloud (3D) → Feature engineering → SMOTE → MLP training → 3D grid prediction → VTK export
```

## Key Results

| Approach                    | Accuracy | F1 (macro) |
|-----------------------------|----------|------------|
| Baseline MLP                | 86%      | 68%        |
| MLP + SMOTE                 | 86%      | 79%        |
| MLP + SMOTE + Features      | **90%**  | **85%**    |

## Tech Stack

Python · scikit-learn · NumPy · Pandas · PyVista · Plotly · Streamlit · SMOTE · joblib

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py                # Streamlit portfolio app
├── geoModel.py           # Core ML pipeline class
│── p2_modeling.ipynb     # Notebook with model training & evaluation
├── requirements.txt
└── README.md
```

## Author

**Maciej Nikiel** — [LinkedIn](https://linkedin.com/in/maciej-nikiel) · [GitHub](https://github.com/macwojs)
