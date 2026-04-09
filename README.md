# WildGuard AI 🌍🐘

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/XGBoost_|_Prophet-Passed-green.svg)](https://xgboost.ai)
[![Generative AI](https://img.shields.io/badge/Groq-Llama_3.1-black.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WildGuard AI** is a predictive, machine-learning-powered conservation intelligence platform. Designed bridging the gap between raw ecological data and real-time conservation efforts, it utilizes multi-model ML architectures (XGBoost, LSTM, Facebook Prophet) to forecast wildlife population trends, categorize extinction risks, and generate automated mitigation strategies before populations hit critical points of no return.

---

## ✨ Core Features

*   **📈 Predictive Population Forecasting:** Utilizes Facebook Prophet for time-series forecasting, projecting species population trajectories up to 2030 based on historical census data.
*   **🚨 Multi-Class Threat Assessment:** Employs an XGBoost classifier to assign an urgency tier (High, Medium, Low Risk) utilizing complex, non-linear relationships between habitat metrics and historical poaching records.
*   **📡 Threat Radar & Geospatial Mapping:** Displays global risk vectors and poaching hotspots via interactive `PyDeck` maps.
*   **🌿 'Ask Prakriti' AI Assistant:** A globally persistent, floating AI chatbot integrated directly via the Groq API (Llama 3.1 8B), providing 24/7 rule-based and generative wildlife conservation guidance.
*   **🌍 Guardian Community:** A gamified, citizen-science module allowing users to submit verifiable field photos to a decentralized leaderboard, actively crowdsourcing conservation awareness.
*   **📑 Automated PDF Reporting:** A built-in document generation engine using `fpdf2` and `matplotlib`, instantly converting ML metrics into professional, policy-ready PDF reports.

## 🛠️ Technology Stack

**Frontend & Architecture**
*   [Streamlit](https://streamlit.io/) - Core application framework and interactive dashboarding.
*   [PyDeck](https://deckgl.readthedocs.io/) & Plotly - Geospatial intelligence and complex charting.

**Machine Learning Engine**
*   [XGBoost](https://xgboost.ai/) - Gradient-boosted decision trees for risk classification.
*   [Facebook Prophet](https://facebook.github.io/prophet/) - Time-series prediction for population decay/recovery.
*   [Scikit-Learn](https://scikit-learn.org/) / Pandas / Numpy - Data engineering and preprocessing pipelines.
*   [Keras/TensorFlow](https://www.tensorflow.org/) - LSTM-based temporal anomaly detection.

**External Integrations**
*   [Groq API](https://groq.com/) - Lightning-fast LLM inference powering the chatbot fallback mechanics.

## ⚙️ Installation & Usage

### Prerequisites
Make sure you have Python 3.10+ installed on your local environment. 

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wildguard-ai.git
   cd wildguard-ai
   ```

2. **Install dependencies**
   We recommend setting up a virtual environment before installing the requirements.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   To enable the 'Ask Prakriti' chatbot, you will need to add your personal `GROQ_API_KEY` to the `_chatbot_respond()` function inside `app/app.py`.

4. **Launch the application**
   ```bash
   streamlit run app/app.py
   ```
   *(Alternatively, Windows users can simply double-click `run_dashboard.bat`)*

## 📂 Project Structure

```text
WildGuard-AI/
├── app/
│   ├── app.py                      # Main Streamlit application and UI routing
│   ├── inference_utils.py          # Class definitions for ML model loading and inference
│   └── data_validator.py           # Integrity checks for the analytical pipelines
├── data/                           
│   ├── raw_wildlife_data.csv       # Baseline species metrics & census records
│   ├── engineered_wildlife_data.csv# Preprocessed final dataset used by the dashboard
│   ├── feature_engineering.py      # Core logic for extracting training features
│   └── preprocess_data.py          # Scripts to handle raw data cleaning and sorting
├── models/
│   ├── rf_trend_model.pkl          # Trained Random Forest serialization
│   ├── xgboost_risk_model.json     # Trained XGBoost serialization
│   └── *_training.py               # Independent model training configuration scripts
├── plots/ & results/               # Artifacts generated during offline model evaluation
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

## 🧠 The ML Pipeline

WildGuard AI isn't simply a visualization wrapper — the data is heavily engineered before it reaches the frontend:
1. **Data Ingestion:** Structured IUCN-style metrics are parsed.
2. **Feature Engineering:** Calculation of compound variables like `population_volatility`, `decline_rate`, and `normalized_historical_peak` using Pandas.
3. **Training & Inference:** 
   - The dashboard invokes pre-trained XGBoost weights (`.json`) for instant risk-level inference.
   - Prophet modeling operates dynamically *at runtime*, recalculating confidence intervals and seasonality based on real-time slice inputs.

## 🤝 Contributing

Contributions are highly welcome. If you find a bug or have an idea for a feature (such as implementing the planned OpenCV backend for the Guardian Community photo-verification), please:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information. This system was originally developed as an academic Final Year Project.
