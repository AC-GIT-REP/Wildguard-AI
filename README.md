# WildGuard AI 🌍🐘

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/XGBoost-Risk_Classification-green.svg)](https://xgboost.ai)
[![Generative AI](https://img.shields.io/badge/Groq-Llama_3.1-black.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WildGuard AI** is a predictive, machine-learning-powered conservation intelligence platform. It bridges the gap between raw ecological data and real-time conservation action by using multi-model ML architectures (XGBoost, Random Forest) to forecast wildlife population trends, classify species extinction risk levels, and generate automated mitigation strategies — all through a premium, cinematic dashboard experience.

---

## ✨ Core Features

*   **📈 Population Trend Analysis:** Random Forest model trained on engineered wildlife census data to detect population trends (Increasing, Stable, Declining, Critical) across endangered species.
*   **🚨 Multi-Class Risk Classification:** XGBoost classifier assigning urgency tiers (High, Medium, Low Risk) using complex, non-linear relationships between habitat loss metrics, poaching records, and population volatility.
*   **📡 Threat Radar & Geospatial Mapping:** Interactive `PyDeck` globe displaying global poaching hotspots and risk vectors in real-time.
*   **🌿 'Ask Prakriti' AI Assistant:** A globally persistent, floating AI chatbot powered by the Groq API (Llama 3.1 8B), providing rule-based and generative wildlife conservation guidance.
*   **🌍 Guardian Community:** A gamified citizen-science module where users submit field observations to a community leaderboard backed by MongoDB.
*   **📑 Automated PDF Reporting:** Built-in document generation engine using `fpdf2`, converting ML metrics and analysis into professional, policy-ready PDF reports.
*   **🔐 User Authentication:** Secure login/registration system with password hashing (PBKDF2-SHA256) and MongoDB storage.

## 🛠️ Technology Stack

**Frontend & Architecture**
*   [Streamlit](https://streamlit.io/) — Core application framework with custom HTML/CSS cinematic UI.
*   [PyDeck](https://deckgl.readthedocs.io/) & [Plotly](https://plotly.com/) — Geospatial intelligence and interactive charting.

**Machine Learning Engine**
*   [XGBoost](https://xgboost.ai/) — Gradient-boosted decision trees for risk classification.
*   [Scikit-Learn](https://scikit-learn.org/) — Random Forest for population trend detection + preprocessing pipelines.
*   [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — Data engineering and feature extraction.

**Backend & Integrations**
*   [MongoDB](https://www.mongodb.com/) (via PyMongo) — User authentication & Guardian Community leaderboard storage.
*   [Groq API](https://groq.com/) — Lightning-fast LLM inference powering the Prakriti chatbot.
*   [FPDF2](https://py-pdf.github.io/fpdf2/) — Automated PDF report generation.

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.10+ installed
- A [Groq API key](https://console.groq.com/) (free tier available)
- A [MongoDB Atlas](https://www.mongodb.com/atlas) connection string (free tier available)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AC-GIT-REP/wildware-ai.git
   cd wildware-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   MONGO_URI=your_mongodb_connection_string_here
   ```

4. **Launch the application**
   ```bash
   streamlit run app/app.py
   ```

## 📂 Project Structure

```text
WildGuard-AI/
├── app/
│   ├── app.py                       # Main Streamlit application (UI + logic)
│   ├── inference_utils.py           # ML model loading and inference engine
│   ├── data_validator.py            # Data integrity checks for analytical pipelines
│   ├── mongodb_manager.py           # MongoDB connection, auth, and leaderboard logic
│   └── assets/                      # Wildlife imagery for the landing page gallery
├── data/
│   ├── raw_wildlife_data.csv        # Baseline species metrics & census records
│   ├── cleaned_wildlife_data.csv    # Cleaned dataset after preprocessing
│   ├── engineered_wildlife_data.csv # Final feature-engineered dataset
│   ├── classification_dataset.csv   # XGBoost training dataset
│   ├── trend_dataset.csv            # Random Forest training dataset
│   ├── forecast_dataset.csv         # Population forecasting dataset
│   ├── poaching_incidents.csv       # Global poaching incident records
│   ├── demo_upload.csv              # Sample file for the upload demo feature
│   ├── guardian_leaderboard.json    # Community leaderboard data
│   ├── feature_engineering.py       # Feature extraction logic
│   ├── preprocess_data.py           # Raw data cleaning scripts
│   └── generate_poaching_data.py    # Synthetic poaching data generator
├── models/
│   ├── rf_trend_model.pkl           # Trained Random Forest (trend detection)
│   ├── xgboost_risk_model.json      # Trained XGBoost (risk classification)
│   ├── poaching_threat_model.pkl    # Trained poaching threat model
│   ├── xgboost_metrics.json         # XGBoost evaluation metrics
│   ├── train_rf_trend.py            # RF training script
│   ├── xgboost_risk_classification.py  # XGBoost training script
│   ├── prophet_forecasting.py       # Prophet forecasting script (offline)
│   ├── lstm_trend_detection.py      # LSTM training script (offline)
│   └── model_comparison.py          # Model comparison and evaluation
├── plots/                           # Pre-generated model evaluation plots
├── results/                         # Comparison tables and offline results
├── .env                             # API keys (not committed to Git)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🧠 The ML Pipeline

WildGuard AI is not just a visualization wrapper — the data goes through a rigorous engineering pipeline:

1. **Data Ingestion:** IUCN-style wildlife metrics and census records are parsed and validated.
2. **Feature Engineering:** Compound variables like `population_volatility`, `decline_rate`, `habitat_fragmentation_index`, and `normalized_historical_peak` are computed.
3. **Model Training (Offline):**
   - Random Forest trained on `trend_dataset.csv` for population trend classification.
   - XGBoost trained on `classification_dataset.csv` for multi-class risk level prediction.
4. **Runtime Inference:** The dashboard loads pre-trained model weights (`.pkl`, `.json`) for instant predictions on new data — no retraining required.

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
