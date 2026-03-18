# WildGuard AI — Gamma PPT Prompt

> **Paste this entire prompt into Gamma.app to generate your presentation.**
> Theme: Dark modern, nature/wildlife aesthetic. Use green + dark teal + white color palette.
> Font: Inter or Outfit. 10 slides maximum.

---

## SLIDE 1 — Title Slide

**Title:** WildGuard AI
**Subtitle:** A Machine Learning Framework for Endangered Species Population Forecasting & Conservation Risk Assessment

**Content:**
- Anshuman Chowdhury (JIS/2022/0788)
- Bishop Debnath (JIS/2022/0418)
- Argha Dutta (JIS/2022/1064)
- Department of Information Technology, JIS College of Engineering, Kolkata
- Under the guidance of Prof. Proloy Ghosh

**Style:** Full-bleed dark background with a faded wildlife silhouette (tiger or elephant). Title in large bold white text, subtitle in green accent. Team names in a clean row at the bottom.

---

## SLIDE 2 — Problem & Motivation

**Title:** Why This Matters

**Content (use icons beside each point):**
- 🔴 **44,000+ species** face extinction globally (IUCN Red List 2024)
- ⏳ Wildlife censuses happen only every **2–5 years** — too slow for early intervention
- 📉 By the time decline is confirmed, recovery options are limited
- 🌍 IUCN tracks **150,000+ species** — manual monitoring is impossible at this scale

**Key Statement (large, centered, green text):**
> "We need automated systems that can forecast, classify, and flag at-risk species — before it's too late."

**Style:** Dark background, each bullet with a colored icon on the left. The key statement stands alone at the bottom in accent color.

---

## SLIDE 3 — Our Solution: WildGuard AI

**Title:** What We Built

**Content (3-column layout):**

| 🔮 Forecasting | 📊 Trend Classification | ⚠️ Risk Assessment |
|---|---|---|
| Predict population 5 years ahead using ARIMA(1,1,1) | Classify trajectory: Sharp Decline, Moderate Decline, Stable, Recovering | Predict urgency: High, Medium, Low |
| Best model out of 3 tested | Best model out of 3 tested | Best model out of 3 tested |

**Below the columns:**
- Built on **426 census records** across **13 endangered species**, **6 regions**, **55 years** (1969–2024)
- **44 engineered features** from raw data
- Deployed as a **live Streamlit dashboard**

**Style:** Three equal cards side by side with icons. Each card has a gradient border (green → teal). Summary stats below in a horizontal bar.

---

## SLIDE 4 — System Architecture

**Title:** How It Works

**Content (flowchart — top to bottom):**

```
Raw Census Data (IUCN + NTCA)
        ↓
Preprocessing (426 records cleaned)
        ↓
Feature Engineering (44 features)
        ↓
   ┌────────────┬────────────┐
   ↓            ↓            ↓
ARIMA(1,1,1)  Random Forest  XGBoost
Forecasting   Trend Class.   Risk Class.
   ↓            ↓            ↓
5-Year        Trend Label   Risk Level
Forecast      (4 classes)   (H / M / L)
   └────────────┴────────────┘
              ↓
    Interactive Dashboard
    (Streamlit + Plotly)
```

**Validation Strategy (small box, bottom-right):**
- Train: All data ≤ 2020
- Test: Real census data 2021–2024
- Also compared against naive baseline

**Style:** Clean vertical flowchart with rounded boxes. Use green for data nodes, blue for model nodes, orange for output nodes. Small "Validation" callout in bottom corner.

---

## SLIDE 5 — Model Comparison Results

**Title:** 9 Models, 3 Tasks — Here's What Won

**Content (3 result blocks):**

**Forecasting (MAPE ↓ is better):**
| Model | Avg MAPE |
|-------|----------|
| ✅ **ARIMA** | **1.79%** |
| Naive Baseline | 5.86% |
| LSTM | 7.70% |
| Prophet | 19.04% |

**Trend Classification:**
| Model | F1 Score |
|-------|----------|
| ✅ **Random Forest** | **1.000** |
| SVM | 0.955 |
| LSTM | 0.827 |

**Risk Classification:**
| Model | Accuracy |
|-------|----------|
| ✅ **XGBoost** | **100%** |
| Random Forest | 100% |
| Logistic Reg. | 100% |

**Highlight callout (green box):**
> ARIMA beats the naive baseline by **69.4%** — proving the model captures real population trends, not just echoing the last data point.

**Style:** Three compact tables side by side. Winning models highlighted in green. The callout box at the bottom spans full width.

---

## SLIDE 6 — Forecast Accuracy Visualization

**Title:** Predicted vs Actual: How Close Did We Get?

**Content (2×2 grid of charts):**

Show 4 line charts (use actual data from the paper):

**Bengal Tiger (MAPE: 0.17%)**
- Historical (gray): 2017→2782, 2018→2967, 2019→3146, 2020→3324
- Actual (red circles): 2021→3503, 2022→3682
- Predicted (blue triangles): 2021→3500, 2022→3673

**Asiatic Lion (MAPE: 0.76%)**
- Historical (gray): 2017→583, 2018→614, 2019→644, 2020→674
- Actual (red): 2021→706, 2022→737, 2023→768, 2024→800
- Predicted (blue): 2021→704, 2022→733, 2023→761, 2024→790

**California Condor (MAPE: 2.32%)**
- Historical (gray): 2017→464, 2018→478, 2019→491, 2020→504
- Actual (red): 2021→523, 2022→542, 2023→561
- Predicted (blue): 2021→517, 2022→529, 2023→542

**Gharial (MAPE: 0.71%)**
- Historical (gray): 2017→650, 2018→720, 2019→790, 2020→860
- Actual (red): 2021→930, 2022→1000
- Predicted (blue): 2021→926, 2022→990

**Caption:** Red = actual census data, Blue = ARIMA predictions. The predicted values track reality within 0.1–2.3%.

**Style:** 2×2 grid. Each chart has a subtle gray grid, red solid line for actual, blue dashed for predicted. Species name + MAPE as chart title.

---

## SLIDE 7 — Species Conservation Status

**Title:** Current Status of All 13 Species

**Content (table — color-coded by risk):**

| Species | IUCN | Risk | Trend | Population |
|---------|------|------|-------|-----------|
| 🔴 Gt. Indian Bustard | CR | High | Recovery | 150 |
| 🔴 Gharial | CR | High | Recovery | 1,000 |
| 🔴 California Condor | CR | High | Recovery | 561 |
| 🔴 Asian Elephant | EN | High | Mod. Decline | 22,446 |
| 🟡 Bengal Tiger | EN | Medium | Recovery | 3,682 |
| 🟡 Asiatic Lion | EN | Medium | Recovery | 800 |
| 🟡 Ganges R. Dolphin | EN | Medium | Recovery | 4,000 |
| 🟡 African Elephant | EN | Medium | Stable | 415,000 |
| 🟢 Indian Rhinoceros | VU | Low | Recovery | 4,014 |
| 🟢 Giant Panda | VU | Low | Stable | 1,900 |
| 🟢 White Rhinoceros | NT | Low | Stable | 17,900 |
| 🟢 Humpback Whale | LC | Low | Recovering | 89,000 |
| 🟢 Gray Wolf | LC | Low | Stable | 6,800 |

**Key insight (bottom):**
> All 3 critically endangered species show recovery but with dangerously low absolute numbers — the Great Indian Bustard has only ~150 individuals left.

**Style:** Rows color-coded: red background for High risk, yellow for Medium, green for Low. Clean, compact table. Insight text in a callout box below.

---

## SLIDE 8 — Live Dashboard Demo

**Title:** The Dashboard: Making ML Accessible

**Content (screenshots or descriptions of 4 dashboard sections):**

**Panel 1 — Species Explorer:**
Select any species → see full population history + 5-year ARIMA forecast with confidence interval

**Panel 2 — Risk Assessment:**
Color-coded risk badges (High/Medium/Low) for all 13 species at a glance

**Panel 3 — Global Map:**
Interactive world map showing species locations, colored by risk level

**Panel 4 — AI Recommendations:**
Auto-generated conservation insights and action items per species

**Bottom note:**
> Built with Streamlit + Plotly | Real-time inference | No coding needed by end users

**Style:** 2×2 grid of dashboard screenshots (use actual screenshots from your Streamlit app if possible). Each panel has a thin border and a label. Dark background to match the dashboard's actual look.

---

## SLIDE 9 — Limitations & Future Work

**Title:** Where We Are & Where We're Going

**Two columns:**

**⚠️ Limitations:**
- Dataset limited to 13 species (public data only)
- No environmental covariates (climate, deforestation)
- Small test set (27 samples) — perfect scores need larger validation
- No access to restricted government databases (NTCA, WII) — built on publicly available research data

**🚀 Future Work:**
- Expand to 100+ species via IUCN + citizen science (eBird, iNaturalist)
- Add climate variables + satellite habitat metrics
- Geospatial species distribution modelling
- Bayesian uncertainty quantification
- Cross-species transfer learning
- **Live government API integration** (requires institutional authorization)

**Style:** Two-column layout. Left column with amber/yellow accent (limitations). Right column with green accent (future). Each point has a small icon or bullet.

---

## SLIDE 10 — Conclusion & Thank You

**Title:** Key Takeaways

**Content (3 big numbered points):**

1. **Classical models win on small ecological data** — ARIMA (1.79% MAPE) beat LSTM and Prophet by a wide margin
2. **Multi-task approach** — Forecasting + Trend + Risk in one unified pipeline, not three separate tools
3. **Actionable for conservation** — A working dashboard that turns census data into early warnings

**Impact Statement (large, centered, green):**
> "If a conservation officer can open a dashboard and immediately see that a species is declining with high risk — that's more actionable than a spreadsheet of numbers."

**Bottom:**
- **Thank You**
- Anshuman Chowdhury | Bishop Debnath | Argha Dutta
- Prof. Proloy Ghosh | JIS College of Engineering
- Q&A

**Style:** Clean, minimal. The 3 takeaways as large numbered cards. Impact quote in a full-width green gradient box. Team names centered at bottom with "Questions?" text.

---

## GAMMA STYLE INSTRUCTIONS

- **Theme:** Dark mode, nature/wildlife inspired
- **Primary Colors:** #0D1B2A (dark navy), #1B4332 (forest green), #2D6A4F (teal green), #FFFFFF (white text)
- **Accent:** #52B788 (bright green for highlights and callouts)
- **Font:** Inter or Outfit (clean, modern sans-serif)
- **Charts:** Minimal, flat design. No 3D effects. Red (#E63946) for actual data, Blue (#457B9D) for predictions
- **Icons:** Use simple line icons (Lucide or Phosphor style)
- **Transitions:** Subtle fade between slides, no flashy animations
- **Overall Vibe:** Think Nature journal meets tech startup pitch — serious but visually striking
