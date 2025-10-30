# 🌍 SDG 7: Solar-Powered Agricultural Intelligence — Kenya Maize Yield Predictor

## ⚡ Project Overview

This project aligns with **United Nations Sustainable Development Goal (SDG) 7 — Affordable and Clean Energy**, by integrating **solar energy data** into **agricultural intelligence systems**.  
Our goal is to empower smallholder farmers and policymakers in Kenya with **data-driven insights** to increase crop yields while leveraging **clean, renewable energy** resources.

---

## 🚜 Problem Statement

In many parts of Sub-Saharan Africa — including Kenya — maize is a staple food and a major economic crop.  
However, **unpredictable weather patterns**, **poor access to reliable energy**, and **limited use of data** make it difficult for farmers to plan irrigation, fertilizer use, and planting cycles efficiently.

Farmers often:
- Rely on guesswork instead of evidence-based planting schedules.
- Suffer crop losses due to drought or insufficient solar-powered irrigation.
- Have limited visibility into the relationship between **solar energy**, **rainfall**, and **crop performance**.

This leads to **reduced yields**, **food insecurity**, and **economic instability** in rural communities.

---

## 🌞 Our Solution

We are building a **Solar-Powered Agricultural Prediction System** that:
1. Uses **NASA POWER satellite data** to track **solar irradiance**, **rainfall**, and **temperature** in real-time.  
2. Predicts **maize yield potential** based on weather and solar energy patterns.  
3. Suggests **optimal planting and irrigation times** powered by **renewable solar energy insights**.  
4. Offers an interactive dashboard (in the upcoming phase) for visualizing trends and decisions.

By combining **clean energy data (SDG 7)** with **smart agriculture (SDG 2: Zero Hunger)**, the project bridges two critical sustainability goals.

---

## 🧠 Technical Summary

- **Data Source:** NASA POWER API (solar and climate data)
- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`, `xgboost`, `geopandas`
- **Development Environment:** Cursor IDE  
- **Version Control:** GitHub  
- **Deployment:** Netlify (for dashboard or visualization website)

The model takes daily weather and solar irradiance readings and computes seasonal features such as 30-day rainfall, average temperature, and total solar energy exposure — which are used to predict potential maize yield.

---

## 🧩 Project Workflow

1. **Data Collection:** Fetch weather and solar data using NASA POWER API.  
2. **Feature Engineering:** Clean and transform data (e.g., rolling rainfall, average temperature).  
3. **Model Training:** Use machine learning (XGBoost / RandomForest) to predict yield outcomes.  
4. **Visualization:** Plot rainfall, temperature, and solar patterns.  
5. **Deployment:** Build an interactive visualization web app (future phase).

---

## 📸 Screenshots

> _(Add screenshots of your Colab/IDE and output graphs here once your code runs successfully)_

Example:
- NASA POWER data sample
- Rainfall & temperature visualization
- Model prediction results (optional)

---

## 💡 Impact

By using free and renewable solar energy as a data backbone:
- Farmers can plan **irrigation schedules** powered by sunlight patterns.  
- Governments and NGOs can monitor **regional yield potential** in real-time.  
- The project supports **clean energy adoption** and **climate resilience** in African agriculture.

This project demonstrates how **AI + Renewable Energy** can create inclusive, sustainable solutions that directly address SDG 7 and SDG 2.

---

## 🧭 Team / Contributors
- **Developer:** [Your Name]  
- **Institution:** PLP Academy  
- **Mentor:** [Mentor’s Name, if applicable]  

---

## 🚀 How to Run This Project

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/solar-yield-predictor.git

# 2. Navigate to folder
cd solar-yield-predictor

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the main script
python maize_yield_model.py
