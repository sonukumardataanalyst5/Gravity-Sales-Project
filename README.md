# 📊 Superstore Marketing Campaign Dashboard

An **AI-powered, interactive 2-page dashboard** that analyzes a Superstore's Gold Membership campaign (only **14.91% acceptance rate**) using Machine Learning to predict who will accept and what drives customer spending.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-green?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-orange?logo=scikit-learn)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualization-yellow)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 🎯 Problem Statement

A Superstore runs a **yearly Gold Membership campaign** offering **20% discount** on all purchases. But only **14.91% customers say YES** — why so low?

**Our Goal:**
- Find **WHO** will accept the campaign
- Find **WHAT** drives spending
- Improve the next campaign with data-driven recommendations

---

## 🔗 Live Demo

> **[Click here to view the live dashboard](https://gravity-sales-project.onrender.com/)**

---

## ✨ Features

### 📈 Page 1 — KPI Executive Dashboard

**8 Live KPI Cards:**
| KPI | Description |
|---|---|
| 👥 Total Customers | Total customer count (filtered) |
| 💰 Average Income | Mean income across selected segment |
| 🛒 Total Spend | Sum of all product category spending |
| 📊 Avg Spend / Customer | Per-customer average spending |
| 🎯 Campaign Response Rate | % who accepted Gold Membership |
| 📅 Avg Recency | Average days since last purchase |
| ⚠️ Complaint Rate | % of customers who complained |
| 🌐 Avg Web Visits / Month | Average monthly website visits |

**15+ Interactive Charts:**
- Spend by Category (Wines, Meat, Fish, Fruits, Sweets, Gold)
- Purchase Channels — Doughnut (Web, Catalog, Store, Deals)
- Avg Spend by Education, Marital Status, Age Group
- Response Rate by Education, Marital Status, Age Group
- Response Rate by Web Purchases, Income Group, Recency
- Customer Count by Education (Doughnut)
- Top 10 Highest-Spending Customers (Table)

**5 Interactive Slicers (Filters):**
| Slicer | Options |
|---|---|
| Education | All, Basic, Graduation, Master, PhD |
| Marital Status | All, Single, Married, Together, Divorced, Widow |
| Age Group | All, 18-30, 31-40, 41-50, 51-60, 61-70, 70+ |
| Income Group | All, <25K, 25K-50K, 50K-75K, 75K-100K, 100K+ |
| Has Children | All, Yes, No |

🔥 **Cross-Filtering:** Click any chart bar or doughnut slice → entire dashboard filters instantly!

---

### 🤖 Page 2 — ML & Statistical Analysis

**Random Forest Classifier Setup:**
| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Trees | 200 |
| Max Depth | 12 |
| Features | 19 numeric features |
| Train/Test Split | 75% / 25% (stratified) |
| Class Balancing | SMOTE Oversampling |

**SMOTE (Synthetic Minority Oversampling):**
- Original data was **highly imbalanced** (~85% rejected, ~15% accepted)
- After SMOTE → Balanced 50/50
- **Buyer recall improved significantly** — model finds more actual buyers per 100
- Business value: Saves campaign cost and improves ROI

**ML Visualizations:**
- Feature Importance — Top 15 (horizontal bar chart)
- Original vs SMOTE Model Comparison (Accuracy, Recall, Buyers per 100)
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score, Support)
- Feature Importance Comparison: Original vs SMOTE (Top 10)

**Statistical Analysis:**
- Correlation Matrix Heatmap (15 numeric columns, color-coded)
- Descriptive Statistics (count, mean, std, min, 25%, 50%, 75%, max, skewness, kurtosis)

---

## 🧹 Data Cleaning & Preprocessing

| Step | What Was Done | Why |
|---|---|---|
| Missing Values | Income NaN → filled with **median** | Median is best for skewed distributions |
| Date Formatting | `Dt_Customer` parsed with mixed format | Original dates had inconsistent formats |
| Age Derivation | Calculated from `Year_Birth`, removed < 18 or > 100 | Some records had birth year 1893! |
| Marital Status | Merged `Alone`, `Absurd`, `YOLO` → `Single` | Cleaning inconsistent values |
| Education | Merged `2n Cycle` → `Basic` | Standardizing education levels |
| TotalSpend | Sum of Wines + Fruits + Meat + Fish + Sweets + Gold | Single metric for total value |
| TotalPurchases | Sum of Web + Catalog + Store + Deals | Single metric for purchase activity |
| HasChildren | 1 if Kidhome + Teenhome > 0 | Binary flag for family segmentation |
| Age Groups | Binned into 6 groups (18-30 to 70+) | For demographic analysis |
| Income Groups | Binned into 5 groups (<25K to 100K+) | For income-based segmentation |
| TopCategory | Each customer's highest spending category | Identifies product preference |
| Final Catch-all | All remaining numeric NaN → 0 | Bulletproof preprocessing |

---

## 🔑 Key Insights Found

### 📊 Spending Insights
- **Wines** is the #1 spending category, followed by **Meat Products**
- **PhD & Master's** holders spend significantly more than Basic/Graduation
- **Customers without children** spend nearly 2x more than families
- **Store purchases** are the dominant channel, followed by Web

### 🎯 Campaign Acceptance Insights
- Overall acceptance rate: **~14.91%** (very low!)
- **PhD customers** have the highest response rate (~15-20%)
- **High-income customers ($75K+)** are most likely to accept
- **Recent shoppers** (last 30 days) respond much better
- **More web purchases** → higher acceptance
- **Family acceptance is low** at ~10.3%

### 🤖 ML Model Insights
- **Top 3 Predictors:** TotalSpend, Income, Recency
- SMOTE improved **buyer recall** — finds more actual buyers
- Original model was biased toward predicting "No" (majority class)
- After SMOTE, model catches more true buyers

---

## 💡 Strategic Recommendations

| # | Recommendation | Rationale |
|---|---|---|
| 1 | **Priority Email for PhD & Master's** | 15-20% acceptance — highest among education groups |
| 2 | **Family Gold Bundle** | Only 10.3% family acceptance — bundle Meat & Fruits discounts |
| 3 | **Win-Back Campaign (High Recency)** | Customers >60 days rarely accept — send reactivation offers |
| 4 | **Digital Conversion for In-Store** | 10% online discount to build web behavior — drives acceptance |
| 5 | **"Gourmet Club" Positioning** | Wines & Meat are top categories — premium food club, not generic discount |
| 6 | **VIP Outreach for High Spenders** | Top 11% are most likely to accept — VIP calls & free delivery |

---

## 👤 Ideal Target Customer Profile

- **💰 Income:** Above $75,000
- **🎓 Education:** PhD or Graduation level
- **🌐 Behavior:** 5+ web visits/month, high web purchasers
- **🛒 Spending:** Already high Total Spend
- **⏱️ Recency:** Shopped within last 30 days

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask (10+ REST API endpoints) |
| ML | Scikit-Learn (Random Forest Classifier) |
| Class Balancing | Imbalanced-Learn (SMOTE) |
| Statistics | SciPy, Pandas |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| UI Design | Dark-mode glassmorphism, responsive |
| Deployment | Render.com (free, 24/7) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Open browser → http://localhost:5000
```

---

## 📁 Project Structure

```
├── app.py                                    # Flask backend + ML pipeline
├── requirements.txt                          # Python dependencies
├── Superstore Marketing Campaign Dataset.csv # Raw dataset
├── templates/
│   └── index.html                            # 2-page dashboard
├── static/
│   └── style.css                             # Dark-mode glassmorphism styles
├── deployment_guide.md                       # Render.com deployment guide
└── README.md
```

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

⭐ **If you found this useful, please star the repo!**
