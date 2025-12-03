# Amazon Company Hub — Two Month Insights (CMSE 830 Midterm)

**Goal:**  
Provide a clear and interactive Streamlit dashboard that helps non-technical teams understand what happened, why it happened, what it means, and what to do next through structured charts, captions, and exportable insights.

---

## Quick Start

To run locally:

```
pip install -r requirements.txt
streamlit run final_app.py
```

---

## Files Expected (Place beside `final_app.py`)

| File Name | Required | Description |
|------------|-----------|-------------|
| amazon_sales_with_two_months.csv | ✅ Yes | Core dataset showing purchases, discounts, prices, ratings, reviews, category, and month |
| amazon_reviews.csv | Optional | Review text, sentiment, and keywords for content planning |
| amazon_products.csv | Optional | Extra product metadata such as names, prices, and categories |

---

## App Structure

| Tab | Description |
|------|-------------|
| **1. How To Use** | Project overview, dataset details, purpose, and reading guide |
| **2. Executive Snapshot** | KPIs, monthly trend, category mix, highlights, and improvement plan |
| **3. Finance & Ops** | Discount posture, volume versus discount, and operational checklist |
| **4. Marketing & Content Strategy** | Engagement map, conversion proxy, keyword analysis, platform mix, and seven-day content calendar |
| **5. Product & Pricing** | Ratings distribution, pricing posture, and underperforming items |
| **6. Plans & Accountability** | Auto-generated team actions with owners, priorities, and due dates |

---

## Data Preparation

- Standardized all date fields into monthly format  
- Converted numeric columns such as purchases, discounts, and ratings  
- Normalized product categories for consistency  
- Created flags for promotional items such as best seller, coupon, and sponsored  
- Combined data sources for two months to enable comparison  

---

## Key Design Choices

- Focused on clarity and explainability over complex modeling  
- Added gray subtext beneath each chart summarizing:  
  - **Why:** the motivation for the visualization  
  - **What:** what the data shows  
  - **So What:** what the results mean  
  - **What Next:** recommended next steps  
- Used traffic light colors (green, yellow, red) for quick executive scanning  
- Included sentiment from text reviews to strengthen marketing insights  
- Designed all visuals so they can be understood without technical knowledge  

---

## Deployment Notes

- Tested with Streamlit 1.38 and Python 3.12  
- All datasets must be stored in the same folder as `final_app.py`  
- Do not use nested directories for data paths  
- Can be deployed on Streamlit Community Cloud for easy access  

---

## Summary

This project demonstrates end-to-end data analysis for Amazon product sales across two months.  
It integrates finance, marketing, and product analytics to provide a full company snapshot.  
The app encourages collaboration between teams through shared, easy-to-read insights rather than code or technical reports.

---

## 📁 Repository Structure

Below is the structure of this GitHub repository and what each file or folder contains.

```
📦 CMSE830_Midterm_AmazonHub/
│
├── final_app.py              # Main Streamlit dashboard application
├── README.md                 # Overview, setup instructions, and documentation
├── WHY_REPORT.md             # Written project report aligned with course rubric
│
├── requirements.txt          # Python dependencies (Streamlit, Plotly, Pandas, etc.)
├── runtime.txt               # Chnaging the Python version to get streamlit app to deploy
├── .gitignore                # Excludes temporary and cache files from version control
│
└── data/                     # Folder containing compressed datasets
    └── data.zip              # Zipped archive with three smaller CSVs:
                              #   • amazon_sales_with_two_months_small.csv
                              #   • amazon_reviews_small.csv
                              #   • amazon_products_small.csv
```

Each file in this repository serves a distinct purpose:
- **`final_app.py`** — contains all Streamlit logic, layout, and visualizations.
- **`README.md`** — provides setup, project summary, and documentation.
- **`WHY_REPORT.md`** — contains the in-depth written explanation of the full project process.
- **`requirements.txt`** — lists packages required to run the app locally or deploy online.
- **CSV files** — store the datasets used for this analysis.
- **`.gitignore`** — keeps unnecessary files out of GitHub (e.g., `__pycache__`, temporary data).

---
