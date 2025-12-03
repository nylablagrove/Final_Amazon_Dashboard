# ============================================================
# 📦 AMAZON COMPANY HUB — TWO-MONTH INSIGHTS (CMSE 830 MIDTERM)
# ============================================================
# what this file delivers (plain english):
#   - a non-technical, executive-friendly hub with seven tabs:
#       1) how to use (full project intro, datasets, why it matters)
#       2) imputation & modeling (missing data, imputation examples, linear regression)
#       3) executive snapshot (kpis, trend, mix, highlights, plan)
#       4) finance & ops (discount posture, volumes, ops checklist)
#       5) marketing & content strategy (engagement, keywords, platform plan, content calendar)
#       6) product & pricing (ratings, price ladder, underperformers)
#       7) plans & accountability (cross-team action list + download)
#   - everything is filter-aware by month and category
#   - color system:
#         • green / yellow / red  → results or alerts only
#         • blue                  → informational guidance
#   - heavy comments above blocks (not inline), clear sentences, no em dashes
# data files expected beside this file inside the repo:
#   - required: data/data.zip containing:
#         • amazon_sales_with_two_months_small.csv
#         • amazon_reviews_small.csv
#         • amazon_products_small.csv
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta
from textblob import TextBlob
import os
import zipfile

# setting a page title and making the layout wide so charts can use the space
st.set_page_config(page_title="Amazon Company Hub", layout="wide")

# top-level page title so users know where they are
st.title("📦 Amazon Company Hub — Two-Month Insights")

# defining a small color system for result and info boxes
PALETTE = {
    "good": {"bg": "#DCFCE7", "bar": "#22C55E", "emoji": "✅"},
    "ok":   {"bg": "#FEF9C3", "bar": "#EAB308", "emoji": "🟨"},
    "bad":  {"bg": "#FEE2E2", "bar": "#EF4444", "emoji": "🟥"},
    "info": {"bg": "#E0E7FF", "bar": "#6366F1", "emoji": "ℹ️"},
    "warn": {"bg": "#FFE4E6", "bar": "#FB7185", "emoji": "⚠️"},
}

# helper to render a colored callout box with a title, bold headline, and body text
def _box(title, headline, text, tone="info"):
    style = PALETTE.get(tone, PALETTE["info"])
    st.markdown(
        f"""
<div style="border-left:8px solid {style['bar']}; background:{style['bg']};
            padding:14px 18px; border-radius:10px; margin:14px 0;">
  <div style="font-weight:700; color:#111827; margin-bottom:6px;">{style['emoji']} {title}</div>
  <div style="font-weight:800; color:#111827; margin-bottom:6px;">{headline}</div>
  <div style="color:#111827; line-height:1.6;">{text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def box_result(headline, text, level="ok", title="Result"):
    _box(title=title, headline=headline, text=text, tone=level)

def box_info(headline, text, title="Notes"):
    _box(title=title, headline=headline, text=text, tone="info")

def box_warn(headline, text, title="Attention"):
    _box(title=title, headline=headline, text=text, tone="warn")

# helper to place small gray figure captions under a chart
def fig_notes(why, what, sowhat, nextstep):
    st.markdown(
        f"""
<div style="font-size: 0.92rem; color:#374151; background:#F3F4F6; border-radius:10px; padding:12px 14px; margin-top:-6px;">
  <div><strong>Why:</strong> {why}</div>
  <div><strong>What:</strong> {what}</div>
  <div><strong>So What:</strong> {sowhat}</div>
  <div><strong>What Next:</strong> {nextstep}</div>
</div>
""",
        unsafe_allow_html=True
    )

# === basic helpers ==========================================================

def safe_read_csv(path):
    if path is None:
        return None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    try:
        return pd.read_csv(path, encoding="latin1", encoding_errors="ignore", low_memory=False)
    except Exception:
        return None

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def safe_bubble_sizes(series, min_size=8, max_size=40):
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if len(s) == 0:
        return None
    m = float(s.max())
    if m <= 0:
        return None
    return (min_size + (s / m) * (max_size - min_size)).astype(float)

def guard(df, message):
    if df is None or df.empty:
        st.info(message)
        return False
    return True

def pct(n, d):
    if d in [0, None, np.nan] or pd.isna(d):
        return np.nan
    return 100.0 * n / d

# median-by-category imputation for charts only
def impute_by_category(df):
    """
    fill missing numeric values with median by category, then overall median
    this is used only for chart copies, not for core KPIs
    """
    if df is None or df.empty:
        return df
    if "Category_std" not in df.columns:
        return df
    df_imp = df.copy()
    num_cols = df_imp.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return df_imp
    grp = df_imp.groupby("Category_std")
    for col in num_cols:
        med = grp[col].transform("median")
        overall = df_imp[col].median()
        df_imp[col] = df_imp[col].fillna(med)
        if not pd.isna(overall):
            df_imp[col] = df_imp[col].fillna(overall)
    return df_imp

# === load data from data/data.zip ===========================================

data_zip_path = os.path.join("data", "data.zip")

def _read_from_zip(zf: zipfile.ZipFile, member: str):
    try:
        with zf.open(member) as f:
            return pd.read_csv(f, low_memory=False)
    except KeyError:
        return None

try:
    with zipfile.ZipFile(data_zip_path) as z:
        sales = _read_from_zip(z, "amazon_sales_with_two_months_small.csv")
        if sales is None:
            st.error("Missing amazon_sales_with_two_months_small.csv inside data/data.zip.")
            st.stop()
        reviews = _read_from_zip(z, "amazon_reviews_small.csv")
        catalog = _read_from_zip(z, "amazon_products_small.csv")
except FileNotFoundError:
    sales   = safe_read_csv(os.path.join("data", "amazon_sales_with_two_months_small.csv"))
    reviews = safe_read_csv(os.path.join("data", "amazon_reviews_small.csv"))
    catalog = safe_read_csv(os.path.join("data", "amazon_products_small.csv"))
    if sales is None:
        st.error("Could not find data. Expect data/data.zip or data/amazon_sales_with_two_months_small.csv.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data from zip: {e}")
    st.stop()

# === prep functions =========================================================

def prep_sales(df):
    """
    standardize month, numerics, category, and promo flags
    """
    df = df.copy()
    if "month" not in df.columns:
        if "data_collected_at" in df.columns:
            df["month"] = pd.to_datetime(df["data_collected_at"], errors="coerce").dt.to_period("M").astype(str)
        else:
            df["month"] = "2025-08"
    for c in ["purchased_last_month", "product_rating", "total_reviews",
              "discount_percentage", "discounted_price", "original_price"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    if "product_category" in df.columns:
        df["Category_std"] = df["product_category"].astype(str).str.strip()
    else:
        df["Category_std"] = "Unknown"
    for src, dst in [("is_best_seller", "flag_best_seller"),
                     ("has_coupon", "flag_coupon"),
                     ("is_sponsored", "flag_sponsored")]:
        if src in df.columns:
            s = df[src].astype(str).str.lower().str.strip()
            df[dst] = s.isin(["true", "1", "yes", "y", "t"])
    return df.dropna(subset=["month"])

def prep_reviews(df):
    """
    light cleaning: month, numeric rating, review text, sentiment, category
    """
    if df is None:
        return None
    df = df.copy()
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").astype(str)
    rate_col = next((c for c in df.columns if "rating" in c.lower()), None)
    if rate_col:
        df["rating"] = safe_num(df[rate_col])
    text_col = next((c for c in df.columns if ("text" in c.lower() or "review" in c.lower())), None)
    if text_col:
        df["review_text"] = df[text_col].astype(str).fillna("")
        try:
            df["sentiment"] = df["review_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
        except Exception:
            df["sentiment"] = np.nan
    cat_col = next((c for c in df.columns if "category" in c.lower()), None)
    if cat_col:
        df["Category_std"] = df[cat_col].astype(str).str.strip()
    return df

def prep_catalog(df):
    """
    optional: normalize product metadata if available
    """
    if df is None:
        return None
    df = df.copy()
    for c in ["discounted_price", "actual_price", "original_price"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    for c in ["product_name", "product_title"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "category" in df.columns and "Category_std" not in df.columns:
        df["Category_std"] = df["category"].astype(str).str.strip()
    return df

sales   = prep_sales(sales)
reviews = prep_reviews(reviews)
catalog = prep_catalog(catalog)

# === sidebar controls =======================================================

st.sidebar.header("Controls")

months_all = sorted(sales["month"].dropna().unique().tolist())
month_focus = st.sidebar.selectbox("Month", months_all, index=len(months_all) - 1)

prev_idx = max(0, months_all.index(month_focus) - 1)
prev_month = months_all[prev_idx]

cat_counts = (
    sales[sales["month"] == month_focus]
    .groupby("Category_std")["purchased_last_month"]
    .sum()
    .sort_values(ascending=False)
)
labels = [f"{c} ({int(cat_counts[c]):,})" for c in cat_counts.index]
label_to_val = dict(zip(labels, cat_counts.index.tolist()))
chosen = st.sidebar.multiselect("Categories", labels, default=labels[:5] if len(labels) > 5 else labels)
cat_sel = [label_to_val[x] for x in chosen] if chosen else []

sales_f = sales[sales["Category_std"].isin(cat_sel)] if cat_sel else sales.copy()
if sales_f.empty:
    st.warning("That category selection returned no rows. Showing all categories instead.")
    sales_f = sales.copy()

curm  = sales_f[sales_f["month"] == month_focus]
prevm = sales_f[sales_f["month"] == prev_month]

# charts always use an imputed copy; KPIs and action rules use the raw filtered data
chart_sales_f = impute_by_category(sales_f)
chart_curm    = chart_sales_f[chart_sales_f["month"] == month_focus]
chart_prevm   = chart_sales_f[chart_sales_f["month"] == prev_month]

st.sidebar.markdown("### Legend")
st.sidebar.write("✅ Good result")
st.sidebar.write("🟨 Okay result")
st.sidebar.write("🟥 Needs attention")
st.sidebar.write("ℹ️ Information")
st.sidebar.write("⚠️ Reminder")

st.sidebar.markdown("### Reminders")
st.sidebar.write("• Change the month to see different snapshots.")
st.sidebar.write("• Narrow categories to focus the story.")
st.sidebar.write("• Download actions at the end for follow up.")

# === KPI + regression helpers ==============================================

def kpi_block(df):
    if df is None or df.empty:
        return dict(purchases=0, rating=np.nan, disc=np.nan, reviews=0)
    return dict(
        purchases = safe_num(df.get("purchased_last_month")).sum(skipna=True),
        rating    = safe_num(df.get("product_rating")).mean(),
        disc      = safe_num(df.get("discount_percentage")).mean(),
        reviews   = safe_num(df.get("total_reviews")).sum(skipna=True),
    )

def simple_regression_purchases(df):
    """
    multiple linear regression with standardized predictors using numpy only
    y = purchased_last_month
    X = rating, discount, total reviews (where available)
    """
    if df is None or df.empty:
        return None, None, None

    driver_specs = [
        ("Rating (★)", "product_rating"),
        ("Discount %", "discount_percentage"),
        ("Total Reviews", "total_reviews"),
    ]
    needed = ["purchased_last_month"] + [col for _, col in driver_specs]
    if not all(c in df.columns for c in needed):
        return None, None, None

    temp = df[needed].copy().dropna()
    if len(temp) < 20:
        return None, None, None

    y = safe_num(temp["purchased_last_month"]).astype(float).values

    X_list = []
    used_names = []
    used_cols = []
    for label, col in driver_specs:
        vals = safe_num(temp[col]).astype(float).values
        std = vals.std()
        if std == 0 or np.isnan(std):
            continue
        vals_z = (vals - vals.mean()) / std
        X_list.append(vals_z)
        used_names.append(label)
        used_cols.append(col)

    if not X_list:
        return None, None, None

    X = np.vstack(X_list).T
    y_mean = y.mean()
    y_std = y.std()
    y_z = (y - y_mean) / y_std if y_std not in [0, np.nan] else y - y_mean

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y_z, rcond=None)
    except Exception:
        return None, None, None

    coef_df = pd.DataFrame({
        "Driver": used_names,
        "column": used_cols,
        "Standardized Effect (β)": beta
    })
    coef_df["|β|"] = coef_df["Standardized Effect (β)"].abs()
    coef_df = coef_df.sort_values("|β|", ascending=False).reset_index(drop=True)
    return coef_df, temp, used_cols

cur = kpi_block(curm)
pre = kpi_block(prevm)

delta_purch  = None if pre["purchases"] in [0, None, np.nan] else cur["purchases"] - pre["purchases"]
delta_rating = None if pd.isna(pre["rating"]) else cur["rating"] - pre["rating"]
delta_disc   = None if pd.isna(pre["disc"]) else cur["disc"] - pre["disc"]

# create tabs for each audience to keep the story clean and focused
# ------------------------------------------------------------
# TABS LAYOUT
# ------------------------------------------------------------
tabs = st.tabs([
    "How To Use",
    "Imputation & Modeling",
    "Executive Snapshot",
    "Finance & Ops",
    "Marketing & Content Strategy",
    "Product & Pricing",
    "Plans & Accountability",
])

# ---------------------- TAB 0: HOW TO USE ----------------------
with tabs[0]:
    st.subheader("Welcome")
    st.write(
        "This hub helps executives and teams understand performance without reading code. "
        "Use the left sidebar to pick a month and narrow to a few categories. "
        "Each tab shows clear visuals and short text that explains what the chart means. "
        "Green, yellow, and red result boxes call out what is going well or needs attention."
    )

    st.subheader("Project Goal")
    st.write(
        "The goal is to give a repeatable, plain-language view of performance for Amazon-style product sales. "
        "Executives get a fast summary. Teams get more detailed views by topic like finance, marketing, and product. "
        "The same pattern can be reused for other companies without changing the logic too much."
    )

    st.subheader("What This App Does")
    st.write(
        "• Combines product-level purchases with ratings, reviews, and discount information.  \n"
        "• Shows where volume concentrates and how discounts are used by category.  \n"
        "• Uses review language to suggest future content and customer education.  \n"
        "• Turns insights into a simple action list with owners and due dates.  \n"
        "• Adds an imputation and modeling tab so the data science work is visible, not hidden."
    )

    st.subheader("Key Terms (Plain Language)")
    st.write(
        "**Purchases** – Count of units bought in the selected month.  \n"
        "**Discount %** – Average percent off the original list price.  \n"
        "**Rating (★)** – Average product star rating from reviews.  \n"
        "**Total Reviews** – How many reviews a product or category has.  \n"
        "**Imputation** – A simple way to fill missing numbers (for charts only) so shapes look stable.  \n"
        "**Regression model** – A basic model that estimates how strongly each driver (rating, discount, reviews) "
        "relates to purchases when they move together."
    )

    st.subheader("Datasets")
    ds_rows = []
    ds_rows.append([
        "amazon_sales_with_two_months.csv",
        len(sales),
        "Required",
        "Sales by product and category with month, rating, review counts, and discount fields."
    ])
    if reviews is not None:
        ds_rows.append([
            "amazon_reviews.csv",
            len(reviews),
            "Optional",
            "Text reviews, ratings, and sentiment for keyword and content ideas."
        ])
    else:
        ds_rows.append([
            "amazon_reviews.csv",
            0,
            "Optional",
            "Not loaded. Add this file to unlock keyword and sentiment views."
        ])
    if catalog is not None:
        ds_rows.append([
            "amazon_products.csv",
            len(catalog),
            "Optional",
            "Extra product metadata such as titles, categories, and prices."
        ])
    else:
        ds_rows.append([
            "amazon_products.csv",
            0,
            "Optional",
            "Not loaded."
        ])
    st.dataframe(
        pd.DataFrame(ds_rows, columns=["File", "Rows", "Role", "What It Adds"]),
        use_container_width=True
    )

    st.subheader("Data Quality At A Glance")
    def null_table(df, name):
        if df is None or df.empty:
            return pd.DataFrame({"column": [], "missing_count": [], "missing_%": []})
        t = df.isna().sum().reset_index()
        t.columns = ["column", "missing_count"]
        t["missing_%"] = (100 * t["missing_count"] / len(df)).round(1)
        t = t[t["missing_count"] > 0].sort_values("missing_%", ascending=False)
        t.insert(0, "dataset", name)
        return t

    dq = pd.concat([
        null_table(sales, "sales"),
        null_table(reviews, "reviews"),
        null_table(catalog, "catalog")
    ], ignore_index=True)
    if dq.empty:
        st.write("No missing values detected in the loaded columns.")
    else:
        st.dataframe(dq, use_container_width=True)

    st.subheader("How Missing Values Are Handled")
    st.write(
        "• Core KPIs (totals, averages, and action rules) use the raw data with missing values left as they are.  \n"
        "• For charts only, you can turn on a simple median-by-category imputation using the sidebar toggle.  \n"
        "• The Imputation & Modeling tab shows a direct before-and-after comparison using the real data."
    )

    st.subheader("How To Use This Dashboard")
    st.write(
        "1. Start with **Executive Snapshot** to see the key metrics and top categories for the selected month.  \n"
        "2. Open **Finance & Ops**, **Marketing**, and **Product & Pricing** to see deeper views for each team.  \n"
        "3. Use **Imputation & Modeling** if you want to understand the data science steps behind the charts.  \n"
        "4. Finish with **Plans & Accountability** and download the CSV so owners have clear next steps."
    )

    st.write(
        "_If a chart is empty, widen the category selection or pick another month to get a fuller picture._"
    )


# ------------- TAB 1: IMPUTATION & MODELING --------------------
with tabs[1]:
    st.header("Imputation & Modeling")
    st.caption("🧮 How missing values are filled for charts and what the regression model says about purchases.")

    view = st.radio(
        "Choose a view",
        ["Imputation", "Modeling"],
        horizontal=True
    )

    # ---------- IMPUTATION VIEW ----------
    if view == "Imputation":
        st.subheader("Imputation Approach (Charts Only)")
        st.write(
            "I used a simple median-based strategy so that charts stay stable when some numeric values are missing.  \n"
            "This does not change the raw KPIs or the action logic. It only affects copies of the data that feed charts."
        )
        st.write(
            "**Steps:**  \n"
            "1. Group rows by `Category_std`.  \n"
            "2. For each numeric column, fill missing values with the median within that category.  \n"
            "3. If an entire category is missing for a column, fill with the overall median for that column.  \n"
            "4. Use these imputed copies only for visualizations like box plots and scatter plots."
        )

        st.subheader("Example: Raw vs Imputed Median Discount by Category")
        raw = curm.copy()
        imp = impute_by_category(curm)

        if guard(raw, "No rows available for this month to illustrate imputation. Try another month or more categories."):
            raw_med = (
                raw[["Category_std", "discount_percentage"]]
                .dropna()
                .groupby("Category_std")["discount_percentage"]
                .median()
                .reset_index()
                .rename(columns={"discount_percentage": "Raw median discount %"})
            )

            imp_med = (
                imp[["Category_std", "discount_percentage"]]
                .dropna()
                .groupby("Category_std")["discount_percentage"]
                .median()
                .reset_index()
                .rename(columns={"discount_percentage": "Imputed median discount %"})
            )

            merged = pd.merge(raw_med, imp_med, on="Category_std", how="inner")
            if guard(merged, "No overlap in categories to compare raw and imputed medians."):
                long = merged.melt(
                    id_vars="Category_std",
                    value_vars=["Raw median discount %", "Imputed median discount %"],
                    var_name="Version",
                    value_name="Median discount %"
                )
                fig_imp = px.bar(
                    long.sort_values("Median discount %", ascending=False).head(30),
                    x="Category_std",
                    y="Median discount %",
                    color="Version",
                    barmode="group",
                    title="Raw vs Imputed Median Discount % (Current Month)"
                )
                fig_imp.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Median discount %",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                st.caption(
                    "This comparison shows how much the median discount for each category shifts "
                    "once missing values are filled using medians."
                )

        st.write(
            "_Imputation is only used for visual stability. The KPIs and action rules still rely on the raw data._"
        )

    # ---------- MODELING VIEW ----------
    else:
        st.subheader("Why Use Linear Regression Here?")
        st.write(
            "Beyond descriptive charts, I wanted one simple model that ranks which levers relate most to purchases. "
            "I used a linear regression on standardized variables so coefficients can be compared directly."
        )
        st.write(
            "**Model setup:**  \n"
            "• Response: `purchased_last_month` (standardized).  \n"
            "• Predictors: product rating, discount percentage, and total reviews.  \n"
            "• Only rows with complete values for these fields are used.  \n"
            "• All predictors are standardized before fitting so the β values are on the same scale."
        )

        coef_df, reg_data, reg_cols = simple_regression_purchases(sales_f)

        if coef_df is None or coef_df.empty:
            st.write(
                "There are not enough complete rows across the current category selection to fit the model. "
                "Try widening the category filters or using a different month."
            )
        else:
            st.subheader("Standardized Effects (β)")
            st.dataframe(
                coef_df[["Driver", "Standardized Effect (β)"]]
                .style.format({"Standardized Effect (β)": "{:.2f}"}),
                use_container_width=True
            )
            st.caption(
                "Larger absolute β values mean that driver moves more with purchases after controlling "
                "for the other fields in the model."
            )

            best = coef_df.iloc[0]
            driver_col = best["column"]
            driver_label = best["Driver"]

            reg_slice = sales_f.loc[reg_data.index].copy()
            if "Category_std" not in reg_slice.columns:
                reg_slice["Category_std"] = "All"

            fig_reg = px.scatter(
                reg_slice,
                x=driver_col,
                y="purchased_last_month",
                color="Category_std",
                opacity=0.7,
                title=f"Purchases vs {driver_label} (All Months, Filtered Categories)"
            )
            fig_reg.update_layout(xaxis_title=driver_label, yaxis_title="Purchases (count)")
            st.plotly_chart(fig_reg, use_container_width=True)
            st.caption(
                "Each dot is a product in the filtered slice. The slope of the cloud plus the β table above "
                "support the same story about which driver is most important."
            )

            desc = (
                f"{best['Driver']} has the strongest modeled link with purchases after controlling for "
                "discounts and review volume. Moves that improve this driver should be tested before "
                "deeper discounts."
            )
            box_result("Model Insight", desc, level="ok", title="Model Insight")


# ---------------- TAB 2: EXECUTIVE SNAPSHOT --------------------
# ---------------- TAB 2: EXECUTIVE SNAPSHOT --------------------
with tabs[2]:
    st.header(f"Executive Snapshot — {month_focus}")
    st.caption("📅 High level view of performance for the selected month.")

    if not guard(curm, "No rows for this month after filters. Pick more categories or another month."):
        st.stop()

    # ==== 1. KPI STRIP (RAW DATA) ==================================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Purchases",
        f"{cur['purchases']:,.0f}",
        None if delta_purch is None else f"{delta_purch:,.0f}"
    )
    c2.metric(
        "Average Rating",
        f"{cur['rating']:.2f}" if pd.notna(cur['rating']) else "n/a",
        None if delta_rating is None else f"{delta_rating:+.2f}"
    )
    c3.metric(
        "Average Discount %",
        f"{cur['disc']:.1f}" if pd.notna(cur['disc']) else "n/a",
        None if delta_disc is None else f"{delta_disc:+.1f}"
    )
    c4.metric(
        "Total Reviews",
        f"{cur['reviews']:,.0f}",
        None
    )

    st.markdown("---")

    # ==== 2. WOW VISUAL: CATEGORY TRENDS OVER TIME =================
    st.subheader("Category Trends Over Time")

    trend_cat = (
        chart_sales_f
        .groupby(["month", "Category_std"], as_index=False)["purchased_last_month"]
        .sum()
        .sort_values("month")
    )

    if guard(trend_cat, "No monthly trend to display for these categories."):
        fig_cat = px.line(
            trend_cat,
            x="month",
            y="purchased_last_month",
            color="Category_std",
            markers=True,
            title="Purchases by Category Across Months"
        )
        fig_cat.update_layout(
            xaxis_title="Month",
            yaxis_title="Purchases (count)"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.caption(
            "Each line is a category. This lets an executive see which categories are gaining or fading over time, "
            "not just the total line."
        )

    # ==== 3. QUICK MoM NARRATIVE ===================================
    if delta_purch is None:
        box_result(
            "Baseline Month",
            f"In {month_focus}, the selected categories recorded {cur['purchases']:,.0f} purchases. "
            "This is the starting point for future comparisons.",
            level="info",
            title="Baseline"
        )
    elif delta_purch > 0:
        box_result(
            "Month-over-Month Growth",
            f"Purchases grew by {delta_purch:,.0f} compared to {prev_month}. "
            "Protect the categories that drove this lift and repeat what worked on their product pages.",
            "good"
        )
    else:
        box_result(
            "Purchases Decreased",
            f"Purchases fell by {abs(delta_purch):,.0f} compared to {prev_month}. "
            "Focus on the categories that lost the most volume and check pricing, images, and returns.",
            "bad"
        )

    st.markdown("---")

    # ==== 4. CATEGORY MIX: TREEMAP (REPLACES TOP-12 BAR) ===========
    st.subheader(f"Where This Month’s Purchases Come From")

    mix = (
        chart_curm
        .groupby("Category_std", as_index=False)["purchased_last_month"]
        .sum()
        .rename(columns={"purchased_last_month": "purchases"})
    )

    if guard(mix, "No category contribution to display."):
        fig_tree = px.treemap(
            mix,
            path=["Category_std"],
            values="purchases",
            title=f"Category Share of Purchases — {month_focus}"
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        st.caption(
            "Each rectangle is a category. Bigger areas mean more purchases this month. "
            "Executives can see the mix at a glance."
        )

        top_cat = mix.sort_values("purchases", ascending=False).iloc[0]
        box_result(
            "Leader Category",
            f"{top_cat['Category_std']} is the largest slice of demand this month. "
            "Keep this category well stocked and featured.",
            "good"
        )

    st.markdown("---")

    # ==== 5. BIGGEST CATEGORY SHIFTS (MoM BAR) =====================
    st.subheader("Biggest Month-over-Month Shifts")

    g = (
        chart_sales_f
        .groupby(["month", "Category_std"])["purchased_last_month"]
        .sum()
        .reset_index()
    )

    if guard(g, "No data for month-over-month category change."):
        last_two = g[g["month"].isin([prev_month, month_focus])]
        if guard(last_two, "Not enough months to compare."):
            pivot = (
                last_two
                .pivot(index="Category_std", columns="month", values="purchased_last_month")
                .fillna(0)
            )
            if prev_month in pivot.columns and month_focus in pivot.columns:
                pivot["delta"] = pivot[month_focus] - pivot[prev_month]
                ranked = pivot.sort_values("delta", ascending=False).reset_index()

                fig_delta = px.bar(
                    ranked,
                    x="Category_std",
                    y="delta",
                    text_auto=".2s",
                    title=f"Change in Purchases by Category ({prev_month} → {month_focus})"
                )
                fig_delta.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Δ Purchases"
                )
                st.plotly_chart(fig_delta, use_container_width=True)
                st.caption(
                    "Positive bars are categories that gained since last month. "
                    "Negative bars are the ones that slipped."
                )

                winner = ranked.iloc[0]
                loser = ranked.iloc[-1]
                st.write(
                    f"**Biggest gain:** {winner['Category_std']} (+{winner['delta']:,.0f}).  \n"
                    f"**Biggest drop:** {loser['Category_std']} ({abs(loser['delta']):,.0f} down)."
                )

    st.markdown("---")

    # ==== 6. NEXT MONTH PLAN — MORE VISIBLE ========================
    st.subheader("Next Month Plan")

    recs = []

    if delta_purch is not None and delta_purch < 0:
        recs.append(
            "Finance & Ops: test bundles and lighter discounts instead of relying on deep cuts in categories that dropped."
        )
    if pd.notna(cur["rating"]) and cur["rating"] < 4.0:
        recs.append(
            "Marketing & Product: add 20–30 second demos and highlight verified reviews for lower-rated categories."
        )
    if pd.notna(cur["disc"]) and cur["disc"] > 25:
        recs.append(
            "Finance & Product: tighten discount bands and improve above-the-fold value bullets on the PDP."
        )
    if not recs:
        recs.append(
            "All teams: keep scaling creator content on top categories and keep PDP copy short, visual, and specific."
        )

    box_result(
        "Agreed Actions For Next Month",
        "• " + "\n• ".join(recs),
        "ok"
    )

# ---------------- TAB 3: FINANCE & OPS -------------------------
with tabs[3]:
    st.header(f"Finance & Operations — {month_focus}")
    st.caption("📊 How volume, discounts, and operations connect for this month.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    # Wow visual: purchases vs average discount (chart copy)
    by_cat = (
        chart_curm.groupby("Category_std", as_index=False)
        .agg(
            purchases=("purchased_last_month", "sum"),
            avg_disc=("discount_percentage", "mean")
        )
    )
    if guard(by_cat, "No category statistics available."):
        fig1 = px.scatter(
            by_cat,
            x="avg_disc",
            y="purchases",
            color="Category_std",
            title="Purchases vs Average Discount by Category"
        )
        fig1.update_layout(
            xaxis_title="Average Discount %",
            yaxis_title="Purchases (count)"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(
            "Each dot is a category. Higher points show more volume, farther right shows heavier discounting."
        )

        lead = by_cat.sort_values("purchases", ascending=False).head(1)
        if not lead.empty:
            r = lead.iloc[0]
            level = "good" if r["avg_disc"] < 25 else "ok"
            box_result(
                "Top Volume Driver",
                f"{r['Category_std']} drives the most purchases with an average discount of {r['avg_disc']:.1f}%.",
                level
            )

    # Discount distribution (chart copy)
    if "discount_percentage" in chart_curm.columns and chart_curm["discount_percentage"].notna().any():
        fig2 = px.box(
            chart_curm.dropna(subset=["discount_percentage"]),
            x="Category_std",
            y="discount_percentage",
            color="Category_std",
            title="Discount % Distribution by Category"
        )
        fig2.update_layout(xaxis_title="", yaxis_title="Discount %")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Boxes show how discount levels spread within each category. High medians or long upper tails signal promo dependence."
        )
        tone = "bad" if (pd.notna(cur["disc"]) and cur["disc"] >= 30) else "ok"
        box_result(
            "Discount Posture",
            f"Average discount this month is {cur['disc']:.1f}%." if pd.notna(cur["disc"]) else "Average discount not available.",
            tone
        )

    st.subheader("Operations Checklist")
    st.write(
        "• Check stock and shipping for the top gaining categories so they do not stock out.  \n"
        "• Review return reasons for any category with high discounts but weak volume.  \n"
        "• Confirm the delivery promise on product pages matches true delivery time."
    )


# ------------- TAB 4: MARKETING & CONTENT ----------------------
with tabs[4]:
    st.header(f"Marketing & Content Strategy — {month_focus}")
    st.caption("🎬 How ratings, reviews, and content support conversion.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    st.subheader("Focus For This Month")
    st.write(
        "The goal is to see which categories convince people without heavy discounts, "
        "which topics block conversion, and what content to ship on each platform."
    )

    # Wow visual: engagement map (chart copy)
    plot_df = chart_curm.dropna(subset=["product_rating"])
    if guard(plot_df, "No rating values to visualize."):
        sizes = safe_bubble_sizes(plot_df.get("purchased_last_month"))
        if sizes is None:
            figm1 = px.scatter(
                plot_df,
                x="product_rating",
                y="total_reviews",
                color="Category_std",
                opacity=0.9,
                title="Engagement Map — Rating vs Reviews"
            )
        else:
            figm1 = px.scatter(
                plot_df,
                x="product_rating",
                y="total_reviews",
                color="Category_std",
                size=sizes,
                opacity=0.9,
                title="Engagement Map — Rating vs Reviews (Bubble = Purchases)"
            )
        figm1.update_layout(
            xaxis_title="Average Rating (★)",
            yaxis_title="Review Volume"
        )
        st.plotly_chart(figm1, use_container_width=True)
        st.caption(
            "Dots with high rating and many reviews are strong advocacy signals. Larger bubbles mark heavier volume."
        )

    # Conversion proxy (chart copy)
    conv = (
        chart_curm.groupby("Category_std", as_index=False)
        .agg(
            purchases=("purchased_last_month", "sum"),
            reviews=("total_reviews", "sum")
        )
    )
    if guard(conv, "No review totals available for conversion proxy."):
        conv["purch_per_review"] = conv.apply(
            lambda r: r["purchases"] / r["reviews"] if r["reviews"] not in [0, np.nan] else np.nan,
            axis=1
        )
        conv = (
            conv.dropna(subset=["purch_per_review"])
            .sort_values("purch_per_review", ascending=False)
            .head(10)
        )
        if guard(conv, "No conversion proxy to show."):
            figc = px.bar(
                conv,
                x="Category_std",
                y="purch_per_review",
                text_auto=".2f",
                title="Purchases Per Review (Conversion Proxy)"
            )
            figc.update_layout(
                xaxis_title="Category",
                yaxis_title="Purchases / Review"
            )
            st.plotly_chart(figc, use_container_width=True)
            st.caption(
                "Higher bars mean each review converts into more purchases. These categories are strong bets for paid and organic placement."
            )

    # Keyword-style view if reviews exist (raw reviews)
    if reviews is not None and "review_text" in reviews.columns:
        st.subheader("Common Customer Words")
        rcur = reviews.copy()
        if "month" in rcur.columns:
            rcur = rcur[rcur["month"] == month_focus]
        if "Category_std" in rcur.columns and cat_sel:
            rcur = rcur[rcur["Category_std"].isin(cat_sel)]

        if guard(rcur, "No reviews available for this slice."):
            low = rcur["review_text"].astype(str).str.lower()
            words = pd.Series(" ".join(low.tolist()))
            words = words.str.replace(r"[^a-z\s]", "", regex=True).str.split()
            words = pd.Series([w for lst in words.dropna().tolist() for w in lst])
            stop = set(
                "the and to a of is it for in on my this that very too but not as be are you i so at have its "
                "they we just would".split()
            )
            topw = words[~words.isin(stop)].value_counts().head(12).reset_index()
            topw.columns = ["word", "count"]
            if guard(topw, "No frequent words to display."):
                figw = px.bar(
                    topw,
                    x="word",
                    y="count",
                    text_auto=".2s",
                    title="Most Common Words in Reviews (This Slice)"
                )
                st.plotly_chart(figw, use_container_width=True)
                st.caption(
                    "These words hint at what customers care about most. Use them as topics for short videos, FAQs, and product page copy."
                )

    # Platform mix suggestion (chart copy)
    st.subheader("Suggested Platform Mix by Category")
    plat = (
        chart_curm.groupby("Category_std", as_index=False)
        .agg(
            rating=("product_rating", "mean"),
            reviews=("total_reviews", "sum"),
            price=("discounted_price", "median")
        )
    )
    if guard(plat, "Not enough fields to build a platform plan."):
        def platform_row(r):
            tiktok = 0.0
            reels = 0.0
            youtube = 0.0
            paid = 0.0
            if pd.notna(r["rating"]) and pd.notna(r["reviews"]):
                if r["rating"] >= 4.2 and r["reviews"] >= np.nanpercentile(plat["reviews"], 60):
                    tiktok += 0.35
                    reels += 0.35
                if (r["rating"] < 4.2 and pd.notna(r["price"])
                        and r["price"] >= np.nanmedian(plat["price"])):
                    youtube += 0.4
                if r["reviews"] < np.nanpercentile(plat["reviews"], 40) and r["rating"] < 4.1:
                    paid += 0.4
            s = tiktok + reels + youtube + paid
            if s == 0:
                s = 1.0
            return pd.Series({
                "TikTok": round(100 * tiktok / s, 1),
                "IG Reels": round(100 * reels / s, 1),
                "YouTube": round(100 * youtube / s, 1),
                "Paid Search": round(100 * paid / s, 1),
            })

        mix = plat.join(plat.apply(platform_row, axis=1))
        melt = mix.melt(
            id_vars=["Category_std"],
            value_vars=["TikTok", "IG Reels", "YouTube", "Paid Search"],
            var_name="Platform",
            value_name="Share %"
        )
        figpm = px.bar(
            melt,
            x="Category_std",
            y="Share %",
            color="Platform",
            barmode="stack",
            title="Recommended Platform Mix by Category"
        )
        st.plotly_chart(figpm, use_container_width=True)
        st.caption(
            "Stacked bars give a simple starting point for spreading effort across TikTok, IG Reels, YouTube, and paid search."
        )

    # Seven-day content plan (keep calendar emojis)
    st.subheader("Seven-Day Content Plan")
    top_cats = (
        chart_curm.groupby("Category_std")["purchased_last_month"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    if not top_cats:
        top_cats = ["Top Category"]

    cal = [
        f"📅 Day 1 — Creator short (15–30s): “Why {top_cats[0]} solves a real problem.” Show it in use.",
        f"📅 Day 2 — Product page demo: unbox {top_cats[0]} and cover the top three benefits in under 30 seconds.",
        f"📅 Day 3 — Social proof reel: put the best review on screen over product footage.",
        f"📅 Day 4 — Quick fix: answer the most common question from reviews in one short clip.",
        f"📅 Day 5 — Compare: {top_cats[0]} vs {top_cats[1] if len(top_cats)>1 else 'another option'} — who is each for.",
        f"📅 Day 6 — Daily use montage: three quick shots that show real life use cases.",
        f"📅 Day 7 — Community Q&A: answer two real customer questions in 30 seconds each.",
    ]
    st.write("\n".join([f"- {row}" for row in cal]))
    st.caption("Post on TikTok and IG Reels, then reuse the strongest clips on the product page and in ads.")


# ------------- TAB 5: PRODUCT & PRICING ------------------------
with tabs[5]:
    st.header(f"Product & Pricing — {month_focus}")
    st.caption("🧪 Quality, price ladders, and which products need the most help.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    # Wow visual: ratings by category (chart copy)
    if "product_rating" in chart_curm.columns and chart_curm["product_rating"].notna().any():
        figp1 = px.box(
            chart_curm.dropna(subset=["product_rating"]),
            x="Category_std",
            y="product_rating",
            color="Category_std",
            title="Product Ratings by Category"
        )
        figp1.update_layout(xaxis_title="", yaxis_title="Rating (★)")
        st.plotly_chart(figp1, use_container_width=True)
        st.caption(
            "Boxes show the rating range within each category. Lower medians hint at friction or mismatched expectations."
        )
        worst = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
        if not worst.empty:
            wcat, wr = worst.index[0], worst.values[0]
            level = "bad" if wr < 4.0 else "ok"
            box_result(
                "Lowest Average Rating",
                f"{wcat} averages {wr:.2f} stars. This is a priority category for content fixes and expectation setting.",
                level
            )

    # Price ladder (chart copy)
    if {"original_price", "discounted_price"}.issubset(chart_curm.columns):
        ladder = (
            chart_curm[["Category_std", "original_price", "discounted_price"]]
            .dropna()
            .groupby("Category_std")
            .median()
            .reset_index()
        )
        if guard(ladder, "No price ladder to show."):
            figp2 = px.scatter(
                ladder,
                x="original_price",
                y="discounted_price",
                color="Category_std",
                title="Price Ladder (Median Original vs Discounted Price)"
            )
            figp2.update_layout(
                xaxis_title="Original Price (median)",
                yaxis_title="Discounted Price (median)"
            )
            st.plotly_chart(figp2, use_container_width=True)
            st.caption(
                "Dots far below the diagonal represent categories that lean heavily on discounts to move units."
            )

            heavy = ladder[ladder["discounted_price"] <= 0.7 * ladder["original_price"]]
            tone = "ok" if heavy.empty else "bad"
            msg = (
                "Median discounted prices sit close to list. Pricing looks healthy."
                if heavy.empty else
                "Some categories rely on deep discounts. Consider testing lighter promos with stronger value copy."
            )
            box_result("Pricing Posture", msg, tone)

    # Underperformers table (raw data)
    under = (
        curm.assign(
            rating=curm["product_rating"],
            purch=curm["purchased_last_month"]
        )
        .dropna(subset=["rating", "purch"])
    )
    if not under.empty:
        under = under.sort_values(["rating", "purch"], ascending=[True, False])
        under = under.drop_duplicates(subset=["product_title"], keep="first").head(10)
        view = under[[
            "product_title",
            "Category_std",
            "rating",
            "purch",
            "discount_percentage"
        ]]
        st.subheader("Underperformers Needing Attention")
        st.dataframe(
            view.rename(columns={
                "product_title": "Product",
                "Category_std": "Category",
                "rating": "Rating (★)",
                "purch": "Purchases",
                "discount_percentage": "Discount %",
            }),
            use_container_width=True
        )
        st.write(
            "Start with low-rating, high-volume items. Improve images, clarify sizing or features, "
            "and use creator content to reset expectations."
        )


# ------------- TAB 6: PLANS & ACCOUNTABILITY -------------------
with tabs[6]:
    st.header("Plans & Accountability")
    st.caption("🗂️ Concrete followups with teams, owners, and suggested due dates.")

    actions = []

    # Category with largest month over month move (raw sales_f)
    g = (
        sales_f.groupby(["month", "Category_std"])["purchased_last_month"]
        .sum()
        .reset_index()
        .sort_values("month")
    )
    if not g.empty and prev_month in g["month"].values and month_focus in g["month"].values:
        pm = g[g["month"] == prev_month].set_index("Category_std")["purchased_last_month"]
        cm = g[g["month"] == month_focus].set_index("Category_std")["purchased_last_month"]
        both = pd.concat([pm, cm], axis=1)
        both.columns = ["prev", "cur"]
        both["delta"] = both["cur"] - both["prev"]
        both = both.dropna()
        if not both.empty:
            top = both["delta"].abs().sort_values(ascending=False).head(1)
            cat = top.index[0]
            d = both.loc[cat, "delta"]
            direction = "up" if d > 0 else "down"
            actions.append(dict(
                team="Executive",
                action=f"{cat} moved {direction} by {abs(d):,.0f} month over month. Set a numeric target and an owner.",
                priority="High",
                due=(date.today() + timedelta(days=7)).isoformat(),
                status="Open"
            ))

    # Review momentum (raw totals)
    if cur["reviews"] and pre["reviews"] and cur["reviews"] < pre["reviews"]:
        actions.append(dict(
            team="Marketing & Content",
            action="Review volume slipped. Refresh creatives, seed UGC, and highlight top reviews on key PDPs.",
            priority="High",
            due=(date.today() + timedelta(days=10)).isoformat(),
            status="Open"
        ))

    # Heavy discounts (raw)
    if pd.notna(cur["disc"]) and cur["disc"] >= 30:
        actions.append(dict(
            team="Finance & Ops",
            action=f"Average discount is {cur['disc']:.1f}%. Test lighter discounts with bundles and clearer value copy.",
            priority="Medium",
            due=(date.today() + timedelta(days=14)).isoformat(),
            status="Open"
        ))

    # Low rating category (raw)
    lowcat = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
    if not lowcat.empty and lowcat.values[0] < 4.0:
        actions.append(dict(
            team="Product & Pricing",
            action=f"{lowcat.index[0]} averages {lowcat.values[0]:.2f}★. Add demos, clarify sizing and care, and update bullets.",
            priority="Medium",
            due=(date.today() + timedelta(days=14)).isoformat(),
            status="Open"
        ))

    # Always include a share step
    actions.append(dict(
        team="Cross-Team",
        action="Record a 10-minute Loom walkthrough of this month's dashboard and share with key stakeholders.",
        priority="Low",
        due=(date.today() + timedelta(days=5)).isoformat(),
        status="Open"
    ))

    if actions:
        act_df = pd.DataFrame(actions, columns=["team", "action", "priority", "due", "status"])
        st.dataframe(act_df, use_container_width=True)
        st.download_button(
            "Download Action Plan (CSV)",
            act_df.to_csv(index=False).encode("utf-8"),
            file_name="action_plan.csv",
            mime="text/csv"
        )
        st.write(
            "_Use this table as a living plan. Assign owners, update statuses, and rerun the dashboard each month to track progress._"
        )
    else:
        st.write("No actions generated for this slice. Try a different month or a wider category selection.")
