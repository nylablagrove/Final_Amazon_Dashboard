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
tabs = st.tabs([
    "How To Use",
    "Imputation & Modeling",
    "Executive Snapshot",
    "Finance & Ops",
    "Marketing & Content Strategy",
    "Product & Pricing",
    "Plans & Accountability",
])

# --------------- TAB 0: HOW TO USE -----------------
with tabs[0]:
    st.subheader("Welcome")
    st.write(
        "This hub helps executives and teams understand performance without reading code. "
        "Use the left sidebar to pick a month and narrow to a few categories. "
        "Each tab shows clear visuals with short captions in plain language. "
        "Green, yellow, and red callouts are reserved for key results only."
    )

    st.subheader("Project Goal")
    st.write(
        "The goal is a shared snapshot that combines sales activity, ratings, and review signals. "
        "Executives get a one-page story for the month. Teams get deeper views with the why behind the numbers "
        "and a concrete plan. This format is reusable because it avoids jargon and uses plain language."
    )

    st.subheader("What This App Does")
    st.write(
        "• Merges product-level activity by month with quality signals like ratings and review volume.  \n"
        "• Highlights where volume concentrates and how discounts are used.  \n"
        "• Uses a simple regression to understand what most relates to purchases.  \n"
        "• Converts findings into a short action list with owners and due dates."
    )

    st.subheader("Why I Chose This Project")
    st.write(
        "I want to work on the business side where communication matters. "
        "Many dashboards show numbers but do not tell people what to do next. "
        "This app forces a full data science workflow from cleaning to explanation to action, "
        "and it teaches non-technical teammates how to use data without reading code."
    )

    st.subheader("Datasets Used")
    ds_rows = []
    ds_rows.append([
        "amazon_sales_with_two_months.csv",
        len(sales),
        "Required",
        "Sales by product and category with month, rating, reviews, and discount fields."
    ])
    if reviews is not None:
        ds_rows.append([
            "amazon_reviews.csv",
            len(reviews),
            "Optional",
            "Text reviews, ratings, and sentiment used for keywords and content planning."
        ])
    else:
        ds_rows.append([
            "amazon_reviews.csv",
            0,
            "Optional",
            "Not loaded. Add to enable keyword insights and content briefs."
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
    st.dataframe(pd.DataFrame(ds_rows, columns=["File", "Rows", "Role", "What It Adds"]), use_container_width=True)

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

    st.markdown(
        "_Missing values are left raw for KPIs so we do not hide problems. "
        "For charts only, you can optionally see the effect of a simple median-by-category imputation in the next tab._"
    )

    st.subheader("Key Terms (Plain English)")
    st.write(
        "**KPI (Key Performance Indicator)**: A metric that tells you quickly if things are going well or not.  \n"
        "**MoM (Month over Month)**: Comparing this month to the previous month.  \n"
        "**Imputation**: A simple way to fill missing values, for example using the median for that category.  \n"
        "**Regression (Linear Regression)**: A basic model that estimates how much each driver (rating, discount, reviews) "
        "relates to purchases when they move together.  \n"
        "**Conversion**: Roughly how well interest turns into purchases. Here we approximate it with purchases per review."
    )

    st.subheader("How To Move Through The Dashboard")
    st.write(
        "1. **Imputation & Modeling**: See how missing data is handled for charts and what the regression model says.  \n"
        "2. **Executive Snapshot**: Get KPIs, a treemap overview, and category gains or losses.  \n"
        "3. **Finance & Ops**: Check discount posture and volume drivers.  \n"
        "4. **Marketing & Content Strategy**: See engagement patterns, conversion proxy, and content ideas.  \n"
        "5. **Product & Pricing**: Look at ratings, price ladders, and underperformers.  \n"
        "6. **Plans & Accountability**: Download the cross-team action list."
    )

    box_warn(
        "Filter Reminder",
        "If you filter down to very few categories, some charts may be empty. Widen the selection if you see a notice about missing data."
    )


# --------------- TAB 1: IMPUTATION & MODELING -----------------
with tabs[1]:
    st.header("Imputation & Modeling")
    st.caption("🧮 How missing values are handled for charts and what the regression model says about purchases.")

    st.subheader("Imputation Approach (Charts Only)")
    st.write(
        "For core KPIs we keep the data raw. We do not fill missing values when computing total purchases or average rating.  \n"
        "For charts only, we support a median-by-category imputation. This means:  \n"
        "• For each numeric column, missing values are filled with the median of that category.  \n"
        "• If a category has all missing values for a column, we fall back to the overall median.  \n"
        "This keeps shapes of charts stable without inventing extreme values."
    )

    # small interactive comparison: raw vs imputed median discount by category
    st.subheader("Example: Raw vs Imputed Discount by Category")
    example_choice = st.radio(
        "View",
        ["Raw medians (no imputation)", "Imputed medians (median by category)"],
        index=0,
        horizontal=True
    )

    # base frame for example (current month)
    base_raw = curm.copy()
    base_imp = chart_curm.copy()

    if not guard(base_raw, "No rows available for this month to illustrate imputation. Try another month or more categories."):
        pass
    else:
        if example_choice.startswith("Raw"):
            ex = (base_raw[["Category_std", "discount_percentage"]]
                  .dropna()
                  .groupby("Category_std")["discount_percentage"]
                  .median()
                  .reset_index())
            ex_title = "Raw Median Discount % by Category (Current Month)"
            ex_caption = "_Each bar shows the median discount without filling missing values._"
        else:
            ex = (base_imp[["Category_std", "discount_percentage"]]
                  .dropna()
                  .groupby("Category_std")["discount_percentage"]
                  .median()
                  .reset_index())
            ex_title = "Imputed Median Discount % by Category (Current Month)"
            ex_caption = (
                "_Each bar shows the median discount after filling missing numeric values "
                "within each category using the median._"
            )

        if guard(ex, "No discount data to show."):
            fig_ex = px.bar(
                ex.sort_values("discount_percentage", ascending=False).head(15),
                x="Category_std",
                y="discount_percentage",
                text_auto=".1f",
                title=ex_title
            )
            fig_ex.update_layout(xaxis_title="Category", yaxis_title="Median Discount %")
            st.plotly_chart(fig_ex, use_container_width=True)
            st.markdown(ex_caption)

    st.markdown(
        "_Note: Turning on or off imputation only affects charts. KPI tiles and action rules always use raw data._"
    )

    st.subheader("Simple Linear Regression: What Drives Purchases?")
    st.write(
        "To move beyond descriptive charts, I fit a simple linear regression model using only the core sales file.  \n"
        "The goal is not a perfect prediction engine. The goal is to rank basic drivers: rating, discount percentage, "
        "and review volume, and see which one most strongly relates to purchases when they move together."
    )

    coef_df, reg_data, reg_cols = simple_regression_purchases(sales_f)

    if coef_df is None or coef_df.empty:
        box_info(
            "Model Not Available",
            "There are not enough complete rows across these filtered categories to fit a stable regression. "
            "Widen the category filters or use more data to see driver insights."
        )
    else:
        st.markdown("#### Driver Strengths (Standardized Effects)")
        coef_show = coef_df[["Driver", "Standardized Effect (β)", "|β|"]].copy()
        st.dataframe(
            coef_show.style.format({"Standardized Effect (β)": "{:.2f}", "|β|": "{:.2f}"}),
            use_container_width=True
        )

        # wow visualization for the model: bar chart of standardized coefficients
        fig_coef = px.bar(
            coef_df,
            x="Driver",
            y="Standardized Effect (β)",
            text_auto=".2f",
            title="Standardized Effect of Each Driver on Purchases"
        )
        fig_coef.update_layout(xaxis_title="Driver", yaxis_title="Standardized Effect (β)")
        st.plotly_chart(fig_coef, use_container_width=True)
        st.markdown(
            "_Bars above zero mean higher values of that driver tend to pair with more purchases in this slice. "
            "Bars below zero mean the opposite._"
        )

        best = coef_df.iloc[0]
        st.markdown(
            f"**Key takeaway:** In this filtered view, **{best['Driver']}** has the strongest modeled link with purchases. "
            "That means moves that improve this driver should usually come before deeper discounts."
        )

        # scatter for strongest driver vs purchases
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
        st.markdown(
            "_The scatter plot shows the raw relationship between the strongest driver and purchases for the filtered categories._"
        )


# --------------- TAB 2: EXECUTIVE SNAPSHOT -----------------
with tabs[2]:
    st.header(f"Executive Snapshot — {month_focus}")
    st.caption("📅 High-level KPIs, a visual overview of the month, and what to do next.")

    if not guard(curm, "No rows for this month after filters. Pick more categories or another month."):
        st.stop()

    # KPI row (kept as the “stock-like” look)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Purchases", f"{cur['purchases']:,.0f}", None if delta_purch is None else f"{delta_purch:,.0f}")
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
    c4.metric("Total Reviews", f"{cur['reviews']:,.0f}")

    # hero visualization: treemap of purchases by category
    st.subheader("Where Purchases Concentrate")
    treemap = (
        curm.groupby("Category_std", as_index=False)["purchased_last_month"]
        .sum()
        .sort_values("purchased_last_month", ascending=False)
    )
    if guard(treemap, "No category data for this month."):
        fig_t = px.treemap(
            treemap,
            path=["Category_std"],
            values="purchased_last_month",
            title=f"Purchase Mix by Category — {month_focus}"
        )
        st.plotly_chart(fig_t, use_container_width=True)
        st.markdown(
            "_The treemap shows which categories carry the most volume this month. Larger boxes mean more purchases._"
        )

    # monthly trend line (chart copy)
    st.subheader("Trend Across Months")
    trend = (chart_sales_f.groupby("month", as_index=False)["purchased_last_month"]
             .sum()
             .sort_values("month"))
    if guard(trend, "No monthly trend to display."):
        fig = px.line(trend, x="month", y="purchased_last_month", markers=True,
                      title="Total Purchases Across Months")
        fig.update_layout(xaxis_title="Month", yaxis_title="Purchases (count)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "_This line shows whether the selected categories are growing, flat, or slipping over time._"
        )

    # narrative for total purchases movement
    if delta_purch is None:
        box_info(
            "Baseline Month",
            f"In {month_focus}, the selected categories recorded {cur['purchases']:,.0f} purchases. "
            "Use the category and change charts below to see where this volume comes from."
        )
    elif delta_purch > 0:
        box_result(
            "Month-over-Month Growth",
            f"Purchases grew by {delta_purch:,.0f} versus {prev_month}. "
            "The lift likely came from category mix and clearer product pages.",
            "good"
        )
    else:
        box_result(
            "Purchases Decreased",
            f"Purchases fell by {abs(delta_purch):,.0f} versus {prev_month}. "
            "Check category mix and lean on clarity and social proof before deeper discounts.",
            "bad"
        )

    # category bar chart (chart copy)
    st.subheader("Top Categories This Month")
    mix = (
        chart_curm.groupby("Category_std", as_index=False)["purchased_last_month"]
        .sum()
        .sort_values("purchased_last_month", ascending=False)
        .head(12)
    )
    if guard(mix, "No category contribution to display."):
        fig2 = px.bar(mix, x="Category_std", y="purchased_last_month", text_auto=".2s",
                      title=f"Top 12 Categories in {month_focus}")
        fig2.update_layout(xaxis_title="Category", yaxis_title="Purchases")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            "_Bars rank categories by total purchases. This is the quick list of leaders to protect and mid-pack to develop._"
        )
        top_cat = mix.iloc[0]
        box_result(
            "Leader Category",
            f"{top_cat['Category_std']} leads this month. Protect the lead with strong visuals and clear benefits.",
            "good"
        )

    # month-over-month change by category (chart copy)
    st.subheader("Biggest Month-over-Month Movers")
    g = (chart_sales_f.groupby(["month", "Category_std"])["purchased_last_month"].sum().reset_index())
    if guard(g, "No data for month-over-month category change."):
        last_two = g[g["month"].isin([prev_month, month_focus])]
        if guard(last_two, "Not enough months to compare."):
            pivot = last_two.pivot(index="Category_std", columns="month", values="purchased_last_month").fillna(0)
            if prev_month in pivot.columns and month_focus in pivot.columns:
                pivot["delta"] = pivot[month_focus] - pivot[prev_month]
                ranked = pivot.sort_values("delta", ascending=False).reset_index()
                fig3 = px.bar(ranked, x="Category_std", y="delta", text_auto=".2s",
                              title=f"Change in Purchases ({prev_month} → {month_focus})")
                fig3.update_layout(xaxis_title="Category", yaxis_title="Δ Purchases")
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown(
                    "_Bars above zero show categories that grew. Bars below zero show categories that lost ground._"
                )
                winner = ranked.iloc[0]
                loser = ranked.iloc[-1]
                st.markdown(
                    f"**Quick read:** Biggest gain: **{winner['Category_std']}** (+{winner['delta']:,.0f}).  "
                    f"Biggest drop: **{loser['Category_std']}** ({abs(loser['delta']):,.0f} down)."
                )

    # short next month plan
    st.subheader("Next Month Plan (High Level)")
    recs = []
    if delta_purch and delta_purch < 0:
        recs.append("Finance: test lighter discounts with bundles instead of deeper blanket promos.")
    if pd.notna(cur["rating"]) and cur["rating"] < 4.0:
        recs.append("Marketing: add short demo videos and highlight verified reviews at the top of product pages.")
    if pd.notna(cur["disc"]) and cur["disc"] > 25:
        recs.append("Product: strengthen value bullets so pages convert at lighter promo levels.")
    if not recs:
        recs.append("Scale creator content around top categories and keep PDP copy short, visual, and specific.")

    st.markdown("- " + "\n- ".join(recs))
    st.markdown(
        "_For more detail on why purchases move, see the **Imputation & Modeling** tab and the team-specific tabs._"
    )


# --------------- TAB 3: FINANCE & OPS -----------------
with tabs[3]:
    st.header(f"Finance & Operations — {month_focus}")
    st.caption("📊 How volume and discounts interact, plus quick operational checks.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    # hero visualization: purchases vs discount scatter
    st.subheader("Are We Buying Volume With Discounts?")
    by_cat = (
        chart_curm.groupby("Category_std", as_index=False)
        .agg(purchases=("purchased_last_month", "sum"),
             avg_disc=("discount_percentage", "mean"))
    )
    if guard(by_cat, "No category statistics available."):
        fig1 = px.scatter(
            by_cat,
            x="avg_disc",
            y="purchases",
            color="Category_std",
            title="Purchases vs Average Discount by Category"
        )
        fig1.update_layout(xaxis_title="Average Discount %", yaxis_title="Purchases (count)")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(
            "_Each point is a category. If most high-volume points sit far to the right, volume may depend on deep discounts._"
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

    st.subheader("Discount Distribution by Category")
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
        st.markdown(
            "_Boxes show the spread and typical discount level by category. Taller boxes or very high medians hint at promo dependence._"
        )

        tone = "bad" if (pd.notna(cur["disc"]) and cur["disc"] >= 30) else "ok"
        box_result(
            "Discount Posture",
            f"Average discount across selected categories is {cur['disc']:.1f}%. "
            "Use the deepest discounts only for low-rating or overstock items.",
            tone
        )

    st.subheader("Operations Checklist")
    ops = [
        "Check stock and shipping for the top three categories with the largest purchase increases.",
        "Review return reasons for any category that increased discounts without a visible volume lift.",
        "Confirm that delivery promises on product pages match actual average delivery times."
    ]
    st.markdown("- " + "\n- ".join(ops))


# --------------- TAB 4: MARKETING & CONTENT STRATEGY -----------------
with tabs[4]:
    st.header(f"Marketing & Content Strategy — {month_focus}")
    st.caption("🎬 How people react to products and what content to publish.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    box_info(
        "Focus For This Month",
        "Identify categories that convince people without heavy discounts, see which topics block conversion, "
        "and turn review language into content ideas."
    )

    # hero visualization: engagement map bubble chart
    st.subheader("Engagement Map: Rating vs Review Volume")
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
                title="Rating vs Reviews by Product"
            )
        else:
            figm1 = px.scatter(
                plot_df,
                x="product_rating",
                y="total_reviews",
                color="Category_std",
                size=sizes,
                opacity=0.9,
                title="Rating vs Reviews by Product (Bubble Size = Purchases)"
            )
        figm1.update_layout(xaxis_title="Average Rating (★)", yaxis_title="Review Volume")
        st.plotly_chart(figm1, use_container_width=True)
        st.markdown(
            "_Dots in the top-right represent products people both love and talk about. Those are strong candidates for creator content._"
        )

    # conversion proxy: purchases per review
    st.subheader("Conversion Proxy: Purchases Per Review")
    conv = (chart_curm.groupby("Category_std", as_index=False)
            .agg(purchases=("purchased_last_month", "sum"),
                 reviews=("total_reviews", "sum")))
    if guard(conv, "No review totals available for conversion proxy."):
        conv["purch_per_review"] = conv.apply(
            lambda r: r["purchases"] / r["reviews"] if r["reviews"] not in [0, np.nan] else np.nan,
            axis=1
        )
        conv = conv.dropna(subset=["purch_per_review"]).sort_values("purch_per_review", ascending=False).head(10)
        if guard(conv, "No conversion proxy to show."):
            figc = px.bar(
                conv,
                x="Category_std",
                y="purch_per_review",
                text_auto=".2f",
                title="Purchases Per Review by Category"
            )
            figc.update_layout(xaxis_title="Category", yaxis_title="Purchases / Review")
            st.plotly_chart(figc, use_container_width=True)
            st.markdown(
                "_Higher bars suggest categories where each review moves more people to buy. Those are good bets for paid and organic placement._"
            )
            lead = conv.iloc[0]
            box_result(
                "High Intent Signal",
                f"{lead['Category_std']} converts reviews to purchases efficiently.",
                "good"
            )

    # simple keyword view from reviews text
    st.subheader("Common Customer Words In Reviews")
    if reviews is not None and "review_text" in reviews.columns:
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
                "the and to a of is it for in on my this that very too but not as be are you i so at have "
                "its they we just would".split()
            )
            topw = words[~words.isin(stop)].value_counts().head(12).reset_index()
            topw.columns = ["word", "count"]
            if guard(topw, "No frequent words to display."):
                figw = px.bar(topw, x="word", y="count", text_auto=".2s",
                              title="Frequent Words in Reviews (This Slice)")
                st.plotly_chart(figw, use_container_width=True)
                st.markdown(
                    "_These words are hints for topics and language to use in short-form videos and product page copy._"
                )
                box_warn(
                    "Interpret Carefully",
                    "Skim a few underlying reviews before using a word in marketing. Some frequent words show complaints."
                )

    # platform mix suggestion
    st.subheader("Suggested Platform Mix By Category")
    plat = (chart_curm.groupby("Category_std", as_index=False)
            .agg(rating=("product_rating", "mean"),
                 reviews=("total_reviews", "sum"),
                 price=("discounted_price", "median")))
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
                if (r["rating"] < 4.2 and pd.notna(r["price"]) and r["price"] >= np.nanmedian(plat["price"])):
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

        mix_pf = plat.join(plat.apply(platform_row, axis=1))
        melt = mix_pf.melt(
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
            title="Recommended Platform Mix By Category"
        )
        st.plotly_chart(figpm, use_container_width=True)
        st.markdown(
            "_Use this as a starting split by platform, then adjust based on real results._"
        )

    # seven day content plan
    st.subheader("Seven-Day Content Plan")
    top_cats = (chart_curm.groupby("Category_std")["purchased_last_month"]
                .sum()
                .sort_values(ascending=False)
                .head(3)
                .index
                .tolist())
    if not top_cats:
        top_cats = ["Top Category"]

    cal = [
        f"Day 1: 15–30 second creator short on why **{top_cats[0]}** solves a real problem.",
        f"Day 2: Product page demo for **{top_cats[0]}** covering the top three benefits.",
        "Day 3: Social proof reel with one strong review on screen over product footage.",
        "Day 4: Quick FAQ clip answering the most common question from reviews.",
        f"Day 5: Compare **{top_cats[0]}** vs {top_cats[1] if len(top_cats) > 1 else 'a close alternative'} and explain who each product is for.",
        "Day 6: Daily use montage with three real-life use cases.",
        "Day 7: Community Q&A with two real customer questions answered in under 30 seconds each."
    ]
    st.markdown("- " + "\n- ".join(cal))
    st.markdown(
        "_Post on TikTok and IG Reels. Pin the best demo to the product page hero video where possible._"
    )


# --------------- TAB 5: PRODUCT & PRICING -----------------
with tabs[5]:
    st.header(f"Product & Pricing — {month_focus}")
    st.caption("🧪 How customers rate products and how far prices step down from list.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    # hero visualization: rating distributions
    st.subheader("Rating Distributions by Category")
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
        st.markdown(
            "_Boxes show the range and typical rating per category. Lower medians signal friction or mismatched expectations._"
        )

        worst = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
        if not worst.empty:
            wcat, wr = worst.index[0], worst.values[0]
            level = "bad" if wr < 4.0 else "ok"
            box_result(
                "Lowest Average Rating",
                f"{wcat} averages {wr:.2f} stars. Clarify expectations and show the product in use.",
                level
            )

    st.subheader("Price Ladder (List vs Discounted)")
    if {"original_price", "discounted_price"}.issubset(chart_curm.columns):
        ladder = (chart_curm[["Category_std", "original_price", "discounted_price"]]
                  .dropna()
                  .groupby("Category_std")
                  .median()
                  .reset_index())
        if guard(ladder, "No price ladder to show."):
            figp2 = px.scatter(
                ladder,
                x="original_price",
                y="discounted_price",
                color="Category_std",
                title="Median Original vs Discounted Price by Category"
            )
            figp2.update_layout(xaxis_title="Original Price (median)", yaxis_title="Discounted Price (median)")
            st.plotly_chart(figp2, use_container_width=True)
            st.markdown(
                "_Dots far below the diagonal implied line (where discounted equals original) suggest heavy discounting._"
            )

            heavy = ladder[ladder["discounted_price"] <= 0.7 * ladder["original_price"]]
            tone = "ok" if heavy.empty else "bad"
            msg = (
                "Median discounted prices sit close to list. Pricing posture looks balanced."
                if heavy.empty else
                "Some categories lean on deep discounts. Test lighter promos with sharper value messages."
            )
            box_result("Pricing Posture", msg, tone)

    st.subheader("Underperformers Needing Attention")
    under = (curm.assign(
        rating=curm["product_rating"],
        purch=curm["purchased_last_month"]
    ).dropna(subset=["rating", "purch"]))
    if not under.empty:
        under = under.sort_values(["rating", "purch"], ascending=[True, False])
        under = under.drop_duplicates(subset=["product_title"], keep="first").head(10)
        view = under[["product_title", "Category_std", "rating", "purch", "discount_percentage"]]
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
        st.markdown(
            "_Low-rating, high-volume products are prime candidates for better onboarding, clearer photos, or revised descriptions._"
        )


# --------------- TAB 6: PLANS & ACCOUNTABILITY -----------------
with tabs[6]:
    st.header("Plans & Accountability")
    st.caption("🗂️ Cross-team next steps tied to the metrics you saw in the other tabs.")

    actions = []

    # largest month over month move
    g = (sales_f.groupby(["month", "Category_std"])["purchased_last_month"].sum()
         .reset_index()
         .sort_values("month"))
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
                action=f"{cat} moved {direction} by {abs(d):,.0f} MoM. Set a numeric target and assign an owner.",
                priority="High",
                due=(date.today() + timedelta(days=7)).isoformat(),
                status="Open"
            ))

    # review momentum
    if cur["reviews"] and pre["reviews"] and cur["reviews"] < pre["reviews"]:
        actions.append(dict(
            team="Marketing & Content",
            action="Review volume slipped. Refresh creatives, seed UGC, and add social proof blocks on PDP.",
            priority="High",
            due=(date.today() + timedelta(days=10)).isoformat(),
            status="Open"
        ))

    # heavy discounts
    if pd.notna(cur["disc"]) and cur["disc"] >= 30:
        actions.append(dict(
            team="Finance & Ops",
            action=f"Average discount is {cur['disc']:.1f}%. Test lighter discounts using bundles and clearer value bullets.",
            priority="Medium",
            due=(date.today() + timedelta(days=14)).isoformat(),
            status="Open"
        ))

    # low category rating
    lowcat = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
    if not lowcat.empty and lowcat.values[0] < 4.0:
        actions.append(dict(
            team="Product & Pricing",
            action=f"{lowcat.index[0]} averages {lowcat.values[0]:.2f}★. Add 20–30 second demos and clarify sizing and care.",
            priority="Medium",
            due=(date.today() + timedelta(days=14)).isoformat(),
            status="Open"
        ))

    # always share recap
    actions.append(dict(
        team="Cross-Team",
        action="Record a 10-minute Loom or meeting walkthrough of this hub and share highlights in the team channel.",
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
        st.markdown(
            "_Use this table as a living list. After export, you can track status in your project tool or spreadsheet._"
        )
    else:
        box_warn("No Actions Generated", "Try a different month or widen categories to reveal more signals.")
