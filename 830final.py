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

# === tabs ===================================================================

tabs = st.tabs([
    "How To Use",
    "Imputation & Modeling",
    "Executive Snapshot",
    "Finance & Ops",
    "Marketing & Content Strategy",
    "Product & Pricing",
    "Plans & Accountability",
])

# === TAB 0: HOW TO USE ======================================================

with tabs[0]:
    st.subheader("Welcome")
    st.write(
        "This hub helps executives and teams understand performance without reading code. "
        "Use the left sidebar to pick a month and narrow to a few categories. "
        "Each tab shows clear visuals, a short caption that explains the chart, and color-coded results boxes. "
        "Green means good, yellow means needs a look, red means fix now. Blue boxes give simple guidance."
    )

    st.subheader("Project Goal")
    st.write(
        "The goal is a shared snapshot that combines sales activity, ratings, and review signals. "
        "Executives get a one-page story for the month. Teams get deeper views with the why behind the numbers and a concrete plan. "
        "This format is reusable across companies and roles because it avoids jargon and uses plain language."
    )

    # key terms come early so the rest of the story is easier to follow
    st.subheader("Key Terms and Acronyms")
    st.write(
        "• **KPI**: Key Performance Indicator. A small number that tells you quickly if things are going well.  \n"
        "• **MoM**: Month over month. Comparing this month to the previous month.  \n"
        "• **Conversion**: When a shopper actually buys after viewing a product.  \n"
        "• **PDP**: Product detail page. The main page for a single item on Amazon.  \n"
        "• **UGC**: User generated content. Posts, reviews, and videos made by real customers, not the brand.  \n"
        "• **Discount posture**: How aggressive our discounts are on average.  \n"
        "• **Platform mix**: How we split content across TikTok, Instagram Reels, YouTube, and paid search.  \n"
        "• **Price ladder**: How the original price and discounted price line up across categories."
    )

    st.subheader("What This App Does")
    st.write(
        "• Merges product-level activity by month with quality signals like ratings and review volume.  \n"
        "• Highlights where volume concentrates and how discounts are used.  \n"
        "• Surfaces review language to inform content ideas and customer education.  \n"
        "• Converts findings into a simple action list with owners and due dates."
    )

    st.subheader("Why I Chose This Project")
    st.write(
        "I want to work on the business side where communication matters. "
        "Many dashboards show numbers but do not tell people what to do next. "
        "This app forces a full data science workflow from cleaning to explanation to action, "
        "and it teaches non-technical teammates how to use data without reading code."
    )

    st.subheader("Datasets")
    ds_rows = []
    ds_rows.append([
        "amazon_sales_with_two_months_small.csv",
        len(sales),
        "Required",
        "Sales by product and category with month, rating, reviews, and discount fields."
    ])
    if reviews is not None:
        ds_rows.append([
            "amazon_reviews_small.csv",
            len(reviews),
            "Optional",
            "Text reviews, ratings, and sentiment used for keywords and content planning."
        ])
    else:
        ds_rows.append([
            "amazon_reviews_small.csv",
            0,
            "Optional",
            "Not loaded. Add to enable keyword insights and content briefs."
        ])
    if catalog is not None:
        ds_rows.append([
            "amazon_products_small.csv",
            len(catalog),
            "Optional",
            "Extra product metadata such as titles, categories, and prices."
        ])
    else:
        ds_rows.append([
            "amazon_products_small.csv",
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

    box_info(
        "How We Handle Missing Data",
        "Core KPIs stay raw so we do not hide problems. For charts only, the app builds a copy of the data where "
        "missing numeric cells are filled with the median inside each Category_std group. "
        "If a slice has no rows after filters, the page shows a friendly message instead of breaking."
    )

    st.subheader("How To Read This Dashboard")
    st.write(
        "• Start at Executive Snapshot for KPIs, a month-to-month trend, category mix, and highlights.  \n"
        "• Open each team tab for deeper context and a short caption under each chart that says why it matters.  \n"
        "• Go to Imputation & Modeling to see how missing data and linear regression are handled.  \n"
        "• Go to Plans & Accountability to download the action list and assign owners."
    )

    box_warn(
        "Caution",
        "If you filter to a very small set of categories, some charts may be empty. Widen the selection to see a fuller picture."
    )

# === TAB 1: IMPUTATION & MODELING ===========================================

with tabs[1]:
    st.header("Imputation & Modeling")
    st.caption("🧩 This tab shows how missing values were handled and how a simple linear regression model was built.")

    # missingness before imputation
    st.subheader("Missing Values Before Imputation")
    if not guard(sales_f, "No data available after filters."):
        st.stop()

    num_cols_raw = sales_f.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols_raw:
        st.write("No numeric columns detected to summarize.")
    else:
        miss = sales_f[num_cols_raw].isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            st.write("There are no missing numeric values in this filtered slice.")
        else:
            miss_df = miss.reset_index()
            miss_df.columns = ["column", "missing_count"]
            miss_df["missing_%"] = (100 * miss_df["missing_count"] / len(sales_f)).round(1)
            fig_miss = px.bar(
                miss_df,
                x="column",
                y="missing_count",
                text_auto=".0f",
                title="Missing Values Per Numeric Column (Before Imputation)"
            )
            fig_miss.update_layout(xaxis_title="Column", yaxis_title="Missing Count")
            st.plotly_chart(fig_miss, use_container_width=True)
            fig_notes(
                why="Document where data is incomplete before modeling or visualization.",
                what="Bars show how many missing values appear in each numeric column.",
                sowhat="This confirms that the dataset is not perfect and motivates imputation.",
                nextstep="Compare these gaps with charts on other tabs, which use an imputed copy of the data for visuals."
            )

    # explanation of the imputation strategy
    st.subheader("Imputation Strategy Used In This App")
    st.write(
        "For charts only, this app uses a simple median imputation by category. "
        "The idea is to fill missing numeric values with typical values inside each product category, "
        "instead of dropping rows or guessing a single global number."
    )
    st.write(
        "The steps are:  \n"
        "1. Group rows by Category_std.  \n"
        "2. For each numeric column, compute the median inside that category.  \n"
        "3. Fill missing values with the category median.  \n"
        "4. If an entire category is missing, fall back to the overall median for that column.  \n"
        "5. Use these filled values only in chart copies. Core KPIs and action rules still use the raw data."
    )

    # before vs after imputation histogram for a chosen column
    st.subheader("Before vs After Imputation Example")
    if num_cols_raw:
        example_col = st.selectbox(
            "Pick a numeric column to compare before and after median-by-category imputation",
            num_cols_raw
        )
        raw_series = sales_f[example_col]
        imp_series = chart_sales_f[example_col] if example_col in chart_sales_f.columns else raw_series

        comp_df = pd.DataFrame({
            "value": pd.concat([raw_series, imp_series], ignore_index=True),
            "Source": (["Raw (missing allowed)"] * len(raw_series)) + (["Imputed for charts"] * len(imp_series))
        }).dropna(subset=["value"])

        if not comp_df.empty:
            fig_hist = px.histogram(
                comp_df,
                x="value",
                color="Source",
                barmode="overlay",
                opacity=0.6,
                title=f"Distribution of {example_col} Before and After Imputation"
            )
            fig_hist.update_layout(xaxis_title=example_col, yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
            fig_notes(
                why="Show how the median-by-category imputation changes the distribution.",
                what="The two overlapping histograms compare the raw values with the imputed values.",
                sowhat="The imputed version fills gaps but keeps the overall shape similar.",
                nextstep="Use the imputed version for charts that require complete numeric inputs."
            )
        else:
            st.write("Not enough non-missing values to draw a comparison histogram.")
    else:
        st.write("There are no numeric columns to demonstrate imputation.")

    # linear regression explanation and results
    st.subheader("Linear Regression Model: What Drives Purchases?")
    st.write(
        "To connect multiple drivers at once, this app uses a simple multiple linear regression model. "
        "The target (y) is purchased_last_month. The predictors (X) are product_rating, "
        "discount_percentage, and total_reviews, wherever these columns are present and complete."
    )
    st.write(
        "All predictors and the target are standardized before fitting, and the model is solved with "
        "numpy.linalg.lstsq, which is ordinary least squares linear regression."
    )

    coef_df, reg_data, reg_cols = simple_regression_purchases(sales_f)
    if coef_df is None or coef_df.empty:
        box_info(
            "Linear Regression Not Available",
            "There are not enough complete rows across these categories to fit a simple multiple linear regression. "
            "Widen the category filters or include more data to see driver effects."
        )
    else:
        st.dataframe(
            coef_df[["Driver", "Standardized Effect (β)"]],
            use_container_width=True
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
            title=f"Purchases vs {driver_label} (Linear Regression Driver View)"
        )
        fig_reg.update_layout(xaxis_title=driver_label, yaxis_title="Purchases (count)")
        st.plotly_chart(fig_reg, use_container_width=True)

        fig_notes(
            why="Summarize how multiple variables relate to purchases at the same time.",
            what="The table shows standardized linear regression coefficients. The chart shows the strongest driver.",
            sowhat="A larger positive coefficient means that driver tends to move with higher purchases when others are held steady.",
            nextstep="Use this ranking to decide whether to push ratings, review volume, or discount clarity first."
        )

        tone = "good" if best["Driver"].startswith("Rating") and best["Standardized Effect (β)"] > 0 else "ok"
        desc = (
            f"{best['Driver']} has the strongest standardized linear relationship with purchases in this slice. "
            "Focus on this driver before relying on deeper discounts."
        )
        box_result("Model Insight From Linear Regression", desc, tone)

    # plain-language wrap for this tab
    box_info(
        "What This Page Is Really Saying",
        "The data is not perfect, so some cells are empty. For charts, the app fills holes with typical values inside each "
        "category so visuals stay stable. On top of that, a linear regression model shows which levers matter most for purchases "
        "across the filtered categories, so we know whether to push ratings, reviews, or discounts first."
    )

# === TAB 2: EXECUTIVE SNAPSHOT ==============================================

with tabs[2]:
    st.header(f"Executive Snapshot — {month_focus}")
    st.caption("📅 Focus for this month, a quick read on the trend, and what to do next.")

    if not guard(curm, "No rows for this month after filters. Pick more categories or another month."):
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Purchases", f"{cur['purchases']:,.0f}", None if delta_purch is None else f"{delta_purch:,.0f}")
    c2.metric("Average Rating", f"{cur['rating']:.2f}" if pd.notna(cur['rating']) else "n/a",
              None if delta_rating is None else f"{delta_rating:+.2f}")
    c3.metric("Average Discount %", f"{cur['disc']:.1f}" if pd.notna(cur['disc']) else "n/a",
              None if delta_disc is None else f"{delta_disc:+.1f}")
    c4.metric("Total Reviews", f"{cur['reviews']:,.0f}")

    trend = (chart_sales_f.groupby("month", as_index=False)["purchased_last_month"]
             .sum().sort_values("month"))
    if guard(trend, "No monthly trend to display."):
        fig = px.line(trend, x="month", y="purchased_last_month", markers=True, title="Purchases Across Months")
        fig.update_layout(xaxis_title="Month", yaxis_title="Purchases (count)")
        st.plotly_chart(fig, use_container_width=True)
        fig_notes(
            why="Show the overall direction month to month.",
            what="Line shows total purchases across the selected categories.",
            sowhat="Confirms if we are growing, flat, or slipping.",
            nextstep="If down, check category deltas and discount posture below."
        )

    if delta_purch is None:
        box_info(
            "Baseline Month",
            f"In {month_focus}, the selected categories recorded {cur['purchases']:,.0f} purchases. "
            "This is the baseline for comparison. Use the Category Contribution to see where performance concentrates."
        )
    elif delta_purch > 0:
        box_result(
            "Month-over-Month Growth",
            f"Purchases grew by {delta_purch:,.0f} versus {prev_month}. "
            "The lift likely came from category mix and clearer product pages. "
            "Keep momentum with creator videos and concise value bullets at the top of each page.",
            "good"
        )
    else:
        box_result(
            "Purchases Decreased",
            f"Purchases fell by {abs(delta_purch):,.0f} versus {prev_month}. "
            "Check category mix and reduce reliance on discounts. Improve page clarity with short demos and verified reviews.",
            "bad"
        )

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
        fig_notes(
            why="Find where purchases concentrate this month.",
            what="Bars rank categories by total purchases.",
            sowhat="Shows the leaders to protect and the mid-pack to develop.",
            nextstep="Keep leaders stocked and featured. Test bundles in the middle group."
        )
        top_cat = mix.iloc[0]
        box_result(
            "Leader Category",
            f"{top_cat['Category_std']} leads this month. Protect the lead with strong visuals and clear benefits.",
            "good"
        )

    g = (chart_sales_f.groupby(["month", "Category_std"])["purchased_last_month"].sum().reset_index())
    if guard(g, "No data for month-over-month category change."):
        last_two = g[g["month"].isin([prev_month, month_focus])]
        if guard(last_two, "Not enough months to compare."):
            pivot = last_two.pivot(index="Category_std", columns="month", values="purchased_last_month").fillna(0)
            if prev_month in pivot.columns and month_focus in pivot.columns:
                pivot["delta"] = pivot[month_focus] - pivot[prev_month]
                ranked = pivot.sort_values("delta", ascending=False).reset_index()
                fig3 = px.bar(ranked, x="Category_std", y="delta", text_auto=".2s",
                              title=f"Month-over-Month Change ({prev_month} → {month_focus})")
                fig3.update_layout(xaxis_title="Category", yaxis_title="Δ Purchases")
                st.plotly_chart(fig3, use_container_width=True)
                fig_notes(
                    why="Spot what changed the most month to month.",
                    what="Bars show increases and decreases by category.",
                    sowhat="Explains the headline KPI move.",
                    nextstep="Double-click the biggest moves in each team tab."
                )
                winner = ranked.iloc[0]; loser = ranked.iloc[-1]
                box_info(
                    "Monthly Highlights",
                    f"📈 Biggest gain: {winner['Category_std']} (+{winner['delta']:,.0f}). "
                    f"📉 Biggest drop: {loser['Category_std']} ({abs(loser['delta']):,.0f} down). "
                    "Review images, copy, and return reasons for the declining category."
                )

    recs = []
    if delta_purch and delta_purch < 0:
        recs.append("🔴 Finance: Reduce discount depth and test bundles with clear benefits.")
    if pd.notna(cur["rating"]) and cur["rating"] < 4.0:
        recs.append("🟡 Marketing: Add 20 to 30 second demo videos and highlight verified reviews.")
    if pd.notna(cur["disc"]) and cur["disc"] > 25:
        recs.append("🟡 Product: Improve above-the-fold value bullets so pages can convert at lighter promos.")
    if not recs:
        recs.append("✅ Keep scaling creator content and keep PDP copy short, visual, and specific.")
    box_result("Next Month Plan", " • " + "\n • ".join(recs), "ok")

    # plain-language wrap for this tab
    if delta_purch is None:
        headline = f"This is the starting month for the filtered categories in {month_focus}."
    elif delta_purch > 0:
        headline = f"Purchases are up this month compared to {prev_month}."
    else:
        headline = f"Purchases are down this month compared to {prev_month}."

    box_info(
        "What This Page Is Really Saying",
        headline
        + " The line chart shows whether momentum is up or down. The bar charts reveal which categories carry the month "
        "and which ones moved the most since last month. The result boxes at the top give a fast read on volume, average "
        "rating, discount posture, and review activity so you know if this month is healthy or needs a course correction."
    )

# === TAB 3: FINANCE & OPS ===================================================

with tabs[3]:
    st.header(f"Finance & Operations — {month_focus}")
    st.caption("📊 This tab focuses on volume drivers, discount posture, and operational guardrails.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    by_cat = (
        chart_curm.groupby("Category_std", as_index=False)
        .agg(purchases=("purchased_last_month", "sum"),
             avg_disc=("discount_percentage", "mean"))
    )
    if guard(by_cat, "No category statistics available."):
        fig1 = px.scatter(by_cat, x="avg_disc", y="purchases", color="Category_std",
                          title="Purchases vs Average Discount by Category")
        fig1.update_layout(xaxis_title="Average Discount %", yaxis_title="Purchases (count)")
        st.plotly_chart(fig1, use_container_width=True)
        fig_notes(
            why="Check if volume depends on deep discounts.",
            what="Each dot is a category with its average discount and purchases.",
            sowhat="If high volume sits at high discount, margin is at risk.",
            nextstep="Move some gains to bundles and sharper product value copy."
        )
        lead = by_cat.sort_values("purchases", ascending=False).head(1)
        if not lead.empty:
            r = lead.iloc[0]
            level = "good" if r["avg_disc"] < 25 else "ok"
            box_result(
                "Volume Driver",
                f"{r['Category_std']} drives the most purchases with an average discount of {r['avg_disc']:.1f}%.",
                level
            )

    if "discount_percentage" in chart_curm.columns and chart_curm["discount_percentage"].notna().any():
        fig2 = px.box(chart_curm.dropna(subset=["discount_percentage"]),
                      x="Category_std", y="discount_percentage", color="Category_std",
                      title="Discount % Distribution by Category")
        fig2.update_layout(xaxis_title="", yaxis_title="Discount %")
        st.plotly_chart(fig2, use_container_width=True)
        fig_notes(
            why="See how discounts are spread within each category.",
            what="Boxes show the range and median discount by category.",
            sowhat="Heavy tails or high medians hint at promo dependence.",
            nextstep="Tighten discount bands and fix clarity on pages before promos."
        )
        tone = "bad" if (pd.notna(cur["disc"]) and cur["disc"] >= 30) else "ok"
        box_result(
            "Discount Posture",
            f"Average discount is {cur['disc']:.1f}%. Use the deepest discounts only for low-rating or overstock items.",
            tone
        )
        if tone == "bad":
            box_warn(
                "Caution On Discount Depth",
                "Heavy average discounts can hide product clarity problems. Fix value messaging before raising promo depth."
            )

    ops = [
        "✅ Check stock and shipping for top 3 categories with the largest increase.",
        "✅ Review return reasons for any category that increased discounts without a volume lift.",
        "✅ Confirm delivery promise on PDP matches actual average delivery time.",
    ]
    box_info("Operations Checklist", " • " + "\n • ".join(ops))

    # plain-language wrap for this tab
    box_info(
        "What This Page Is Really Saying",
        "This page connects money and mechanics. The scatter shows whether big sales only happen when discounts are deep, "
        "which puts margin at risk. The box plot shows how wild or controlled your discounts are inside each category. "
        "If discounts are heavy and spread out, finance and ops should tighten promo rules and clean up product pages "
        "so volume is less dependent on price cuts."
    )

# === TAB 4: MARKETING & CONTENT ============================================

with tabs[4]:
    st.header(f"Marketing & Content Strategy — {month_focus}")
    st.caption("🎬 This tab turns ratings and reviews into a content plan that non-technical teams can use.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    box_info(
        "Focus For This Month",
        "We will identify which categories convince people without heavy discounts, "
        "which topics block conversion, and what content to publish by platform."
    )

    plot_df = chart_curm.dropna(subset=["product_rating"])
    if guard(plot_df, "No rating values to visualize."):
        sizes = safe_bubble_sizes(plot_df.get("purchased_last_month"))
        if sizes is None:
            figm1 = px.scatter(plot_df, x="product_rating", y="total_reviews",
                               color="Category_std", opacity=0.9,
                               title="Engagement Map — Rating vs Reviews")
        else:
            figm1 = px.scatter(plot_df, x="product_rating", y="total_reviews",
                               color="Category_std", size=sizes, opacity=0.9,
                               title="Engagement Map — Rating vs Reviews (Bubble = Purchases)")
        figm1.update_layout(xaxis_title="Average Rating (★)", yaxis_title="Review Volume")
        st.plotly_chart(figm1, use_container_width=True)
        fig_notes(
            why="Find products people like and talk about.",
            what="Dots show rating on the x-axis and review volume on the y-axis.",
            sowhat="High rating with many reviews is a signal to scale with social proof.",
            nextstep="Add short creator videos and pin top reviews on the product page."
        )
        top = (plot_df.assign(score=lambda d: d["product_rating"].rank(pct=True)
                                           + safe_num(d["total_reviews"]).rank(pct=True))
                      .sort_values("score", ascending=False).head(1))
        if not top.empty:
            box_result(
                "Advocacy Engine",
                "Products with high ratings and many reviews grow on their own. "
                "Run creator shorts and position social proof above the fold to compound this advantage.",
                "good"
            )

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
            figc = px.bar(conv, x="Category_std", y="purch_per_review", text_auto=".2f",
                          title="Purchases Per Review (Conversion Proxy)")
            figc.update_layout(xaxis_title="Category", yaxis_title="Purchases / Review")
            st.plotly_chart(figc, use_container_width=True)
            fig_notes(
                why="Approximate how efficiently reviews convert to purchases.",
                what="Bars show purchases per review by category.",
                sowhat="High values are strong bets for paid and organic placement.",
                nextstep="Feature the top categories across paid and owned channels."
            )
            lead = conv.iloc[0]
            box_result(
                "High Intent Signal",
                f"{lead['Category_std']} converts reviews to purchases efficiently.",
                "good"
            )

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
            stop = set("the and to a of is it for in on my this that very too but not as be are you i so at have its they we just would".split())
            topw = words[~words.isin(stop)].value_counts().head(12).reset_index()
            topw.columns = ["word", "count"]
            if guard(topw, "No frequent words to display."):
                figw = px.bar(topw, x="word", y="count", text_auto=".2s", title="Common Customer Words (This Month)")
                st.plotly_chart(figw, use_container_width=True)
                fig_notes(
                    why="Turn common complaints and questions into content.",
                    what="Bars show the most frequent words in reviews this month.",
                    sowhat="These are plain-language topics customers care about.",
                    nextstep="Record short clips that answer these questions directly."
                )
                box_warn(
                    "Caution",
                    "Do not copy words blindly. Skim a few reviews to confirm real meaning before posting."
                )

    plat = (chart_curm.groupby("Category_std", as_index=False)
                 .agg(rating=("product_rating", "mean"),
                      reviews=("total_reviews", "sum"),
                      price=("discounted_price", "median")))
    if guard(plat, "Not enough fields to build a platform plan."):
        def platform_row(r):
            tiktok = 0.0; reels = 0.0; youtube = 0.0; paid = 0.0
            if pd.notna(r["rating"]) and pd.notna(r["reviews"]):
                if r["rating"] >= 4.2 and r["reviews"] >= np.nanpercentile(plat["reviews"], 60):
                    tiktok += 0.35; reels += 0.35
                if (r["rating"] < 4.2 and pd.notna(r["price"]) and r["price"] >= np.nanmedian(plat["price"])):
                    youtube += 0.4
                if r["reviews"] < np.nanpercentile(plat["reviews"], 40) and r["rating"] < 4.1:
                    paid += 0.4
            s = tiktok + reels + youtube + paid
            if s == 0: s = 1.0
            return pd.Series({
                "TikTok": round(100 * tiktok / s, 1),
                "IG Reels": round(100 * reels / s, 1),
                "YouTube": round(100 * youtube / s, 1),
                "Paid Search": round(100 * paid / s, 1),
            })
        mix_plat = plat.join(plat.apply(platform_row, axis=1))
        melt = mix_plat.melt(id_vars=["Category_std"], value_vars=["TikTok", "IG Reels", "YouTube", "Paid Search"],
                             var_name="Platform", value_name="Share %")
        figpm = px.bar(melt, x="Category_std", y="Share %", color="Platform", barmode="stack",
                       title="Recommended Platform Mix By Category")
        st.plotly_chart(figpm, use_container_width=True)
        fig_notes(
            why="Give a simple starting point for content allocation.",
            what="Stacked bars show suggested share by platform for each category.",
            sowhat="Avoids guessing and helps the team start fast.",
            nextstep="Use this split for the next sprint and adjust based on results."
        )

    st.markdown("### Seven-Day Content Plan")
    top_cats = (chart_curm.groupby("Category_std")["purchased_last_month"]
                .sum().sort_values(ascending=False).head(3).index.tolist())
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
    box_info("Publishing Guide", "Post on TikTok and IG Reels. Pin the best demo to the product page hero video.")
    st.markdown("\n".join([f"- {row}" for row in cal]))

    # plain-language wrap for this tab
    box_info(
        "What This Page Is Really Saying",
        "This page is about telling the story so shoppers believe and buy. The engagement map shows which categories already "
        "have strong ratings and lots of reviews, which are perfect for creator content and social proof. The conversion proxy "
        "shows where each review is pulling real weight. Frequent words from reviews turn into content topics. The platform mix "
        "and seven-day plan give a simple posting schedule so the marketing team does not have to guess where to start."
    )

# === TAB 5: PRODUCT & PRICING ==============================================

with tabs[5]:
    st.header(f"Product & Pricing — {month_focus}")
    st.caption("🧪 This tab shows quality signals and how prices step down from list.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    if "product_rating" in chart_curm.columns and chart_curm["product_rating"].notna().any():
        figp1 = px.box(chart_curm.dropna(subset=["product_rating"]),
                       x="Category_std", y="product_rating", color="Category_std",
                       title="Product Ratings by Category")
        figp1.update_layout(xaxis_title="", yaxis_title="Rating (★)")
        st.plotly_chart(figp1, use_container_width=True)
        fig_notes(
            why="Assess perceived quality by category.",
            what="Boxes show distribution and median rating.",
            sowhat="Low medians signal friction or wrong expectations.",
            nextstep="Add short demos and clarify sizing and care details."
        )
        worst = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
        if not worst.empty:
            wcat, wr = worst.index[0], worst.values[0]
            level = "bad" if wr < 4.0 else "ok"
            box_result("Quality Signal", f"{wcat} averages {wr:.2f} stars. Add demos and clear benefit bullets.", level)

    if {"original_price", "discounted_price"}.issubset(chart_curm.columns):
        ladder = (chart_curm[["Category_std", "original_price", "discounted_price"]]
                  .dropna().groupby("Category_std").median().reset_index())
        if guard(ladder, "No price ladder to show."):
            figp2 = px.scatter(ladder, x="original_price", y="discounted_price", color="Category_std",
                               title="Price Ladder (Median Original vs Discounted)")
            figp2.update_layout(xaxis_title="Original Price (median)", yaxis_title="Discounted Price (median)")
            st.plotly_chart(figp2, use_container_width=True)
            fig_notes(
                why="See how far discounting steps prices down from list.",
                what="Each dot is a category with median original and discounted price.",
                sowhat="Large gaps suggest reliance on promos.",
                nextstep="Test lighter promos paired with clearer product value bullets."
            )
            heavy = ladder[ladder["discounted_price"] <= 0.7 * ladder["original_price"]]
            tone = "ok" if heavy.empty else "bad"
            msg = "Median discounted prices sit near list. This is healthy." if heavy.empty else \
                  "Some categories lean on deep discounts. Test lighter promos with sharper value messages."
            box_result("Pricing Posture", msg, tone)

    under = (curm.assign(rating=curm["product_rating"],
                         purch=curm["purchased_last_month"])
                  .dropna(subset=["rating", "purch"]))
    if not under.empty:
        under = under.sort_values(["rating", "purch"], ascending=[True, False])
        under = under.drop_duplicates(subset=["product_title"], keep="first").head(10)
        view = under[["product_title", "Category_std", "rating", "purch", "discount_percentage"]]
        st.markdown("### Underperformers Needing Attention")
        st.dataframe(view.rename(columns={
            "product_title": "Product",
            "Category_std": "Category",
            "rating": "Rating (★)",
            "purch": "Purchases",
            "discount_percentage": "Discount %",
        }), use_container_width=True)
        box_result(
            "What To Fix First",
            "Start with low-rating, high-volume items. Clarify expectations and show the product in use.",
            "bad"
        )
        box_warn(
            "Watch Returns & CS Load",
            "Low-rating, high-volume items often create support tickets and returns. Fix these first to protect margin."
        )

    # plain-language wrap for this tab
    box_info(
        "What This Page Is Really Saying",
        "Here you see whether the products themselves are doing their job. The rating boxes show which categories customers "
        "like and which ones feel confusing or disappointing. The price ladder shows how far you are cutting from list to get "
        "a sale. The underperformer table points to specific products that are both visible and low rated, which are the first "
        "place to fix descriptions, images, and expectations before spending more on traffic."
    )

# === TAB 6: PLANS & ACCOUNTABILITY =========================================

with tabs[6]:
    st.header("Plans & Accountability")
    st.caption("🗂️ Concrete items with owners and due dates that match this month’s story.")

    actions = []

    g = (sales_f.groupby(["month", "Category_std"])["purchased_last_month"].sum()
         .reset_index().sort_values("month"))
    if not g.empty and prev_month in g["month"].values and month_focus in g["month"].values:
        pm = g[g["month"] == prev_month].set_index("Category_std")["purchased_last_month"]
        cm = g[g["month"] == month_focus].set_index("Category_std")["purchased_last_month"]
        both = pd.concat([pm, cm], axis=1); both.columns = ["prev", "cur"]
        both["delta"] = both["cur"] - both["prev"]; both = both.dropna()
        if not both.empty:
            top = both["delta"].abs().sort_values(ascending=False).head(1)
            cat = top.index[0]; d = both.loc[cat, "delta"]; direction = "up" if d > 0 else "down"
            actions.append(dict(team="Executive",
                                action=f"{cat} moved {direction} by {abs(d):,.0f} MoM. Set a numeric target and an owner.",
                                priority="High", due=(date.today() + timedelta(days=7)).isoformat(), status="Open"))

    if cur["reviews"] and pre["reviews"] and cur["reviews"] < pre["reviews"]:
        actions.append(dict(team="Marketing & Content",
                            action="Review volume slipped. Refresh creatives, seed UGC, and add social proof blocks on PDP.",
                            priority="High", due=(date.today() + timedelta(days=10)).isoformat(), status="Open"))

    if pd.notna(cur["disc"]) and cur["disc"] >= 30:
        actions.append(dict(team="Finance & Ops",
                            action=f"Average discount {cur['disc']:.1f}%. Test lighter discounts with bundles and sharper value bullets.",
                            priority="Medium", due=(date.today() + timedelta(days=14)).isoformat(), status="Open"))

    lowcat = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
    if not lowcat.empty and lowcat.values[0] < 4.0:
        actions.append(dict(team="Product & Pricing",
                            action=f"{lowcat.index[0]} averages {lowcat.values[0]:.2f}★. Add 20 to 30 second demos and clarify sizing and care.",
                            priority="Medium", due=(date.today() + timedelta(days=14)).isoformat(), status="Open"))

    actions.append(dict(team="Cross-Team",
                        action="Record a 10-minute Loom walkthrough linking to this hub and the key wins and issues.",
                        priority="Low", due=(date.today() + timedelta(days=5)).isoformat(), status="Open"))

    if actions:
        act_df = pd.DataFrame(actions, columns=["team", "action", "priority", "due", "status"])
        st.dataframe(act_df, use_container_width=True)
        st.download_button(
            "Download Action Plan (CSV)",
            act_df.to_csv(index=False).encode("utf-8"),
            "amazon_action_plan.csv",
            "text/csv"
        )
    else:
        st.write("No actions were generated for this slice.")

    # plain-language wrap for this tab
    box_info(
        "What This Page Is Really Saying",
        "This page turns the whole dashboard into a to-do list. Each row assigns a team, an action, a priority, and a due date "
        "based on what happened this month. Instead of just staring at charts, leaders can download this list, share it, and "
        "track whether the next month’s numbers moved in the right direction."
    )
