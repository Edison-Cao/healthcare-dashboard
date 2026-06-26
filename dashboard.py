# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Healthcare Analytics DA Dashboard",
    layout="wide"
)

# =========================
# Generate Synthetic Data (with realistic messiness)
# =========================
@st.cache_data
def generate_raw_data(n=500, seed=42):
    np.random.seed(seed)

    departments = ["Cardiology", "Orthopedics", "Neurology", "Oncology", "Emergency"]
    genders = ["Male", "Female"]

    data = {
        "PatientID": range(1, n + 1),
        "Age": np.random.normal(55, 15, n).clip(18, 95),
        "Gender": np.random.choice(genders, n),
        "Department": np.random.choice(
            departments, n, p=[0.3, 0.25, 0.2, 0.15, 0.1]
        ),
        "Length_of_Stay": np.random.exponential(5, n).clip(1, 30),
        "Num_Procedures": np.random.randint(1, 8, n),
        "Num_Diagnoses": np.random.randint(1, 10, n),
    }

    df = pd.DataFrame(data)
    df["Age"] = df["Age"].round(1)
    df["Length_of_Stay"] = df["Length_of_Stay"].round(1)

    # Generate cost with realistic relationships
    base_cost = (
        2000
        + df["Age"] * 30
        + df["Length_of_Stay"] * 800
        + df["Num_Procedures"] * 500
        + df["Num_Diagnoses"] * 200
        + np.where(df["Department"] == "Oncology", 5000, 0)
        + np.where(df["Department"] == "Cardiology", 3000, 0)
        + np.where(df["Department"] == "Neurology", 2000, 0)
        + np.random.normal(0, 1500, n)
    )
    df["Cost"] = base_cost.clip(500, 80000).round(2)

    # --- Introduce realistic messiness ---

    # 1. Missing values
    df.loc[np.random.choice(n, 15, replace=False), "Age"] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), "Cost"] = np.nan
    df.loc[np.random.choice(n, 8, replace=False), "Length_of_Stay"] = np.nan

    # 2. Cost outliers
    df.loc[np.random.choice(n, 5, replace=False), "Cost"] = np.random.uniform(
        100000, 200000, 5
    )

    # 3. Duplicate rows
    dup_idx = np.random.choice(n, 8, replace=False)
    df = pd.concat([df, df.iloc[dup_idx].copy()], ignore_index=True)

    # 4. Inconsistent Gender encoding
    inconsistent_idx = np.random.choice(len(df), 5, replace=False)
    df.loc[inconsistent_idx, "Gender"] = np.random.choice(
        ["male", "FEMALE", "M", "f", " Female"], 5
    )

    return df


# =========================
# Data Cleaning
# =========================
@st.cache_data
def clean_data(df):
    cleaning_log = []
    df_clean = df.copy()

    # Step 1: Remove duplicates
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    cleaning_log.append(f"Removed {n_before - len(df_clean)} duplicate rows")

    # Step 2: Standardize Gender values
    df_clean["Gender"] = df_clean["Gender"].str.strip().str.lower()
    df_clean["Gender"] = df_clean["Gender"].replace(
        {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    )
    df_clean["Gender"] = df_clean["Gender"].str.capitalize()
    cleaning_log.append(
        "Standardized Gender values (e.g., 'm' → 'Male', 'FEMALE' → 'Female')"
    )

    # Step 3: Fill missing values
    n_age = df_clean["Age"].isna().sum()
    df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
    cleaning_log.append(
        f"Filled {n_age} missing Age values with median ({df_clean['Age'].median():.1f})"
    )

    n_cost = df_clean["Cost"].isna().sum()
    df_clean["Cost"] = df_clean["Cost"].fillna(
        df_clean.groupby("Department")["Cost"].transform("median")
    )
    cleaning_log.append(
        f"Filled {n_cost} missing Cost values with department-level median"
    )

    n_stay = df_clean["Length_of_Stay"].isna().sum()
    df_clean["Length_of_Stay"] = df_clean["Length_of_Stay"].fillna(
        df_clean["Length_of_Stay"].median()
    )
    cleaning_log.append(
        f"Filled {n_stay} missing Length_of_Stay values with median"
    )

    # Step 4: Remove outliers using 3×IQR on Cost
    Q1 = df_clean["Cost"].quantile(0.25)
    Q3 = df_clean["Cost"].quantile(0.75)
    upper_bound = Q3 + 3 * (Q3 - Q1)
    n_outliers = (df_clean["Cost"] > upper_bound).sum()
    df_clean = df_clean[df_clean["Cost"] <= upper_bound]
    cleaning_log.append(
        f"Removed {n_outliers} cost outliers above ${upper_bound:,.0f} (3×IQR method)"
    )

    # Step 5: Cast types
    df_clean["Age"] = df_clean["Age"].round(0).astype(int)

    return df_clean, cleaning_log, upper_bound


# =========================
# Load Data
# =========================
raw_df = generate_raw_data()
df, cleaning_log, outlier_upper = clean_data(raw_df)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("🎛 Filters")

selected_department = st.sidebar.selectbox(
    "Department", ["All"] + sorted(df["Department"].unique())
)
selected_gender = st.sidebar.selectbox(
    "Gender", ["All"] + sorted(df["Gender"].unique())
)
age_range = st.sidebar.slider(
    "Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 70)
)

# =========================
# Apply Filters
# =========================
filtered_df = df.copy()
if selected_department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == selected_department]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
filtered_df = filtered_df[
    (filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])
]

# =========================
# Navigation
# =========================
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Data Cleaning",
        "Patient Insights",
        "Statistical Analysis",
        "Patient Segmentation",
        "Clinical Operations",
        "About",
    ],
)

# Guard: empty filter result
if filtered_df.empty and page != "Data Cleaning":
    st.warning("⚠️ No data matches current filters. Please adjust the sidebar filters.")
    st.stop()

# =========================
# EXECUTIVE OVERVIEW
# =========================
if page == "Executive Overview":

    st.title("🏥 Healthcare Analytics DA Dashboard")

    total_patients = len(filtered_df)
    total_cost = filtered_df["Cost"].sum()
    avg_stay = filtered_df["Length_of_Stay"].mean()
    avg_cost = filtered_df["Cost"].mean()

    X = filtered_df[["Age", "Length_of_Stay", "Num_Procedures", "Num_Diagnoses"]]
    y = filtered_df["Cost"]
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Patients", total_patients)
    col2.metric("Total Cost", f"${total_cost:,.0f}")
    col3.metric("Avg Stay", f"{avg_stay:.1f} days")
    col4.metric("Avg Cost", f"${avg_cost:,.0f}")
    col5.metric(
        "Cost Model R²",
        f"{r2:.2f}",
        help="Variance in cost explained by Age, Stay, Procedures, Diagnoses",
    )

    st.divider()

    corr_stay, p_stay = stats.pearsonr(
        filtered_df["Length_of_Stay"], filtered_df["Cost"]
    )
    corr_age, p_age = stats.pearsonr(filtered_df["Age"], filtered_df["Cost"])

    st.subheader("📌 Key Statistical Findings")

    col1, col2 = st.columns(2)
    with col1:
        p_stay_str = "< 0.001" if p_stay < 0.001 else f"= {p_stay:.3f}"
        st.info(
            f"**Length of Stay ↔ Cost:** r = {corr_stay:.2f} (p {p_stay_str})\n\n"
            "Strong positive correlation — each additional day significantly drives up cost."
        )
    with col2:
        p_age_str = "< 0.001" if p_age < 0.001 else f"= {p_age:.3f}"
        st.info(
            f"**Age ↔ Cost:** r = {corr_age:.2f} (p {p_age_str})\n\n"
            "Older patients tend to incur higher costs, supporting age-adjusted resource planning."
        )

    col1, col2 = st.columns(2)
    with col1:
        dept_cost = (
            filtered_df.groupby("Department")["Cost"].mean().reset_index()
        )
        fig = px.bar(
            dept_cost,
            x="Department",
            y="Cost",
            title="Average Cost by Department",
            color="Cost",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            filtered_df,
            x="Length_of_Stay",
            y="Cost",
            color="Department",
            trendline="ols",
            title="Length of Stay vs Cost (with OLS Trend Lines)",
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# DATA CLEANING
# =========================
elif page == "Data Cleaning":

    st.title("🧹 Data Cleaning Process")
    st.markdown(
        "This page documents every data quality issue found and the action taken to resolve it."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data")
        st.dataframe(raw_df.head(20), use_container_width=True)
        st.caption(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    with col2:
        st.subheader("Cleaned Data")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.divider()
    st.subheader("📋 Issues Found & Actions Taken")
    for i, log in enumerate(cleaning_log, 1):
        st.success(f"**Step {i}:** {log}")

    st.divider()
    st.subheader("📊 Missing Value Rate by Column")

    missing = raw_df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(raw_df) * 100).round(2)
    missing = missing[missing["Missing Count"] > 0]

    if not missing.empty:
        fig = px.bar(
            missing,
            x="Column",
            y="Missing %",
            title="Missing Value Rate (%)",
            color="Missing %",
            color_continuous_scale="Oranges",
            text="Missing Count",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Outlier Detection — Cost (Raw Data)")

    Q1 = raw_df["Cost"].quantile(0.25)
    Q3 = raw_df["Cost"].quantile(0.75)
    upper = Q3 + 3 * (Q3 - Q1)

    fig = px.box(
        raw_df,
        y="Cost",
        title="Cost Distribution (Raw) — Outliers Visible",
        points="outliers",
        color_discrete_sequence=["#EF553B"],
    )
    fig.add_hline(
        y=upper,
        line_dash="dash",
        line_color="red",
        annotation_text=f"3×IQR Upper Bound: ${upper:,.0f}",
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PATIENT INSIGHTS
# =========================
elif page == "Patient Insights":

    st.title("📊 Patient Insights")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            filtered_df,
            x="Age",
            nbins=15,
            title="Patient Age Distribution",
            color_discrete_sequence=["#636EFA"],
        )
        fig.add_vline(
            x=filtered_df["Age"].mean(),
            line_dash="dash",
            annotation_text=f"Mean: {filtered_df['Age'].mean():.1f}",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(filtered_df, names="Gender", title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📈 Pearson Correlation Matrix")

    num_cols = ["Age", "Length_of_Stay", "Num_Procedures", "Num_Diagnoses", "Cost"]
    corr_matrix = filtered_df[num_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix (all numeric variables)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Correlation with Cost — Statistical Significance")

    corr_cost = corr_matrix["Cost"].drop("Cost").sort_values(ascending=False)
    col1, col2 = st.columns(2)

    for i, (var, corr_val) in enumerate(corr_cost.items()):
        _, p_val = stats.pearsonr(
            filtered_df[var].dropna(),
            filtered_df.loc[filtered_df[var].notna(), "Cost"],
        )
        sig = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
        p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
        target = col1 if i % 2 == 0 else col2
        target.metric(
            label=f"{var} ↔ Cost",
            value=f"r = {corr_val:.3f}",
            delta=f"p {p_str} — {sig}",
        )

# =========================
# STATISTICAL ANALYSIS
# =========================
elif page == "Statistical Analysis":

    st.title("📐 Statistical Analysis")

    tab1, tab2 = st.tabs(["🔢 Regression Analysis", "📊 ANOVA Test"])

    # --- Regression ---
    with tab1:
        st.subheader("Multiple Linear Regression: Predicting Patient Cost")
        st.markdown(
            "**Research Question:** Which patient factors significantly predict hospitalization cost?"
        )

        features = ["Age", "Length_of_Stay", "Num_Procedures", "Num_Diagnoses"]
        X = filtered_df[features]
        y = filtered_df["Cost"]
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.3f}", help="Proportion of variance explained by the model")
        col2.metric("Intercept", f"${model.intercept_:,.0f}")
        col3.metric("Sample Size", len(X))

        coef_df = pd.DataFrame(
            {
                "Feature": features,
                "Coefficient ($)": model.coef_.round(2),
                "Interpretation": [
                    f"Each +1 year of age adds ${model.coef_[0]:,.0f} to cost",
                    f"Each +1 day of stay adds ${model.coef_[1]:,.0f} to cost",
                    f"Each +1 procedure adds ${model.coef_[2]:,.0f} to cost",
                    f"Each +1 diagnosis adds ${model.coef_[3]:,.0f} to cost",
                ],
            }
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                coef_df,
                x="Coefficient ($)",
                y="Feature",
                orientation="h",
                title="Regression Coefficients",
                color="Coefficient ($)",
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            residuals = y - y_pred
            fig = px.scatter(
                x=y_pred,
                y=residuals,
                title="Residual Plot (should be random around 0)",
                labels={"x": "Predicted Cost ($)", "y": "Residuals"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Coefficient Interpretation")
        st.dataframe(
            coef_df[["Feature", "Coefficient ($)", "Interpretation"]],
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            f"**Model Summary:** The model explains **{r2:.1%}** of variance in patient cost. "
            f"Length of Stay (${model.coef_[1]:,.0f}/day) and Num_Procedures "
            f"(${model.coef_[2]:,.0f}/procedure) are the strongest cost drivers."
        )

    # --- ANOVA ---
    with tab2:
        st.subheader("One-Way ANOVA: Are Department Costs Significantly Different?")
        st.markdown(
            "**H₀:** All departments have the same mean cost  \n"
            "**H₁:** At least one department differs significantly"
        )

        dept_groups = [
            g["Cost"].values for _, g in filtered_df.groupby("Department")
        ]
        f_stat, p_value = stats.f_oneway(*dept_groups)

        col1, col2, col3 = st.columns(3)
        col1.metric("F-Statistic", f"{f_stat:.3f}")
        col2.metric("p-value", "< 0.001" if p_value < 0.001 else f"{p_value:.4f}")
        col3.metric("Result", "Reject H₀ ✅" if p_value < 0.05 else "Fail to Reject H₀")

        if p_value < 0.05:
            st.success(
                f"F({len(dept_groups)-1}, {len(filtered_df)-len(dept_groups)}) = {f_stat:.2f}, "
                f"p < 0.001 — There is a **statistically significant difference** "
                "in mean costs across departments."
            )

        fig = px.box(
            filtered_df,
            x="Department",
            y="Cost",
            color="Department",
            title="Cost Distribution by Department",
            points="outliers",
        )
        st.plotly_chart(fig, use_container_width=True)

        dept_summary = (
            filtered_df.groupby("Department")["Cost"]
            .agg(["mean", "median", "std", "count"])
            .round(2)
            .reset_index()
        )
        dept_summary.columns = ["Department", "Mean Cost", "Median Cost", "Std Dev", "N"]
        st.subheader("📋 Department Cost Summary Statistics")
        st.dataframe(dept_summary, use_container_width=True, hide_index=True)

# =========================
# PATIENT SEGMENTATION
# =========================
elif page == "Patient Segmentation":

    st.title("🔵 Patient Segmentation — K-Means Clustering")
    st.markdown(
        "**Research Question:** Can we identify distinct patient groups for targeted care management?"
    )

    features = ["Age", "Length_of_Stay", "Num_Procedures", "Num_Diagnoses", "Cost"]
    seg_df = filtered_df[features].dropna().copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(seg_df)

    # Elbow method
    st.subheader("📐 Choosing K — Elbow Method")
    inertias = []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)

    fig = px.line(
        x=list(k_range),
        y=inertias,
        markers=True,
        labels={"x": "Number of Clusters (K)", "y": "Inertia"},
        title="Elbow Method — K=3 Selected",
    )
    fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="K = 3")
    st.plotly_chart(fig, use_container_width=True)

    # Fit final model
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    # Sort clusters by mean Cost so labels are intuitive
    cluster_costs = {
        c: seg_df.iloc[labels == c]["Cost"].mean() for c in range(3)
    }
    sorted_clusters = sorted(cluster_costs, key=cluster_costs.get)
    label_map = {
        sorted_clusters[0]: "Low Risk",
        sorted_clusters[1]: "Medium Risk",
        sorted_clusters[2]: "High Risk",
    }
    seg_df["Cluster"] = [label_map[l] for l in labels]

    color_map = {
        "Low Risk": "#00CC96",
        "Medium Risk": "#FFA15A",
        "High Risk": "#EF553B",
    }

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            seg_df,
            x="Length_of_Stay",
            y="Cost",
            color="Cluster",
            size="Num_Procedures",
            title="Segments: Stay vs Cost",
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            seg_df,
            x="Age",
            y="Cost",
            color="Cluster",
            title="Segments: Age vs Cost",
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Cluster Profiles")
    cluster_summary = seg_df.groupby("Cluster")[features].mean().round(2)
    cluster_summary.insert(0, "Patient Count", seg_df["Cluster"].value_counts())
    st.dataframe(cluster_summary, use_container_width=True)

    high_risk = seg_df[seg_df["Cluster"] == "High Risk"]
    pct = len(high_risk) / len(seg_df)

    st.warning(
        f"**High Risk patients ({pct:.0%} of cohort)** average ${high_risk['Cost'].mean():,.0f} in cost, "
        f"{high_risk['Length_of_Stay'].mean():.1f} days of stay, and "
        f"{high_risk['Num_Procedures'].mean():.1f} procedures — "
        "priority candidates for care coordination intervention."
    )

# =========================
# CLINICAL OPERATIONS
# =========================
elif page == "Clinical Operations":

    st.title("🏥 Clinical Operations Analysis")

    dept_stats = (
        filtered_df.groupby("Department")
        .agg({"Cost": "mean", "Length_of_Stay": "mean"})
        .reset_index()
    )
    dept_stats["Cost_per_Day"] = (
        dept_stats["Cost"] / dept_stats["Length_of_Stay"]
    ).round(2)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            dept_stats,
            x="Department",
            y="Cost",
            title="Average Total Cost by Department",
            color="Cost",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            filtered_df,
            x="Length_of_Stay",
            y="Cost",
            color="Department",
            trendline="ols",
            title="Length of Stay vs Cost (OLS Trend)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("⚡ Efficiency Metric: Cost Per Day by Department")

    fig = px.bar(
        dept_stats.sort_values("Cost_per_Day", ascending=False),
        x="Department",
        y="Cost_per_Day",
        title="Cost Per Day — Operational Efficiency Indicator",
        color="Cost_per_Day",
        color_continuous_scale="Reds",
        text="Cost_per_Day",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    ineff = dept_stats.sort_values("Cost_per_Day", ascending=False).iloc[0]
    st.warning(
        f"**{ineff['Department']}** has the highest cost-per-day at ${ineff['Cost_per_Day']:,.0f}/day. "
        "Cost-per-day is a more accurate efficiency metric than total cost alone — "
        "recommend operational review of procedure utilization in this department."
    )

# =========================
# ABOUT
# =========================
else:

    st.title("👨‍⚕️ About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Edison
            MSc Biomedical Engineering (NTU)

            ### Project
            Healthcare Analytics DA Dashboard

            ### Skills Demonstrated
            - **Data Cleaning** — Missing values, outliers, duplicates, encoding normalization
            - **Exploratory Analysis** — Correlation matrix, distribution analysis
            - **Statistical Testing** — Pearson correlation, One-Way ANOVA (F-test)
            - **Predictive Modeling** — Multiple Linear Regression (R², residuals)
            - **Unsupervised ML** — K-Means Clustering with Elbow Method
            - **Data Visualization** — Plotly, Streamlit

            ### Analysis Question
            *Which patient and clinical factors significantly drive hospitalization cost,
            and can we identify distinct patient risk segments for targeted care management?*
            """
        )

    with col2:
        st.subheader("📊 Dataset Overview")
        dataset_info = pd.DataFrame(
            {
                "Field": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Non-Null Count": df.notnull().sum().values,
                "Sample Value": [str(df[c].iloc[0]) for c in df.columns],
            }
        )
        st.dataframe(dataset_info, use_container_width=True, hide_index=True)
        st.caption(
            f"Synthetic dataset — {len(df)} patients × {len(df.columns)} variables  \n"
            "Generated with realistic distributions and intentional data quality issues for cleaning demonstration."
        )