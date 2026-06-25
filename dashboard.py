# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Healthcare Analytics PM Dashboard",
    layout="wide"
)

# =========================
# Load Data
# =========================
df = pd.read_csv("patient_data.csv")

# =========================
# Sidebar Filters (PM Level)
# =========================
st.sidebar.header("🎛 Filters")

selected_department = st.sidebar.selectbox(
    "Department",
    ["All"] + sorted(df["Department"].unique())
)

selected_gender = st.sidebar.selectbox(
    "Gender",
    ["All"] + sorted(df["Gender"].unique())
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (20, 70)
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
    (filtered_df["Age"] >= age_range[0]) &
    (filtered_df["Age"] <= age_range[1])
]

# =========================
# Sidebar Navigation
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview", "Patient Insights", "Clinical Operations", "About"]
)

# =========================
# EXECUTIVE OVERVIEW (PM核心)
# =========================
if page == "Executive Overview":

    st.title("🏥 Healthcare Analytics PM Dashboard")

    # KPIs
    total_patients = len(filtered_df)
    total_cost = filtered_df["Cost"].sum()
    avg_stay = filtered_df["Length_of_Stay"].mean()
    avg_cost = filtered_df["Cost"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Patients", total_patients)
    col2.metric("Total Cost", f"${total_cost:,.0f}")
    col3.metric("Avg Stay", f"{avg_stay:.1f} days")
    col4.metric("Avg Cost", f"${avg_cost:.0f}")

    st.divider()

    # Executive Insight (PM必须有)
    st.subheader("📌 Key Insight")

    top_dept = (
        filtered_df.groupby("Department")["Cost"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    st.info(
        f"Highest cost department: **{top_dept}** "
        f"(Potential optimization target for hospital cost reduction strategy)"
    )

# =========================
# PATIENT INSIGHTS
# =========================
elif page == "Patient Insights":

    st.title("📊 Patient Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig_age = px.histogram(
            filtered_df,
            x="Age",
            nbins=10,
            title="Patient Age Distribution"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        fig_gender = px.pie(
            filtered_df,
            names="Gender",
            title="Gender Distribution"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    # PM增强分析
    st.subheader("💡 Insight")

    high_cost_ratio = (filtered_df["Cost"] > filtered_df["Cost"].mean()).mean()

    st.success(
        f"{high_cost_ratio:.0%} of patients are above average cost → "
        "Potential high-value patient segment for hospital resource planning"
    )

# =========================
# CLINICAL OPERATIONS (核心PM页面)
# =========================
elif page == "Clinical Operations":

    st.title("🏥 Clinical Operations Analysis")

    col1, col2 = st.columns(2)

    with col1:

        dept_stats = (
            filtered_df.groupby("Department")
            .agg({
                "Cost": "mean",
                "Length_of_Stay": "mean"
            })
            .reset_index()
        )

        fig1 = px.bar(
            dept_stats,
            x="Department",
            y="Cost",
            title="Average Cost by Department"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        fig2 = px.scatter(
            filtered_df,
            x="Length_of_Stay",
            y="Cost",
            color="Department",
            title="Length of Stay vs Cost (Operational Efficiency)"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # PM分析语句
    st.subheader("📌 Operational Insight")

    ineff_dept = dept_stats.sort_values("Cost", ascending=False).iloc[0]["Department"]

    st.warning(
        f"{ineff_dept} shows highest cost efficiency risk → "
        "Recommend resource allocation review"
    )

# =========================
# ABOUT
# =========================
else:

    st.title("👨‍⚕️ About This Project")

    st.markdown("""
    ### Edison

    MSc Biomedical Engineering (NTU)

    ### Project
    Healthcare Analytics PM Dashboard

    ### Skills Demonstrated
    - Product Thinking (KPIs + Insights)
    - Data Analysis (Pandas)
    - Data Visualization (Plotly)
    - Dashboard Design (Streamlit)
    - Healthcare Domain Understanding

    ### Goal
    Build data-driven healthcare decision support tools
    """)