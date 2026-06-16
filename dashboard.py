# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    layout="wide"
)

# -------------------
# Load Data
# -------------------

df = pd.read_csv("patient_data.csv")

# -------------------
# Sidebar
# -------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Patient Analysis",
        "Department Analysis",
        "About Me"
    ]
)

# -------------------
# Overview
# -------------------

if page == "Overview":

    st.title("🏥 Healthcare Analytics Dashboard")

    total_patients = len(df)
    total_cost = df["Cost"].sum()
    avg_stay = df["Length_of_Stay"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Patients", total_patients)
    col2.metric("Revenue ($)", f"{total_cost:,}")
    col3.metric("Avg Stay", round(avg_stay, 1))

# -------------------
# Patient Analysis
# -------------------

elif page == "Patient Analysis":

    st.title("Patient Analysis")

    fig_age = px.histogram(
        df,
        x="Age",
        nbins=10,
        title="Age Distribution"
    )

    st.plotly_chart(fig_age, use_container_width=True)

    fig_gender = px.pie(
        df,
        names="Gender",
        title="Gender Distribution"
    )

    st.plotly_chart(fig_gender, use_container_width=True)

# -------------------
# Department Analysis
# -------------------

elif page == "Department Analysis":

    st.title("Department Analysis")

    dept_patients = (
        df.groupby("Department")
        .size()
        .reset_index(name="Patients")
    )

    fig = px.bar(
        dept_patients,
        x="Department",
        y="Patients",
        title="Patients by Department"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------
# About Me
# -------------------

else:

    st.title("About Me")

    st.markdown("""
    ## Edison

    MSc Biomedical Engineering

    Nanyang Technological University

    Interests:

    - Healthcare AI
    - Medical Imaging
    - Data Analytics
    - Product Management
    """)

    st.markdown(
        "[GitHub](https://github.com/)"
    )

    st.markdown(
        "[LinkedIn](https://linkedin.com/)"
    )