import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pyprojroot import here
import os
from skimpy import clean_columns

# Load your data here
# data_member_index = pd.read_csv('your_data.csv')

data_member_index_means = (
    data_member_index.drop(columns=["Member ID"])
    .groupby("High risk member")
    .mean()
    .round(2)
).reset_index()

def what_if_analysis(n_members_to_move, high_risk_pmpm, average_pmpm):
    pmpm_difference_per_member = high_risk_pmpm - average_pmpm
    total_savings = pmpm_difference_per_member * n_members_to_move
    return total_savings

# Streamlit app layout
st.title("What-If Analysis for Member Risk Adjustment")

# Get average PMPM values for high-risk and average members
high_risk_average_pmpm = data_member_index_means[data_member_index_means["High risk member"] == True]["PMPM"].iloc[0]
average_member_pmpm = data_member_index_means[data_member_index_means["High risk member"] == False]["PMPM"].iloc[0]

# User input for the number of members to move
n_members_to_move = st.number_input("Number of High-Risk Members to Move to Average Risk", min_value=1, value=10)

# Calculate savings
savings = what_if_analysis(n_members_to_move, high_risk_average_pmpm, average_member_pmpm)

if st.button("Calculate Savings"):
    st.write(f"Total savings in PMPM: {savings}")

# Optionally, you can add more interactive elements or visualizations here
