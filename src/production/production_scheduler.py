import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pyprojroot import here
import os
from skimpy import clean_columns

# Set the directory to the data folder
path_data = here("./data")
os.chdir(path_data)

# Load your data here
data_member_index = pd.read_csv('data_member_index.csv')

# Process the data to get means, round to 0 decimal places
data_member_index_means = (
    data_member_index.drop(columns=["Member ID"])
    .groupby("High risk member")
    .mean()
).reset_index()

# Function to calculate what-if analysis
def what_if_analysis(n_members_to_move, high_risk_pmpm, average_pmpm):
    pmpm_difference_per_member = round(high_risk_pmpm - average_pmpm, 0)
    total_savings = round(pmpm_difference_per_member * n_members_to_move, 0)
    return total_savings

# Streamlit app layout
st.title("What-If Analysis for Member Risk Adjustment")

# Get average PMPM values for high-risk and average members, rounded to 0 decimal places
high_risk_average_pmpm = data_member_index_means[data_member_index_means["High risk member"] == True]["PMPM"].iloc[0].round(0)
average_member_pmpm = data_member_index_means[data_member_index_means["High risk member"] == False]["PMPM"].iloc[0].round(0)

# Display the rounded average PMPM values with dollar sign
st.write(f"Average PMPM for High-Risk Members: ${high_risk_average_pmpm}")
st.write(f"Average PMPM for Average Members: ${average_member_pmpm}")

# User input for the number of members to move
n_members_to_move = st.number_input("Number of High-Risk Members to Move to Average Risk", min_value=1, value=10)

# Calculate and display savings
if st.button("Calculate Savings"):
    savings = what_if_analysis(n_members_to_move, high_risk_average_pmpm, average_member_pmpm)
    st.write(f"Total savings in PMPM: ${savings}")

st.write("Data Member Index Means:")
st.dataframe(data_member_index_means.round(2))

sorted_data = data_member_index.sort_values(by="Member index").round(2)


# Display the data_member_index DataFrame
st.dataframe(sorted_data, hide_index = True)
