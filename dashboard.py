import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Student Behavior Analysis Dashboard")

# Load dataset
df = pd.read_csv("study_behavior_dataset.csv")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Study Time vs Quiz Score
st.subheader("Study Time vs Quiz Score")
plt.figure()
sns.scatterplot(x="Study_Time_Hours", y="Quiz_Score_Percentage", data=df)
st.pyplot(plt)

# Distraction vs Performance
st.subheader("Distraction vs Performance")
plt.figure()
sns.barplot(x="Distraction_Level", y="Quiz_Score_Percentage", data=df)
st.pyplot(plt)