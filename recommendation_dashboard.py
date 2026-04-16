import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 


st.set_page_config(layout="wide")

st.title("📘 Milestone 3: Recommendation Engine ")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
df = pd.read_csv("study_behavior_dataset.csv")

numeric_cols = [
    'Study_Time_Hours',
    'Attention_Span_Minutes',
    'Past_Score_Percentage',
    'Quiz_Score_Percentage'
]
# ----------------------------------------
# Prediction Model (Score Prediction)
# ----------------------------------------

from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()

X_reg = df[['Study_Time_Hours']]
y_reg = df['Quiz_Score_Percentage']

reg_model.fit(X_reg, y_reg)

# Predict score for 3 hours study time
predicted_score = reg_model.predict([[3]])

st.write("📊 Predicted Score if student studies 3 hours:", round(predicted_score[0],2))
# ----------------------------------------
# Clustering
# ----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ----------------------------------------
# Select Student
# ----------------------------------------
student_id = st.selectbox("Select Student ID", df["Student_ID"])

student = df[df["Student_ID"] == student_id].iloc[0]
cluster = student["Cluster"]

st.subheader("📌 Personalized Study Recommendations")

# ----------------------------------------
# Recommendation Logic Based On Cluster
# ----------------------------------------

if cluster == 0:
    time_slots = ["8:00-10:00 AM", "2:00-3:00 PM"]
    duration = "90 minutes"
    breaks = "10 min break every 30 minutes"
    tools = ["Pomodoro Timer", "Digital Notes"]
elif cluster == 1:
    time_slots = ["10:00-12:00 PM", "6:00-7:30 PM"]
    duration = "60 minutes"
    breaks = "5 min break every 25 minutes"
    tools = ["Site Blocker", "Focus Music"]
else:
    time_slots = ["7:00-8:30 PM"]
    duration = "45 minutes"
    breaks = "5 min break every 20 minutes"
    tools = ["Pomodoro Timer", "Site Blocker"]

# Effectiveness (based on cluster improvement)
cluster_mean_improvement = (
    df[df["Cluster"] == cluster]["Quiz_Score_Percentage"].mean()
    - df[df["Cluster"] == cluster]["Past_Score_Percentage"].mean()
)

effectiveness = round(cluster_mean_improvement, 1)

# ----------------------------------------
# Layout
# ----------------------------------------

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### 🗓 Optimal Study Time Slots")
    for t in time_slots:
        st.write("•", t)

    st.markdown("### ⏳ Study Duration Pattern")
    st.write(duration)

    st.markdown("### ☕ Break Schedule")
    st.write(breaks)

    st.markdown("### 📈 Effectiveness")
    st.progress(min(max(effectiveness, 0)/100, 1))
    st.write(f"Expected Improvement: {effectiveness}%")

with col2:
    st.markdown("### 🛠 Recommended Study Tools")

    for tool in tools:
        st.success(tool)

# ----------------------------------------
# Weekly Study Schedule
# ----------------------------------------

st.subheader("📅 Weekly Study Schedule")

weekly_data = pd.DataFrame({
    "Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    "Hours": [
        student["Study_Time_Hours"] * 0.8,
        student["Study_Time_Hours"],
        student["Study_Time_Hours"] * 1.1,
        student["Study_Time_Hours"] * 0.7,
        student["Study_Time_Hours"],
        student["Study_Time_Hours"] * 0.5,
        student["Study_Time_Hours"] * 0.6
    ]
})

fig = px.bar(weekly_data, x="Day", y="Hours",
             title="Weekly Study Schedule")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# Expected Performance Improvement
# ----------------------------------------

st.subheader("📊 Expected Performance Improvement")

performance_data = pd.DataFrame({
    "Week": ["Week 1","Week 2","Week 3","Week 4"],
    "Score": [
        student["Quiz_Score_Percentage"],
        student["Quiz_Score_Percentage"] + 3,
        student["Quiz_Score_Percentage"] + 6,
        student["Quiz_Score_Percentage"] + 10
    ]
})

fig2 = px.line(performance_data, x="Week", y="Score",
               title="Projected Quiz Score Growth")
st.plotly_chart(fig2, use_container_width=True)