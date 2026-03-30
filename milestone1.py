import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def run():

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="border:1px solid rgba(255,255,255,0.25);border-radius:10px;
        padding:16px 20px;margin-bottom:20px;background:rgba(255,255,255,0.04)">
            <h2 style="color:white;margin:0">📊 Data Preprocessing & EDA</h2>
            <p style="color:#aaa;margin:6px 0 0 0;font-size:13px">
            Module: Gather behavioral datasets • Explore patterns in time spent,
            quiz scores and distraction logs • Perform correlation analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Sample Data ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "Study_Time": [1, 2, 3, 4, 5, 6, 7, 8],
        "Quiz_Score": [65, 70, 75, 82, 85, 88, 90, 92],
        "Distraction": [1, 2, 3, 1, 2, 3, 1, 2]
    })

    CARD = ("border:1px solid rgba(255,255,255,0.18);border-radius:10px;"
            "padding:16px;margin-bottom:16px;background:rgba(255,255,255,0.02)")
    TITLE = "color:white;font-weight:600;margin:0 0 10px 0"
    BG    = "rgba(0,0,0,0)"

    # ── ROW 1 ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Study Time vs Quiz Scores</p>', unsafe_allow_html=True)
        fig1 = px.scatter(df, x="Study_Time", y="Quiz_Score", trendline="ols",
                          color_discrete_sequence=["#60a5fa"])
        fig1.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color="white",
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Correlation Heatmap</p>', unsafe_allow_html=True)
        corr = df.corr()
        fig2, ax = plt.subplots(facecolor="none")
        ax.set_facecolor("none")
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, linewidths=0.5)
        plt.setp(ax.get_xticklabels(), color="white")
        plt.setp(ax.get_yticklabels(), color="white")
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 2 ─────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Distraction Frequency vs Performance</p>', unsafe_allow_html=True)
        avg = df.groupby("Distraction")["Quiz_Score"].mean().reset_index()
        avg.columns = ["Distraction_Level", "Avg_Quiz_Score"]
        fig3 = px.bar(avg, x="Distraction_Level", y="Avg_Quiz_Score",
                      color_discrete_sequence=["#60a5fa"],
                      labels={"Distraction_Level": "Distraction Level", "Avg_Quiz_Score": "Avg Quiz Score"})
        fig3.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color="white",
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Study Pattern Distribution</p>', unsafe_allow_html=True)
        study_pattern = pd.DataFrame({
            "Session": ["Morning", "Afternoon", "Evening", "Night"],
            "Value":   [25, 35, 25, 15]
        })
        fig4 = px.pie(study_pattern, names="Session", values="Value", hole=0.5,
                      color_discrete_sequence=["#60a5fa", "#a78bfa", "#34d399", "#f87171"])
        fig4.update_layout(paper_bgcolor=BG, font_color="white", margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)