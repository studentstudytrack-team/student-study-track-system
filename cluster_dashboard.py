import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Reusable card wrapper ─────────────────────────────────────────────────
CARD  = ("border:1px solid rgba(255,255,255,0.18);border-radius:10px;"
         "padding:16px;margin-bottom:16px;background:rgba(255,255,255,0.02)")
TITLE = "color:white;font-weight:600;font-size:14px;margin:0 0 10px 0"
BG    = "rgba(0,0,0,0)"
CHART_LAYOUT = dict(plot_bgcolor=BG, paper_bgcolor=BG, font_color="white",
                    margin=dict(t=30, b=10, l=10, r=10))


def run():

    st.markdown("""
        <div style="border:1px solid rgba(255,255,255,0.25);border-radius:10px;
        padding:16px 20px;margin-bottom:20px;background:rgba(255,255,255,0.04)">
            <h2 style="color:white;margin:0">📊 Student Behavior Clustering Dashboard</h2>
            <p style="color:#aaa;margin:6px 0 0 0;font-size:13px">
            K-Means behavioral clustering · Cluster profiles · Add new students
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv("study_behavior_dataset.csv")
    df.columns = df.columns.str.strip()

    try:
        logs = pd.read_csv("study_logs.csv")
        logs.columns = logs.columns.str.strip()
    except Exception:
        logs = pd.DataFrame()

    numeric_cols   = ['Study_Hours_per_Week', 'Stress_Level (1-10)', 'Midterm_Score', 'Quizzes_Avg']
    display_labels = ['Study Hours/Week', 'Stress Level', 'Midterm Score', 'Quiz Avg']

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # ── Sidebar ───────────────────────────────────────────────────────────
    k = st.sidebar.slider("Select Number of Clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    if "Department" in df.columns:
        dept_filter = st.sidebar.selectbox("Filter by Department",
                                           ["All"] + list(df["Department"].unique()))
        if dept_filter != "All":
            df = df[df["Department"] == dept_filter]

    # ── Scatter + Cluster list ────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Student Behavior Clusters</p>', unsafe_allow_html=True)
        fig = px.scatter(df, x="Study_Hours_per_Week", y="Quizzes_Avg",
                         color=df["Cluster"].astype(str), size="Midterm_Score",
                         hover_data=["Student_ID", "Grade"],
                         color_discrete_sequence=["#60a5fa","#a78bfa","#34d399","#f87171","#fbbf24"])
        fig.update_layout(**CHART_LAYOUT, legend_title_text="Cluster")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Behavior Clusters</p>', unsafe_allow_html=True)
        cluster_counts = df["Cluster"].value_counts().sort_index()
        total = len(df)
        for i, count in cluster_counts.items():
            pct = round((count / total) * 100, 1)
            st.markdown(
                f"""<div style="border:1px solid rgba(255,255,255,0.15);border-radius:8px;
                padding:8px 12px;margin-bottom:8px;background:rgba(255,255,255,0.03)">
                <span style="color:white;font-weight:600">Cluster {i}</span>
                <span style="float:right;color:#aaa;font-size:12px">{count}</span><br>
                <span style="color:#aaa;font-size:12px">{pct}% of students</span>
                </div>""", unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Radar Chart ───────────────────────────────────────────────────────
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Cluster Characteristics</p>', unsafe_allow_html=True)
    cluster_means = df.groupby("Cluster")[numeric_cols].mean()
    fig_radar = go.Figure()
    colors_r = ["#60a5fa","#a78bfa","#34d399","#f87171","#fbbf24"]
    for cluster in cluster_means.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_means.loc[cluster].values,
            theta=display_labels,
            fill='toself',
            name=f"Cluster {cluster}",
            line=dict(color=colors_r[cluster % len(colors_r)])
        ))
    fig_radar.update_layout(**CHART_LAYOUT,
                            polar=dict(radialaxis=dict(visible=True, color="#aaa"),
                                       bgcolor="rgba(0,0,0,0)"),
                            showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Cluster Profile ───────────────────────────────────────────────────
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">Cluster Profile</p>', unsafe_allow_html=True)
    selected_cluster = st.selectbox("Select Cluster to View Profile", sorted(df["Cluster"].unique()))
    cluster_data = df[df["Cluster"] == selected_cluster]

    # Average Values — metric cards
    st.markdown("<p style='color:#aaa;font-size:13px;margin:8px 0 6px 0'>Average Values</p>", unsafe_allow_html=True)
    avgs  = cluster_data[numeric_cols].mean()
    icons = ["📚", "😓", "📝", "🎯"]
    m_cols = st.columns(4)
    for col_w, col_name, disp, icon in zip(m_cols, numeric_cols, display_labels, icons):
        col_w.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.15);border-radius:8px;
            padding:12px;text-align:center;background:rgba(255,255,255,0.03)">
            <div style="font-size:20px">{icon}</div>
            <div style="color:#aaa;font-size:11px;margin:4px 0">{disp}</div>
            <div style="color:white;font-size:20px;font-weight:700">{avgs[col_name]:.2f}</div>
            </div>""", unsafe_allow_html=True
        )

    # Grade + Dept charts
    st.markdown("<p style='color:#aaa;font-size:13px;margin:16px 0 6px 0'>Distribution</p>", unsafe_allow_html=True)
    g_col, d_col = st.columns(2)

    with g_col:
        grade_counts = cluster_data["Grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        grade_counts["Pct"] = (grade_counts["Count"] / grade_counts["Count"].sum() * 100).round(1)
        fig_g = px.bar(grade_counts, x="Grade", y="Count", text="Pct",
                       color_discrete_sequence=["#60a5fa"],
                       title=f"Grade Distribution — Cluster {selected_cluster}")
        fig_g.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_g.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_g, use_container_width=True)

    with d_col:
        dept_counts = cluster_data["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig_d = px.pie(dept_counts, names="Department", values="Count", hole=0.4,
                       color_discrete_sequence=["#60a5fa","#a78bfa","#34d399","#f87171","#fbbf24"],
                       title=f"Department Split — Cluster {selected_cluster}")
        fig_d.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Add New Student ───────────────────────────────────────────────────
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">➕ Add New Student</p>', unsafe_allow_html=True)
    with st.form("new_student_form"):
        c1, c2 = st.columns(2)
        with c1:
            study_time = st.number_input("Study Hours per Week", 0.0, 80.0, 20.0)
            stress     = st.number_input("Stress Level (1-10)", 1, 10, 5)
        with c2:
            midterm  = st.number_input("Midterm Score", 0, 100, 70)
            quiz_avg = st.number_input("Quizzes Avg", 0, 100, 75)
        submitted = st.form_submit_button("🔍 Assign Cluster", use_container_width=True)

    if submitted:
        new_data    = pd.DataFrame([[study_time, stress, midterm, quiz_avg]], columns=numeric_cols)
        new_scaled  = scaler.transform(new_data)
        new_cluster = kmeans.predict(new_scaled)[0]
        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.25);border-radius:8px;
            padding:12px 18px;margin-top:8px;background:rgba(255,255,255,0.05)">
            <span style="color:white;font-weight:700">✅ New Student → Cluster {new_cluster}</span><br>
            <span style="color:#aaa;font-size:13px">
            Study: <b style="color:white">{study_time}h/wk</b> · Stress: <b style="color:white">{stress}/10</b> ·
            Midterm: <b style="color:white">{midterm}</b> · Quiz Avg: <b style="color:white">{quiz_avg}</b>
            </span></div>""", unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Leaderboard ───────────────────────────────────────────────────────
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">🏆 Top Performing Students</p>', unsafe_allow_html=True)

    clean_cols_logs = ["Date", "Study_Hours_per_Week", "Stress_Level (1-10)",
                       "Midterm_Score", "Quizzes_Avg", "Subject"]
    clean_cols_df   = ["Student_ID", "Study_Hours_per_Week", "Midterm_Score",
                       "Quizzes_Avg", "Final_Score", "Grade", "Department"]

    if not logs.empty and "Quizzes_Avg" in logs.columns:
        logs_clean = logs.dropna(axis=1, how="all")
        keep = [c for c in clean_cols_logs if c in logs_clean.columns]
        lb = logs_clean.sort_values("Quizzes_Avg", ascending=False)[keep].head(5).reset_index(drop=True)
        lb.index = lb.index + 1
        st.dataframe(lb, use_container_width=True)
    elif not df.empty:
        keep = [c for c in clean_cols_df if c in df.columns]
        lb = df.sort_values("Quizzes_Avg", ascending=False)[keep].head(5).reset_index(drop=True)
        lb.index = lb.index + 1
        st.dataframe(lb, use_container_width=True)
    else:
        st.info("No data available.")

    st.markdown("</div>", unsafe_allow_html=True)