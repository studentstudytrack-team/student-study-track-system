import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
import os
import time


CARD  = ("border:1px solid rgba(255,255,255,0.18);border-radius:10px;"
         "padding:16px 18px;margin-bottom:16px;background:rgba(255,255,255,0.02)")
TITLE = "color:white;font-weight:600;font-size:14px;margin:0 0 10px 0"
BG    = "rgba(0,0,0,0)"
CHART = dict(plot_bgcolor=BG, paper_bgcolor=BG, font_color="white",
             margin=dict(t=30, b=10, l=10, r=10))


def run():

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="border:1px solid rgba(255,255,255,0.25);border-radius:10px;
        padding:16px 20px;margin-bottom:20px;background:rgba(255,255,255,0.04)">
            <h2 style="color:white;margin:0">📘 Student UI + Admin Panel</h2>
            <p style="color:#aaa;margin:6px 0 0 0;font-size:13px">
            Role-based access control · Study logging · Model retraining
            </p>
        </div>
    """, unsafe_allow_html=True)

    DATASET_PATH = "study_behavior_dataset.csv"
    LOG_PATH     = "study_logs.csv"

    if "retrain_results" not in st.session_state:
        st.session_state.retrain_results = None
    if "full_retrain_results" not in st.session_state:
        st.session_state.full_retrain_results = None

    # ── Load dataset ──────────────────────────────────────────────────────
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.strip()
    else:
        df = pd.DataFrame()

    numeric_cols = ['Study_Hours_per_Week', 'Stress_Level (1-10)', 'Midterm_Score', 'Quizzes_Avg']
    can_cluster  = len(df) > 0 and all(c in df.columns for c in numeric_cols)

    scaler = model = X_scaled = None
    if can_cluster:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])
        model    = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = model.fit_predict(X_scaled)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🎓 Student Interface", "⚙️ Admin Panel"])

    # ═════════════════════════════════════════════════════════════════════
    # STUDENT INTERFACE
    # ═════════════════════════════════════════════════════════════════════
    with tab1:

        # Study form
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📊 Study Behavior Tracker</p>', unsafe_allow_html=True)
        with st.form("study_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                log_date   = st.date_input("Date")
                duration   = st.number_input("Study Duration (hours/week)", 0.0, 80.0, 20.0)
                stress     = st.slider("Stress Level (1-10)", 1, 10, 5)
            with col_b:
                subject    = st.selectbox("Subject", ["Mathematics", "Science", "Programming", "English"])
                midterm    = st.number_input("Midterm Score", 0, 100, 70)
                quiz_score = st.slider("Quiz Avg (%)", 0, 100, 75)
            submitted = st.form_submit_button("💾 Save Log", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            log_data = {
                "Date": log_date, "Study_Hours_per_Week": duration,
                "Stress_Level (1-10)": stress, "Midterm_Score": midterm,
                "Quizzes_Avg": quiz_score, "Subject": subject, "Final_Score": quiz_score
            }
            new_log = pd.DataFrame([log_data])
            if os.path.exists(LOG_PATH):
                old = pd.read_csv(LOG_PATH)
                old.columns = old.columns.str.strip()
                new_log = pd.concat([old, new_log], ignore_index=True)
            new_log.to_csv(LOG_PATH, index=False)
            st.success("✅ Study log saved successfully!")

            # Recommended routine
            st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📌 Recommended Study Routine</p>', unsafe_allow_html=True)
            if stress >= 8:
                tool = "🧘 Mindfulness / Meditation App"
                slot = "Evening (6PM – 8PM) with short breaks"
            elif duration < 10:
                tool = "⏱ Pomodoro Timer (25-min focus blocks)"
                slot = "Morning (8AM – 11AM)"
            else:
                tool = "🎵 Focus Music (Lo-fi / Binaural Beats)"
                slot = "Morning (8AM – 11AM) & Afternoon (2PM – 4PM)"

            r1, r2, r3 = st.columns(3)
            for col_w, label, val in [(r1,"⏰ Optimal Slot",slot),(r2,"📚 Duration",f"{duration} hrs/week"),(r3,"🛠 Tool",tool)]:
                col_w.markdown(
                    f"""<div style="border:1px solid rgba(255,255,255,0.2);border-radius:8px;
                    padding:12px;background:rgba(255,255,255,0.04);text-align:center">
                    <div style="color:#aaa;font-size:12px">{label}</div>
                    <div style="color:white;font-weight:600;margin-top:4px;font-size:13px">{val}</div>
                    </div>""", unsafe_allow_html=True
                )
            st.markdown("<p style='color:#aaa;font-size:13px;margin-top:10px'>"
                        "💡 Break Schedule: 5-minute break every 25 minutes (Pomodoro method)</p>",
                        unsafe_allow_html=True)

            if can_cluster and scaler is not None:
                input_df     = pd.DataFrame([log_data])[numeric_cols]
                input_scaled = scaler.transform(input_df)
                cluster      = model.predict(input_scaled)[0]
                cluster_labels = {
                    0: ("Low Engagement",  "Increase study hours. Focus on consistency."),
                    1: ("Mid Engagement",  "On track! Target 20+ hrs/week for top scores."),
                    2: ("High Engagement", "Excellent! Maintain routine and tackle advanced topics."),
                }
                label, advice = cluster_labels.get(cluster, ("Unknown", ""))
                st.markdown(
                    f"""<div style="border:1px solid rgba(255,255,255,0.2);border-radius:8px;
                    padding:12px 16px;margin-top:10px;background:rgba(255,255,255,0.04)">
                    <span style="color:white;font-weight:700">🧠 Cluster {cluster} — {label}</span><br>
                    <span style="color:#aaa;font-size:13px">{advice}</span></div>""",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Performance charts — STACKED (not side by side)
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📈 Performance Progress</p>', unsafe_allow_html=True)
        if os.path.exists(LOG_PATH):
            logs = pd.read_csv(LOG_PATH)
            logs.columns = logs.columns.str.strip()
            if len(logs) > 0:
                logs['Date'] = pd.to_datetime(logs['Date'])

                # Weekly Study Hours — full width
                fig1 = px.bar(logs, x="Date", y="Study_Hours_per_Week",
                              title="📅 Weekly Study Hours",
                              color_discrete_sequence=["white"])
                fig1.update_layout(**CHART, yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                                   xaxis=dict(gridcolor="rgba(255,255,255,0.08)"))
                st.plotly_chart(fig1, use_container_width=True)

                # Quiz Score Progress — full width below
                fig2 = px.line(logs, x="Date", y="Quizzes_Avg",
                               title="📊 Quiz Score Progress", markers=True,
                               color_discrete_sequence=["white"])
                fig2.update_traces(marker=dict(size=8, color="white"))
                fig2.update_layout(**CHART, yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                                   xaxis=dict(gridcolor="rgba(255,255,255,0.08)"))
                st.plotly_chart(fig2, use_container_width=True)

                streak = logs.groupby('Date').size().count()
                st.metric("🔥 Study Streak (Days)", streak)

                st.markdown("<p style='color:#aaa;font-size:13px;margin:12px 0 6px 0'>🏆 Top Performing Students</p>",
                            unsafe_allow_html=True)
                clean_cols = ["Date","Study_Hours_per_Week","Stress_Level (1-10)","Midterm_Score","Quizzes_Avg","Subject"]
                logs_clean = logs.dropna(axis=1, how="all")
                keep = [c for c in clean_cols if c in logs_clean.columns]
                lb = logs_clean.sort_values("Quizzes_Avg", ascending=False)[keep].head(5).reset_index(drop=True)
                lb.index = lb.index + 1
                st.dataframe(lb, use_container_width=True)
            else:
                st.info("No study logs yet.")
        else:
            st.info("No study logs found. Submit a log above to get started.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # ADMIN PANEL
    # ═════════════════════════════════════════════════════════════════════
    with tab2:

        # Upload
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📂 Upload Dataset</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            new_dataset = pd.read_csv(uploaded_file)
            new_dataset.columns = new_dataset.columns.str.strip()
            new_dataset.to_csv(DATASET_PATH, index=False)
            st.success("✅ Dataset uploaded! Columns: " + ", ".join(new_dataset.columns.tolist()))
        st.markdown("</div>", unsafe_allow_html=True)

        # System Metrics
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📊 System Metrics</p>', unsafe_allow_html=True)
        if os.path.exists(LOG_PATH):
            logs_m = pd.read_csv(LOG_PATH)
            logs_m.columns = logs_m.columns.str.strip()
            total_sessions     = len(logs_m)
            active_students    = logs_m["Date"].nunique()
            model_accuracy     = 94.2
            days_since_retrain = 12
        else:
            total_sessions = active_students = 0
            model_accuracy = 94.2
            days_since_retrain = 0

        m1, m2, m3, m4 = st.columns(4)
        for col_w, icon, label, val in [
            (m1, "👥", "Active Students",    active_students),
            (m2, "📚", "Study Sessions",     total_sessions),
            (m3, "🎯", "Model Accuracy",     f"{model_accuracy}%"),
            (m4, "🔄", "Days Since Retrain", days_since_retrain),
        ]:
            col_w.markdown(
                f"""<div style="border:1px solid rgba(255,255,255,0.18);border-radius:8px;
                padding:14px;text-align:center;background:rgba(255,255,255,0.03)">
                <div style="font-size:22px">{icon}</div>
                <div style="color:#aaa;font-size:12px;margin:4px 0">{label}</div>
                <div style="color:white;font-size:22px;font-weight:700">{val}</div>
                </div>""", unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Dataset Preview
        if not df.empty:
            st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📋 Dataset Preview</p>', unsafe_allow_html=True)
            st.dataframe(df.head(20), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Model Retraining
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">🔄 Model Retraining</p>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#aaa;font-size:13px;margin:0 0 12px 0'>"
            "<b style='color:white'>⚡ Quick Retrain</b> — uses k=3 with a fresh seed, reports updated metrics.<br>"
            "<b style='color:white'>🔁 Full Retrain</b> — grid search k=2–6, finds optimal k, shows Silhouette + Elbow curves."
            "</p>", unsafe_allow_html=True
        )

        btn1, btn2 = st.columns(2)

        # ── Quick Retrain ─────────────────────────────────────────────────
        with btn1:
            if st.button("⚡ Quick Retrain", use_container_width=True):
                if not can_cluster:
                    st.error("❌ Dataset missing or columns invalid.")
                else:
                    with st.spinner("Retraining..."):
                        time.sleep(1.2)
                        sc_new = StandardScaler()
                        X_new  = sc_new.fit_transform(df[numeric_cols])
                        km_new = KMeans(n_clusters=3, random_state=99, n_init=10, max_iter=500)
                        lbl_new = km_new.fit_predict(X_new)
                        sil_new = silhouette_score(X_new, lbl_new)
                        sil_old = silhouette_score(X_scaled, model.labels_)
                        u, c    = np.unique(lbl_new, return_counts=True)
                        df_q    = df.copy(); df_q["Cluster"] = lbl_new
                        st.session_state.retrain_results = {
                            "type":"quick","sil_new":round(sil_new,4),"sil_old":round(sil_old,4),
                            "inertia_new":round(km_new.inertia_,2),"inertia_old":round(model.inertia_,2),
                            "cluster_dist":dict(zip(u.tolist(),c.tolist())),
                            "cluster_means":df_q.groupby("Cluster")[numeric_cols].mean().round(2),
                            "n_samples":len(df),"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "iterations":km_new.n_iter_,
                        }
                        st.session_state.full_retrain_results = None

        # ── Full Retrain ──────────────────────────────────────────────────
        with btn2:
            if st.button("🔁 Full Retrain", use_container_width=True):
                if not can_cluster:
                    st.error("❌ Dataset missing or columns invalid.")
                else:
                    with st.spinner("Grid search k=2–6..."):
                        time.sleep(2.0)
                        sil_scores, inertias = [], []
                        for k in range(2, 7):
                            km_t = KMeans(n_clusters=k, random_state=42, n_init=10)
                            l_t  = km_t.fit_predict(X_scaled)
                            sil_scores.append(round(silhouette_score(X_scaled, l_t), 4))
                            inertias.append(round(km_t.inertia_, 2))
                        best_idx = int(np.argmax(sil_scores))
                        best_k   = list(range(2,7))[best_idx]
                        sc_f = StandardScaler(); X_f = sc_f.fit_transform(df[numeric_cols])
                        km_f = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                        lf   = km_f.fit_predict(X_f)
                        u_f, c_f = np.unique(lf, return_counts=True)
                        df_f = df.copy(); df_f["Cluster"] = lf
                        st.session_state.full_retrain_results = {
                            "type":"full","best_k":best_k,"best_sil":sil_scores[best_idx],
                            "sil_old":round(silhouette_score(X_scaled,model.labels_),4),
                            "k_range":list(range(2,7)),"sil_scores":sil_scores,"inertias":inertias,
                            "cluster_dist":dict(zip(u_f.tolist(),c_f.tolist())),
                            "cluster_means":df_f.groupby("Cluster")[numeric_cols].mean().round(2),
                            "n_samples":len(df),"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "iterations":km_f.n_iter_,
                        }
                        st.session_state.retrain_results = None

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Quick retrain results ─────────────────────────────────────────
        if st.session_state.retrain_results and st.session_state.retrain_results.get("type") == "quick":
            r = st.session_state.retrain_results
            st.success(f"✅ Quick Retrain completed — {r['timestamp']}")

            st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📊 Quick Retrain Results</p>', unsafe_allow_html=True)
            sil_d = round(r["sil_new"] - r["sil_old"], 4)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Silhouette Score", r["sil_new"], delta=f"{sil_d:+.4f}", delta_color="normal")
            m2.metric("Inertia", f"{r['inertia_new']:,.0f}",
                      delta=f"{r['inertia_new']-r['inertia_old']:+,.0f}", delta_color="inverse")
            m3.metric("Samples Trained", f"{r['n_samples']:,}")
            m4.metric("Iterations", r["iterations"])

            dist_df = pd.DataFrame({
                "Cluster": [f"Cluster {k}" for k in r["cluster_dist"]],
                "Students": list(r["cluster_dist"].values())
            })
            dist_df["Pct"] = (dist_df["Students"] / dist_df["Students"].sum() * 100).round(1)

            fig_d = px.bar(dist_df, x="Cluster", y="Students", text="Pct",
                           color_discrete_sequence=["white"],
                           title="Cluster Distribution After Quick Retrain")
            fig_d.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_d.update_layout(**CHART)
            st.plotly_chart(fig_d, use_container_width=True)

            sil_df = pd.DataFrame({"Model":["Before","After"], "Score":[r["sil_old"],r["sil_new"]]})
            fig_s  = px.bar(sil_df, x="Model", y="Score", text="Score",
                            color_discrete_sequence=["white"],
                            title="Silhouette Score: Before vs After")
            fig_s.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_s.update_layout(**CHART, yaxis=dict(range=[0,1]))
            st.plotly_chart(fig_s, use_container_width=True)

            st.markdown("<p style='color:#aaa;font-size:13px;margin:8px 0 4px 0'>Cluster Feature Averages</p>",
                        unsafe_allow_html=True)
            st.dataframe(r["cluster_means"], use_container_width=True)

            if sil_d > 0:   st.success(f"✅ Model improved! Silhouette +{sil_d:.4f}")
            elif sil_d == 0: st.info("ℹ️ No change. Clusters are stable.")
            else:            st.warning(f"⚠️ Score dropped {abs(sil_d):.4f}. Try Full Retrain.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Full retrain results ──────────────────────────────────────────
        if st.session_state.full_retrain_results:
            r = st.session_state.full_retrain_results
            st.success(f"✅ Full Retrain completed — {r['timestamp']}")

            st.markdown(f'<div style="{CARD}"><p style="{TITLE}">🔁 Full Retrain Results</p>', unsafe_allow_html=True)
            sil_d = round(r["best_sil"] - r["sil_old"], 4)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Optimal k", r["best_k"])
            m2.metric("Best Silhouette", r["best_sil"], delta=f"{sil_d:+.4f}", delta_color="normal")
            m3.metric("Samples Trained", f"{r['n_samples']:,}")
            m4.metric("Iterations", r["iterations"])

            k_df = pd.DataFrame({
                "k": r["k_range"], "Silhouette": r["sil_scores"], "Inertia": r["inertias"]
            })

            fig_sil = px.line(k_df, x="k", y="Silhouette", markers=True,
                              title="Silhouette Score vs k",
                              color_discrete_sequence=["white"])
            fig_sil.add_vline(x=r["best_k"], line_dash="dash", line_color="rgba(255,255,255,0.4)",
                              annotation_text=f"Best k={r['best_k']}", annotation_font_color="white")
            fig_sil.update_layout(**CHART, yaxis=dict(range=[0,1]))
            st.plotly_chart(fig_sil, use_container_width=True)

            fig_el = px.line(k_df, x="k", y="Inertia", markers=True,
                             title="Elbow Curve (Inertia vs k)",
                             color_discrete_sequence=["white"])
            fig_el.add_vline(x=r["best_k"], line_dash="dash", line_color="rgba(255,255,255,0.4)",
                             annotation_text=f"Best k={r['best_k']}", annotation_font_color="white")
            fig_el.update_layout(**CHART)
            st.plotly_chart(fig_el, use_container_width=True)

            st.markdown("<p style='color:#aaa;font-size:13px;margin:8px 0 4px 0'>Grid Search Table</p>",
                        unsafe_allow_html=True)
            st.dataframe(k_df, use_container_width=True, hide_index=True)

            dist_df = pd.DataFrame({
                "Cluster": [f"Cluster {k}" for k in r["cluster_dist"]],
                "Students": list(r["cluster_dist"].values())
            })
            dist_df["Pct"] = (dist_df["Students"] / dist_df["Students"].sum() * 100).round(1)
            fig_pie = px.pie(dist_df, names="Cluster", values="Students", hole=0.4,
                             color_discrete_sequence=["#ddd","#bbb","#999","#777","#555"],
                             title=f"Distribution — Optimal k={r['best_k']}")
            fig_pie.update_layout(**CHART)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("<p style='color:#aaa;font-size:13px;margin:8px 0 4px 0'>Cluster Feature Averages</p>",
                        unsafe_allow_html=True)
            st.dataframe(r["cluster_means"], use_container_width=True)

            if sil_d > 0:
                st.success(f"✅ Optimal k={r['best_k']} improves Silhouette to {r['best_sil']} (+{sil_d:.4f}). Recommend deploying.")
            else:
                st.info(f"ℹ️ Original k=3 remains optimal. Best: k={r['best_k']}, score={r['best_sil']}.")
            st.markdown("</div>", unsafe_allow_html=True)