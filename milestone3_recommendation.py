import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import date

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

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
            <h2 style="color:white;margin:0">📘 Intelligent Study Recommendation Engine</h2>
            <p style="color:#aaa;margin:6px 0 0 0;font-size:13px">
            Personalized study plans · K-Means clustering · Linear Regression · GPT-4o-mini Tutor
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Load & model ──────────────────────────────────────────────────────
    df = pd.read_csv("study_behavior_dataset.csv")
    df.columns = df.columns.str.strip()

    numeric_cols = ['Study_Hours_per_Week', 'Stress_Level (1-10)', 'Midterm_Score', 'Quizzes_Avg']

    reg_model = LinearRegression()
    reg_model.fit(df[['Study_Hours_per_Week']], df['Quizzes_Avg'])
    predicted_score = round(
        reg_model.predict(pd.DataFrame([[20]], columns=['Study_Hours_per_Week']))[0], 2
    )

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
    kmeans   = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Predicted score banner
    st.markdown(
        f"""<div style="border:1px solid rgba(255,255,255,0.2);border-radius:10px;
        padding:12px 18px;margin-bottom:16px;background:rgba(255,255,255,0.03)">
        📊 <span style="color:#aaa">Predicted Quiz Avg if student studies</span>
        <b style="color:white"> 20 hrs/week</b>:
        <b style="color:white;font-size:20px"> {predicted_score}</b>
        </div>""", unsafe_allow_html=True
    )

    # ── Student selector ──────────────────────────────────────────────────
    student_id = st.selectbox("🎓 Select Student ID", df["Student_ID"])
    student    = df[df["Student_ID"] == student_id].iloc[0]
    cluster    = int(student["Cluster"])
    stress     = student["Stress_Level (1-10)"]
    weekly_hrs = student["Study_Hours_per_Week"]
    daily_hrs  = round(weekly_hrs / 5, 1)
    quiz_avg   = student["Quizzes_Avg"]
    midterm    = student["Midterm_Score"]

    if stress >= 8:
        selected_tool = "Mindfulness App"
    elif weekly_hrs < 10:
        selected_tool = "Pomodoro Timer"
    elif quiz_avg > 85:
        selected_tool = "Mock Test Platforms"
    else:
        selected_tool = "Focus Music"

    slots_map  = {0: ["8:00–10:00 AM", "2:00–3:30 PM"],
                  1: ["10:00–12:00 PM", "6:00–7:30 PM"],
                  2: ["7:00–8:30 PM",   "9:00–10:30 PM"]}
    time_slots = slots_map.get(cluster, ["8:00–10:00 AM"])

    improvement   = quiz_avg - midterm
    effectiveness = round(max(improvement, 5), 1)

    if quiz_avg < 50:
        st.error("⚠️ Low performance detected. This student needs to increase study time immediately.")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 1 — Personalized Study Plan
    # ═════════════════════════════════════════════════════════════════════
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📌 Personalized Study Plan</p>', unsafe_allow_html=True)

    plan_col, tool_col = st.columns([3, 2])

    with plan_col:
        # Time slots
        slot_html = "".join([f"<li style='color:#ddd;margin:5px 0'>⏰ {t}</li>" for t in time_slots])
        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.12);border-radius:8px;
            padding:12px 16px;margin-bottom:10px;background:rgba(255,255,255,0.02)">
            <p style="color:#ddd;font-weight:600;margin:0 0 8px 0">🕒 Optimal Study Time Slots</p>
            <ul style="margin:0;padding-left:18px">{slot_html}</ul></div>""",
            unsafe_allow_html=True
        )
        # Daily target
        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.12);border-radius:8px;
            padding:12px 16px;margin-bottom:10px;background:rgba(255,255,255,0.02)">
            <p style="color:#ddd;font-weight:600;margin:0 0 4px 0">⏳ Daily Study Target</p>
            <span style="color:white;font-size:22px;font-weight:700">{daily_hrs} hrs/day</span>
            <span style="color:#aaa;font-size:13px"> ({weekly_hrs} hrs/week)</span></div>""",
            unsafe_allow_html=True
        )
        # Stress bar
        stress_pct   = int(stress) * 10
        stress_color = "white" if stress < 4 else "#f87171" if stress >= 7 else "#fbbf24"
        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.12);border-radius:8px;
            padding:12px 16px;margin-bottom:10px;background:rgba(255,255,255,0.02)">
            <p style="color:#ddd;font-weight:600;margin:0 0 8px 0">😰 Stress Level:
            <span style="color:{stress_color}">{int(stress)}/10</span></p>
            <div style="background:rgba(255,255,255,0.08);border-radius:6px;height:10px">
            <div style="background:{stress_color};width:{stress_pct}%;height:10px;border-radius:6px"></div>
            </div></div>""",
            unsafe_allow_html=True
        )
        # Effectiveness bar
        eff_pct = min(int(effectiveness), 100)
        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.12);border-radius:8px;
            padding:12px 16px;background:rgba(255,255,255,0.02)">
            <p style="color:#ddd;font-weight:600;margin:0 0 8px 0">📈 Expected Improvement:
            <span style="color:white">{effectiveness}%</span></p>
            <div style="background:rgba(255,255,255,0.08);border-radius:6px;height:10px">
            <div style="background:white;width:{eff_pct}%;height:10px;border-radius:6px"></div>
            </div></div>""",
            unsafe_allow_html=True
        )

    with tool_col:
        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.15);border-radius:8px;
            padding:14px;background:rgba(255,255,255,0.02)">
            <p style="color:#ddd;font-weight:600;margin:0 0 12px 0">🛠 Tools & Strategies</p>""",
            unsafe_allow_html=True
        )
        all_tools = [("⏱", "Pomodoro Timer"), ("🎵", "Focus Music"),
                     ("🧘", "Mindfulness App"), ("📝", "Mock Test Platforms")]
        for icon, name in all_tools:
            if name == selected_tool:
                st.markdown(
                    f"""<div style="border:2px solid rgba(255,255,255,0.6);border-radius:8px;
                    padding:10px 14px;margin-bottom:8px;background:rgba(255,255,255,0.08)">
                    <span style="color:white;font-weight:700">{icon} {name}</span>
                    <span style="float:right;border:1px solid rgba(255,255,255,0.4);color:white;
                    border-radius:12px;padding:1px 9px;font-size:11px">✓ Recommended</span>
                    </div>""", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="border:1px solid rgba(255,255,255,0.1);border-radius:8px;
                    padding:10px 14px;margin-bottom:8px;background:rgba(255,255,255,0.02);opacity:0.55">
                    <span style="color:#aaa">{icon} {name}</span>
                    </div>""", unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 2 — Weekly Schedule (STACKED, not side by side)
    # ═════════════════════════════════════════════════════════════════════
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📅 Weekly Study Schedule</p>', unsafe_allow_html=True)

    weekly_df = pd.DataFrame({
        "Day":   ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Hours": [
            round(weekly_hrs/5*0.8, 2), round(weekly_hrs/5, 2),
            round(weekly_hrs/5*1.1, 2), round(weekly_hrs/5*0.7, 2),
            round(weekly_hrs/5, 2),     round(weekly_hrs/5*0.6, 2),
            round(weekly_hrs/5*0.5, 2),
        ]
    })
    fig_bar = px.bar(weekly_df, x="Day", y="Hours",
                     color="Hours", color_continuous_scale=["#555", "#fff"],
                     text="Hours", title="Hours per Day (Mon–Sun)")
    fig_bar.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
    fig_bar.update_layout(**CHART, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 3 — Performance Forecast (STACKED)
    # ═════════════════════════════════════════════════════════════════════
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📊 Expected Performance Improvement</p>', unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        "Week":  ["Current", "Week 1", "Week 2", "Week 3", "Week 4"],
        "Score": [quiz_avg, quiz_avg+2, quiz_avg+4, quiz_avg+7, quiz_avg+10]
    })
    fig_line = px.line(perf_df, x="Week", y="Score", markers=True,
                       color_discrete_sequence=["white"])
    fig_line.update_traces(marker=dict(size=10, color="white", line=dict(width=2, color="white")),
                           line=dict(width=2))
    fig_line.update_layout(**CHART, yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                           xaxis=dict(gridcolor="rgba(255,255,255,0.08)"))
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 4 — Subject Recommendations
    # ═════════════════════════════════════════════════════════════════════
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📚 Subject-wise Recommendations</p>', unsafe_allow_html=True)

    subject_plans = {
        0: [("Math","2.0","Active recall + Flashcards — Cluster 0 shows weak quantitative scores"),
            ("Science","1.5","Video lectures + concept mapping recommended"),
            ("English","1.0","Reading comprehension exercises daily"),
            ("Programming","1.5","Practice 2 LeetCode problems/day")],
        1: [("Math","1.5","Practice problem sets — scores near average, push higher"),
            ("Science","1.5","Lab simulations + past papers"),
            ("English","1.0","Essay writing + grammar drills"),
            ("Programming","2.0","Build mini-projects to solidify concepts")],
        2: [("Math","1.0","Advanced problem solving — maintain high performance"),
            ("Science","1.5","Focus on exam-level questions and peer teaching"),
            ("English","0.5","Quick revision sufficient — already strong"),
            ("Programming","2.5","Competitive programming + system design challenges")],
    }
    plans    = subject_plans.get(cluster, subject_plans[1])
    sub_cols = st.columns(4)
    for col_w, (subj, hrs, tip) in zip(sub_cols, plans):
        col_w.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.2);border-radius:8px;
            padding:14px;background:rgba(255,255,255,0.03);min-height:150px">
            <div style="color:white;font-weight:700;font-size:14px">{subj}</div>
            <div style="color:white;font-size:24px;font-weight:700;margin:6px 0">{hrs}
            <span style="font-size:12px;color:#aaa">hrs/day</span></div>
            <div style="color:#aaa;font-size:12px;line-height:1.4">{tip}</div>
            </div>""", unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 5 — Exam Countdown + Adaptive Feedback
    # ═════════════════════════════════════════════════════════════════════
    ef_col1, ef_col2 = st.columns(2)

    with ef_col1:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">⏳ Exam Countdown Planner</p>', unsafe_allow_html=True)
        today    = date.today()
        nxt_month = today.month + 1 if today.month < 12 else 1
        nxt_year  = today.year if today.month < 12 else today.year + 1
        exam_date = st.date_input("Select Exam Date",
                                  value=date(nxt_year, nxt_month, 15),
                                  min_value=today)
        days_left = (exam_date - today).days

        if days_left > 30:
            msg, note = f"{days_left} days", "🟢 Plenty of time — maintain steady pace"
        elif days_left > 7:
            msg, note = f"{days_left} days", "🟡 Moderate — increase revision intensity"
        elif days_left > 0:
            msg, note = f"{days_left} days", "🔴 URGENT — full revision mode now!"
        else:
            msg, note = "—", "ℹ️ Please select a future date"

        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.18);border-radius:8px;
            padding:14px;background:rgba(255,255,255,0.03)">
            <span style="color:white;font-size:26px;font-weight:700">{msg}</span>
            <span style="color:#aaa"> until exam</span><br>
            <span style="color:#aaa;font-size:13px">{note}</span></div>""",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with ef_col2:
        st.markdown(f'<div style="{CARD}"><p style="{TITLE}">📝 Adaptive Feedback</p>', unsafe_allow_html=True)
        feedback = st.radio("Was this plan helpful?", ["✅ Helpful", "❌ Not Helpful"], horizontal=True)
        if "Not" not in feedback:
            msg_fb = "🎉 Great! We will reinforce this study pattern for your cluster."
        else:
            msg_fb = "🔄 Noted! We will adjust recommendations in the next retrain cycle."
        st.markdown(
            f"""<div style="border:1px solid rgba(255,255,255,0.18);border-radius:8px;
            padding:12px 16px;background:rgba(255,255,255,0.03)">
            <span style="color:white;font-size:13px">{msg_fb}</span></div>""",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 6 — AI Study Tutor (chat)
    # ═════════════════════════════════════════════════════════════════════
    st.markdown(f'<div style="{CARD}"><p style="{TITLE}">🤖 AI Study Tutor</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;font-size:13px;margin:0 0 12px 0'>"
        "Ask any study-related question — the tutor knows your profile, cluster, scores, and stress level."
        "</p>", unsafe_allow_html=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    cluster_name_map = {0: "Low Engagement", 1: "Mid Engagement", 2: "High Engagement"}
    system_ctx = (
        f"Student Profile: ID={student_id}, Cluster={cluster} ({cluster_name_map.get(cluster,'')}), "
        f"Study={weekly_hrs}hrs/wk, Stress={int(stress)}/10, Midterm={midterm}, Quiz={quiz_avg}%, "
        f"Recommended Tool={selected_tool}, Slots={', '.join(time_slots)}. "
        "You are a friendly academic tutor. Give specific, concise, actionable advice (3-5 sentences)."
    )

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"""<div style="display:flex;justify-content:flex-end;margin:6px 0">
                <div style="border:1px solid rgba(255,255,255,0.25);border-radius:12px 12px 2px 12px;
                padding:10px 14px;max-width:75%;color:white;font-size:14px;
                background:rgba(255,255,255,0.07)">💬 {msg['content']}</div></div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="display:flex;justify-content:flex-start;margin:6px 0">
                <div style="border:1px solid rgba(255,255,255,0.18);border-radius:12px 12px 12px 2px;
                padding:10px 14px;max-width:80%;color:#ddd;font-size:14px;
                background:rgba(255,255,255,0.03)">🤖 {msg['content']}</div></div>""",
                unsafe_allow_html=True
            )

    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            question = st.text_input("Question", placeholder="e.g. How can I reduce exam stress?",
                                     label_visibility="collapsed")
        with btn_col:
            send = st.form_submit_button("Send ➤", use_container_width=True)

    if send and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})

        if client is None or not os.getenv("OPENAI_API_KEY"):
            fallbacks = {
                "stress": f"Your stress is {int(stress)}/10. Try {selected_tool}. Break study into 25-min Pomodoro blocks with 5-min breaks.",
                "study":  f"You study {weekly_hrs}hrs/wk. Best slot: {time_slots[0]}. Target {daily_hrs}hrs/day consistently.",
                "score":  f"Quiz avg is {quiz_avg}%. With the 4-week plan, expect +10 points through consistent daily effort.",
                "exam":   "Start with your weakest subject daily. Use active recall and spaced repetition, not passive reading.",
            }
            resp = next((v for k, v in fallbacks.items() if k in question.lower()),
                        f"Cluster {cluster} student ({weekly_hrs}hrs/wk, stress {int(stress)}/10): "
                        f"Use {selected_tool} during {time_slots[0]}. Stay consistent at {daily_hrs}hrs/day.")
        else:
            try:
                msgs = [{"role": "system", "content": system_ctx}]
                msgs += st.session_state.chat_history[-6:]
                response = client.chat.completions.create(
                    model="gpt-4o-mini", messages=msgs, max_tokens=300, temperature=0.7
                )
                resp = response.choices[0].message.content
            except Exception as e:
                resp = f"⚠️ AI Tutor unavailable: {e}. Check your OPENAI_API_KEY in .env file."

        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)