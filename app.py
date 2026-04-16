import streamlit as st
from streamlit_option_menu import option_menu

import milestone1
import cluster_dashboard
import milestone3_recommendation
import milestone4_student_admin_ui
import login
import register
import database

# ✅ FIX 1: set_page_config must be the VERY FIRST Streamlit call
st.set_page_config(page_title="Study AI System", layout="wide")

# ✅ FIX 2: Initialize session state BEFORE any logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

# ✅ Create DB table
database.create_table()

# --------------------------------------------------
# Sidebar Menu (only show when NOT logged in)
# --------------------------------------------------

if not st.session_state.logged_in:

    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        login.login()

    elif choice == "Register":
        register.register()

# --------------------------------------------------
# Main App (only show when logged in)
# --------------------------------------------------

else:

    # ✅ FIX 3: Logout button in sidebar
    with st.sidebar:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()

    selected = option_menu(
        menu_title="Study AI System",
        options=[
            "EDA",
            "Cluster Dashboard",
            "Recommendation Engine",
            "Student Admin Panel"
        ],
        icons=["graph-up", "bar-chart", "lightbulb", "gear"],
        menu_icon="robot",
        orientation="horizontal"  # ✅ FIX 4: horizontal looks better at top
    )

    if selected == "EDA":
        milestone1.run()

    elif selected == "Cluster Dashboard":
        cluster_dashboard.run()

    elif selected == "Recommendation Engine":
        milestone3_recommendation.run()

    elif selected == "Student Admin Panel":
        # ✅ FIX 5: Safe role check with .get() to avoid KeyError
        if st.session_state.get("role") == "Admin":
            milestone4_student_admin_ui.run()
        else:
            st.error("🔒 Admin Access Only")
            st.info("You are logged in as a Student. Please contact an admin for access.")