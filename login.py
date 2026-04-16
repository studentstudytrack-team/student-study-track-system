import streamlit as st
import database

def login():

    st.title("AI Study System Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        result = database.login_user(username,password)

        if result:

            st.session_state.logged_in = True
            st.session_state.role = result[2]

            st.success("Login Successful")

        else:

            st.error("Invalid Credentials")