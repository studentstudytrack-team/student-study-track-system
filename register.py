import streamlit as st
import database

def register():

    st.title("Student Registration")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    role = st.selectbox(
        "Select Role",
        ["Student","Admin"]
    )

    if st.button("Register"):

        database.add_user(username,password,role)

        st.success("Account Created Successfully")