import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

pg = st.navigation([st.Page("Kursomat.py"),
                    st.Page("pagesVis/Premier League.py"),
                    st.Page("pagesHid/Statystyki Premier League.py")], position = "hidden")

st.sidebar.title("Navigation")

if st.sidebar.button(
            "Strona Główna",
            key=f"Home"
        ):
            st.switch_page("Kursomat.py")

if st.sidebar.button(
            "Premier League",
            key=f"PremierLeague"
        ):
            st.switch_page("pagesVis/Premier League.py")

pg.run()