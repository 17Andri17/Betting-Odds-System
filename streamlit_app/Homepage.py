import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

homepage = st.Page(
    "Kursomat.py",
    title="Strona Główna",
    icon="🏠",
    default=True,
)

premier_league = st.Page(
    "pagesVis/Premier League.py",
    title="Premier League",
    icon="⚽",
)

premier_league_stats = st.Page(
    "pagesHid/Statystyki Premier League.py",
    title="Statystyki Premier League",
    icon="📊",
)

pg = st.navigation(pages = [homepage, premier_league, premier_league_stats], position="hidden")

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