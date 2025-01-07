import streamlit as st
import numpy as np
import pandas as pd

def navbar():
    cols = st.columns(6)
    with cols[0]:
        if st.button(
            "Strona Główna",
            key=f"HomeH"
        ):
            st.switch_page("Kursomat.py")
    with cols[1]:
        if st.button(
            "Premier League",
            key=f"PremierLeagueH"
        ):
            st.switch_page("pagesVis/Premier League.py")
    with cols[2]:
        st.write("Serie A")
    with cols[3]:
        st.write("Ligue1")
    with cols[4]:
        st.write("Bundesliga")
    with cols[5]:
        st.write("Ligue 1")

navbar()

@st.cache_data
def loadData():
    df = pd.read_csv("../fbref/data/matches_with_rolling_stats_pl.csv")
    df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)

    standings = pd.read_csv("../fbref/standings_pl.csv")
    standings['date']=pd.to_datetime(standings['date'])
    return df, standings

# Sprawdzenie, czy dane są już w session_state
if "dfPL" not in st.session_state:
    df, standings = loadData()
    st.session_state["dfPL"] = df
    st.session_state["standingsPL"] = standings

if st.button(
            "Premier League",
            key=f"premierLeagueTenTego"
        ):
            st.switch_page("pagesVis/Premier League.py")