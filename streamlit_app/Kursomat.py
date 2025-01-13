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

    players_23_24 = pd.read_csv("../fbref/data/players_pl_23-24_fbref.csv")
    players_22_23 = pd.read_csv("../fbref/data/players_pl_22-23_fbref.csv")
    players_21_22 = pd.read_csv("../fbref/data/players_pl_21-22_fbref.csv")
    players_20_21 = pd.read_csv("../fbref/data/players_pl_20-21_fbref.csv")
    players_19_20 = pd.read_csv("../fbref/data/players_pl_19-20_fbref.csv")
    players_18_19 = pd.read_csv("../fbref/data/players_pl_18-19_fbref.csv")
    players = pd.concat([players_23_24, players_22_23, players_21_22, players_20_21, players_19_20, players_18_19], ignore_index=True)
    players = players.rename(columns={"position": "position_x"})

    standings = pd.read_csv("../fbref/standings_pl.csv")
    standings['date']=pd.to_datetime(standings['date'])

    odds_23_24 = pd.read_csv("../fbref/data/E0_23-24.csv")
    odds_22_23 = pd.read_csv("../fbref/data/E0_22-23.csv")
    odds_21_22 = pd.read_csv("../fbref/data/E0_21-22.csv")
    odds_20_21 = pd.read_csv("../fbref/data/E0_20-21.csv")
    odds_19_20 = pd.read_csv("../fbref/data/E0_19-20.csv")
    odds_18_19 = pd.read_csv("../fbref/data/E0_18-19.csv")
    odds = pd.concat([odds_23_24, odds_22_23, odds_21_22, odds_20_21, odds_19_20, odds_18_19], ignore_index=True)

    return df, standings, players, odds

# Sprawdzenie, czy dane są już w session_state
if "dfPL" not in st.session_state:
    df, standings, players, odds = loadData()
    st.session_state["dfPL"] = df
    st.session_state["standingsPL"] = standings
    st.session_state["playersPL"] = players
    st.session_state["oddsPL"] = odds

if st.button(
            "Premier League",
            key=f"premierLeagueTenTego"
        ):
            st.switch_page("pagesVis/Premier League.py")