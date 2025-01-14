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
    df = pd.read_csv("../prepared_data.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # Najpierw konwersja do datetime
    df["date"] = df["date"].astype(str)
    df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    dfPL = df[df["league"] == "pl"]
    dfLL = df[df["league"] == "ll"]
    dfL1 = df[df["league"] == "l1"]
    dfBun = df[df["league"] == "bl"]
    dfSA = df[df["league"] == "sa"]

    players_23_24 = pd.read_csv("../fbref/data/players_pl_23-24_fbref.csv")
    players_22_23 = pd.read_csv("../fbref/data/players_pl_22-23_fbref.csv")
    players_21_22 = pd.read_csv("../fbref/data/players_pl_21-22_fbref.csv")
    players_20_21 = pd.read_csv("../fbref/data/players_pl_20-21_fbref.csv")
    players_19_20 = pd.read_csv("../fbref/data/players_pl_19-20_fbref.csv")
    players_18_19 = pd.read_csv("../fbref/data/players_pl_18-19_fbref.csv")
    players = pd.concat([players_23_24, players_22_23, players_21_22, players_20_21, players_19_20, players_18_19], ignore_index=True)
    players = players.rename(columns={"position": "position_x"})
    players["date"] = pd.to_datetime(players["date"], errors="coerce")  # Najpierw konwersja do datetime
    players["date"] = players["date"].astype(str)

    standings = pd.read_csv("../standings.csv")
    standings['date']=pd.to_datetime(standings['date'])
    standingsPL = standings[standings["league"] == "pl"]
    standingsLL = standings[standings["league"] == "ll"]
    standingsL1 = standings[standings["league"] == "l1"]
    standingsBun = standings[standings["league"] == "bl"]
    standingsSA = standings[standings["league"] == "sa"]

    odds = pd.read_csv("../odds.csv")
    oddsPL = odds[odds["Div"] == "E0"]
    oddsLL = odds[odds["Div"] == "SP1"]
    oddsL1 = odds[odds["Div"] == "F1"]
    oddsBun = odds[odds["Div"] == "D1"]
    oddsSA = odds[odds["Div"] == "I1"]

    return dfPL, dfLL, dfL1, dfBun, dfSA, standingsPL, standingsLL, standingsL1, standingsBun, standingsSA, players, oddsPL, oddsLL, oddsL1, oddsBun, oddsSA

# Sprawdzenie, czy dane są już w session_state
if "dfPL" not in st.session_state:
    dfPL, dfLL, dfL1, dfBun, dfSA, standingsPL, standingsLL, standingsL1, standingsBun, standingsSA, players, oddsPL, oddsLL, oddsL1, oddsBun, oddsSA = loadData()
    st.session_state["dfPL"] = dfPL
    st.session_state["standingsPL"] = standingsPL
    st.session_state["playersPL"] = players
    st.session_state["oddsPL"] = oddsPL

    st.session_state["dfLL"] = dfLL
    st.session_state["standingsLL"] = standingsLL
    #st.session_state["playersPL"] = players
    st.session_state["oddsLL"] = oddsPL

    st.session_state["dfL1"] = dfL1
    st.session_state["standingsL1"] = standingsL1
    #st.session_state["playersPL"] = players
    st.session_state["oddsL1"] = oddsL1

    st.session_state["dfBun"] = dfBun
    st.session_state["standingsBun"] = standingsBun
    #st.session_state["playersPL"] = players
    st.session_state["oddsBun"] = oddsBun

    st.session_state["dfSA"] = dfSA
    st.session_state["standingsSA"] = standingsSA
    #st.session_state["playersPL"] = players
    st.session_state["oddsSA"] = oddsSA

if st.button(
            "Premier League",
            key=f"premierLeagueTenTego"
        ):
            st.switch_page("pagesVis/Premier League.py")