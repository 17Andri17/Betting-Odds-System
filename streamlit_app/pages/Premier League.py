import streamlit as st
import numpy as np
import pandas as pd

@st.cache_data
def loadData():
    df=pd.read_csv("../premier_league_21_24_stadiums_weather.csv")
    df['date_x']=pd.to_datetime(df['date_x'])
    return df.sort_values(by='date_x')

def restartStats():
    for i in range (number_of_matches):
        if f"show_row_{i}" in st.session_state:
            st.session_state[f"show_row_{i}"] = False
    st.session_state['round_filter2'] = round_filter2
    st.session_state['round_filter1'] = round_filter1

df=loadData()
df_filtered=df

if 'round_filter1' not in st.session_state:
    st.session_state['round_filter1'] = 1

if 'round_filter2' not in st.session_state:
    st.session_state['round_filter2'] = 38

st.title("Premier League")
filtr1, filtr2, filtr3, filtr4, filtr5 = st.columns(5)

with filtr1:
    season_filter = st.multiselect("Wybierz sezon", options=df_filtered['season'].unique(), on_change=restartStats)
with filtr2:
    round_filter1 = st.slider("Wybierz kolejkę początkową", min_value=1, max_value=max(df_filtered['round']), on_change=restartStats)
with filtr3:
    round_filter2 = st.slider("Wybierz kolejkę końcową", min_value=1, max_value=max(df_filtered['round']), value=38, on_change=restartStats)
with filtr4:
    team_filter = st.multiselect("Wybierz drużynę", options = sorted(df_filtered['home_team'].unique()), on_change=restartStats)
with filtr5:
    number_of_matches = st.slider("Wybierz liczbę wyświetlanych meczów", min_value=10, max_value=100, step=5, on_change=restartStats)

season_filter2=season_filter
team_filter2=team_filter

if season_filter == []:
    season_filter2=df['season'].unique()
if team_filter==[]:
    team_filter2=df['home_team'].unique()
df_filtered=df[(df["round"]>=round_filter1)
               & (df['round']<=round_filter2)
               & (df['season'].isin(season_filter2))
               & ((df['home_team'].isin(team_filter2))
                  | (df['away_team'].isin(team_filter2)))]



def showStats(i):
    st.session_state[f"show_row_{i}"] = not st.session_state.get(f"show_row_{i}", False)

for i in range(min(number_of_matches, df_filtered['home_team'].count())):
    col1,col2,col3,col4,col5,col6,col7 = st.columns(7)
    with col3:
        st.write(df_filtered.iloc[i]['home_team'])
    with col6:
        st.write(df_filtered.iloc[i]['away_team'])
    with col2:
        st.write(df_filtered.iloc[i]['round'])
    with col1:
        st.write(df_filtered.iloc[i]['date_x'])
    with col4:
        st.write(df_filtered.iloc[i]['home_goals_x'])
    with col5:
        st.write(df_filtered.iloc[i]['away_goals_x'])
    with col7:
        st.button(
            "Pokaż statystyki",
            key=f"button_{i}",
            on_click=showStats,
            args=(i,),
        )

    # Wyświetlanie dodatkowych informacji pod wierszem, jeśli jest włączone
    if st.session_state.get(f"show_row_{i}", False):
        st.markdown(f"<p style='text-align: center; font-size: 20px;'>Statystyki meczu {df_filtered.iloc[i]['home_team']}-{df_filtered.iloc[i]['away_team']}:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 20px;'>Sędzia: {df_filtered.iloc[i]['referee']}</p>", unsafe_allow_html=True)
        stats1, name, stats2 = st.columns(3)
        with stats1:
           st.write(df_filtered.iloc[i]["Ball Possession_home"])
           st.write(df_filtered.iloc[i]["Goal Attempts_home"])
           st.write(df_filtered.iloc[i]["Shots on Goal_home"])
           st.write(df_filtered.iloc[i]["Free Kicks_home"])
           st.write(df_filtered.iloc[i]["Corner Kicks_home"])
           st.write(df_filtered.iloc[i]["Offsides_home"])
           st.write(df_filtered.iloc[i]["Goalkeeper Saves_home"])
           st.write(df_filtered.iloc[i]["Fouls_home"])
           st.write(df_filtered.iloc[i]["Yellow Cards_home"])
           st.write(df_filtered.iloc[i]["Red Cards_home"])
           st.write(df_filtered.iloc[i]["Total Passes_home"])
           st.write(df_filtered.iloc[i]["Completed Passes_home"])

        with name:
           st.write("Posiadanie piłki")
           st.write("Strzały")
           st.write("Strzały na bramkę")
           st.write("Rzuty wolne")
           st.write("Rzuty rożne")
           st.write("Spalone")
           st.write("Obrony bramkarzy")
           st.write("Faule")
           st.write("Żółte kartki")
           st.write("Czerwone kartki")
           st.write("Podania")
           st.write("Celne podania")

        with stats2:
           st.write(df_filtered.iloc[i]["Ball Possession_away"])
           st.write(df_filtered.iloc[i]["Goal Attempts_away"])
           st.write(df_filtered.iloc[i]["Shots on Goal_away"])
           st.write(df_filtered.iloc[i]["Free Kicks_away"])
           st.write(df_filtered.iloc[i]["Corner Kicks_away"])
           st.write(df_filtered.iloc[i]["Offsides_away"])
           st.write(df_filtered.iloc[i]["Goalkeeper Saves_away"])
           st.write(df_filtered.iloc[i]["Fouls_away"])
           st.write(df_filtered.iloc[i]["Yellow Cards_away"])
           st.write(df_filtered.iloc[i]["Red Cards_away"])
           st.write(df_filtered.iloc[i]["Total Passes_away"])
           st.write(df_filtered.iloc[i]["Completed Passes_away"])

st.session_state