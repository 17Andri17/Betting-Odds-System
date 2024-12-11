import streamlit as st
import numpy as np
import pandas as pd
import runpy

# Ustawianie domyślnego wyglądu strony na wide
st.set_page_config(
    layout="wide"
)

# ładowanie danych
@st.cache_data
def loadData():
    df=pd.read_csv("../premier_league_21_24_stadiums_weather.csv")
    df['date_x']=pd.to_datetime(df['date_x'])
    df['date_x'] = df['date_x'].dt.strftime('%Y-%m-%d %H:%M')

    standings = pd.read_csv("../fbref/standings_pl.csv")
    standings['date']=pd.to_datetime(standings['date'])
    return df.sort_values(by='date_x', ascending=False), standings

# Chowanie statystyk po zmianie filtrów
def restartStats():
    for i in range (number_of_matches):
        if f"show_row_{i}" in st.session_state:
            st.session_state[f"show_row_{i}"] = False
    st.session_state['round_filter2'] = round_filter2
    st.session_state['round_filter1'] = round_filter1

# Pokazywanie statystyk dla i-tego meczu
def showStats(i):
    st.session_state[f"show_row_{i}"] = not st.session_state.get(f"show_row_{i}", False)

def showDateButton():
    if len(season_filter) == 0:
        st.session_state["show_table"] = True
    else:
        st.session_state["show_table"] = False
    restartStats()

# Centrowanie tekstów na całej stronie
st.markdown("""
    <style>
    .center-text {
        text-align: center;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)


df, standings=loadData()
df_filtered=df
standings_filtered=standings

# Na razie nieużywane - do zmiany suwaków filtrowych
if 'round_filter1' not in st.session_state:
    st.session_state['round_filter1'] = 1

if 'round_filter2' not in st.session_state:
    st.session_state['round_filter2'] = 38

# Tytuł i tworzenie filtrów
st.title("Premier League")
# Wybieranie tabeli
season_filter = st.multiselect("Wybierz sezon, z którego chcesz zobaczyć tabelę oraz statystyki",
    options = standings['season'].unique(), on_change=showDateButton, max_selections=1)

season_filter2 = season_filter

if season_filter == []:
    season_filter2=sorted(standings['season'].unique(), reverse=True)[0]
    season_filter_matches = season_filter2[2:4] + '-' + season_filter2[7:9]
else:
    season_filter_matches = season_filter2[0][2:4] + '-' + season_filter2[0][7:9]

date_standings = max(standings_filtered['date'].dt.strftime('%Y-%m-%d'))
selected_season = season_filter2

if st.session_state.get("show_table", False):
    selected_season = season_filter2[0]
    standings_filtered = standings[standings['season'] == selected_season]
    date_standings = st.date_input("Wybierz datę tabeli",
        min_value = min(standings_filtered['date']),
        max_value = max(standings_filtered['date']),
        value = max(standings_filtered['date']))

possible_date = max(standings_filtered[standings_filtered["date"] <= pd.to_datetime(date_standings)]["date"].unique())
standings_filtered = standings_filtered[standings_filtered["date"] == possible_date]

st.header(f"Stan tabeli Premier League w sezonie {selected_season} na {date_standings}")

selected_columns_standings = ['team', 'matches_played', 'wins', 'draws', 'defeats', 'goal_difference', 'goals', 'goals_conceded', 'points']
table = standings_filtered[selected_columns_standings]
table['place'] = range(1, len(table) + 1)
table = table.set_index('place')
table.columns = [ 'Zespół', 'Mecze rozegrane', 'Wygrane', 'Remisy', 'Porażki', 'Różnica bramek', 'Bramki strzelone', 'Bramki stracone', 'Punkty']
st.table(table)

filtr1, filtr2, filtr3, filtr4 = st.columns(4)

with filtr1:
    round_filter1 = st.slider("Wybierz kolejkę początkową", min_value=1, max_value=max(df_filtered['round']), on_change=restartStats)
with filtr2:
    round_filter2 = st.slider("Wybierz kolejkę końcową", min_value=1, max_value=max(df_filtered['round']), value=38, on_change=restartStats)
with filtr3:
    team_filter = st.multiselect("Wybierz drużynę", options = sorted(df_filtered['home_team'].unique()), on_change=restartStats)
with filtr4:
    number_of_matches = st.slider("Wybierz liczbę wyświetlanych meczów", min_value=10, max_value=100, step=5, on_change=restartStats)

team_filter2=team_filter


# Filtrowanie danych

if team_filter==[]:
    team_filter2=df['home_team'].unique()
df_filtered=df[(df["round"]>=round_filter1)
               & (df['round']<=round_filter2)
               & (df['season'] == season_filter_matches)
               & ((df['home_team'].isin(team_filter2))
                  | (df['away_team'].isin(team_filter2)))]

# Wypisywanie danych
for i in range(min(number_of_matches, df_filtered['home_team'].count())):
    col1, col2, col3, col4 = st.columns([4,9,2,2])
    with col1:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['date_x']}
                </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['home_team']} {df_filtered.iloc[i]['home_goals_x']} - {df_filtered.iloc[i]['away_goals_x']} {df_filtered.iloc[i]['away_team']}
                </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">Kolejka {df_filtered.iloc[i]['round']} 
                </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="custom-button">', unsafe_allow_html=True)
        st.button(
            "Pokaż statystyki",
            key=f"button_{i}",
            on_click=showStats,
            args=(i,),
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Wyświetlanie dodatkowych informacji pod wierszem, jeśli jest włączone
    if st.session_state.get(f"show_row_{i}", False):
        row = df_filtered.iloc[i]
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9fa; 
                    border-radius: 10px; 
                    padding: 20px;
                    margin: 20px;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
                    <p style='text-align: center; font-size: 20px;'>Statystyki meczu {df_filtered.iloc[i]['home_team']} - {df_filtered.iloc[i]['away_team']}:</p>
                    <p style='text-align: center; font-size: 20px;'>Sędzia: {df_filtered.iloc[i]['referee']}</p>
                </div>""", unsafe_allow_html=True)
        categories = ["Posiadanie piłki", "Strzały", "Strzały na bramkę", "Rzuty wolne", "Rzuty rózne",
            "Spalone", "Obrony bramkarskie", "Faule", "Żółte kartki", "Czerwone kartki", "Podania", "Celne podania"]
        values_home = [df_filtered.iloc[i]["Ball Possession_home"], df_filtered.iloc[i]["Goal Attempts_home"],
            df_filtered.iloc[i]["Shots on Goal_home"], df_filtered.iloc[i]["Free Kicks_home"],
            df_filtered.iloc[i]["Corner Kicks_home"], df_filtered.iloc[i]["Offsides_home"],
            df_filtered.iloc[i]["Goalkeeper Saves_home"], df_filtered.iloc[i]["Fouls_home"],
            df_filtered.iloc[i]["Yellow Cards_home"], df_filtered.iloc[i]["Red Cards_home"],
            df_filtered.iloc[i]["Total Passes_home"], df_filtered.iloc[i]["Completed Passes_home"]]
        values_away = [df_filtered.iloc[i]["Ball Possession_away"], df_filtered.iloc[i]["Goal Attempts_away"],
            df_filtered.iloc[i]["Shots on Goal_away"], df_filtered.iloc[i]["Free Kicks_away"],
            df_filtered.iloc[i]["Corner Kicks_away"], df_filtered.iloc[i]["Offsides_away"],
            df_filtered.iloc[i]["Goalkeeper Saves_away"], df_filtered.iloc[i]["Fouls_away"],
            df_filtered.iloc[i]["Yellow Cards_away"], df_filtered.iloc[i]["Red Cards_away"],
            df_filtered.iloc[i]["Total Passes_away"], df_filtered.iloc[i]["Completed Passes_away"]]

        data = """
            <div style="text-align: center; font-size: 15px;
                background-color: #f8f9fa; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 20px;
                box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
            """
        for j in range(len(categories)):
            data += f"""
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div style="width: 100px;">{values_home[j]}</div>
                <div style="width: 200px;">{categories[j]}</div>
                <div style="width: 100px;">{values_away[j]}</div>
            </div>
            """
        data += "</div>"
        st.markdown(data, unsafe_allow_html=True)
        if st.button(
            "Pokaż więcej statystyk",
            key=f"show_stats_button_{i}",
            args=(i,),
        ):
            st.session_state["stats_id"] = df_filtered.iloc[i]["Unnamed: 0"]
            st.switch_page("pages/Statystyki Premier League.py")
st.write(df_filtered)
st.write(st.session_state)