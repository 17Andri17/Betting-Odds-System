import streamlit as st
import numpy as np
import pandas as pd
import runpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def navbar():
    cols = st.columns(6)
    with cols[0]:
        if st.button(
            "Strona Główna",
            key=f"HomePL"
        ):
            st.switch_page("Kursomat.py")
    with cols[1]:
        if st.button(
            "Premier League",
            key=f"PremierLeaguePL"
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

# Ustawianie wszystkich filtrów na początkowe wartości
def setFilters():
    st.session_state['PLround_filter1'] = 1
    st.session_state['PLround_filter2'] = 38
    st.session_state['PLteam_filter'] = []
    st.session_state['PLnumber_of_matches'] = 10

# Chowanie statystyk po zmianie filtrów
def restartStats():
    for i in range (number_of_matches):
        if f"show_row_{i}" in st.session_state:
            st.session_state[f"show_row_{i}"] = False

# Pokazywanie statystyk dla i-tego meczu
def showStats(i):
    st.session_state[f"show_row_{i}"] = not st.session_state.get(f"show_row_{i}", False)

# chyba do usunięcia
def showDateButton():
    if len(season_filter_matches) == 0:
        st.session_state["show_table"] = True
    else:
        st.session_state["show_table"] = False
    restartStats()


df = st.session_state["dfPL"].copy()
standings = st.session_state["standingsPL"].copy()
df_filtered=df.copy()
standings_filtered=standings.copy()

# Tytuł i tworzenie filtrów
st.title("Premier League")
# Filtry dla tabeli
col1, col2 = st.columns(2)
with col1:
    season_filter = st.multiselect("Wybierz sezon, z którego chcesz zobaczyć tabelę oraz statystyki",
        options = standings['season'].unique(), on_change=showDateButton, max_selections=1)

    if season_filter == []:
        season_filter_matches = sorted(standings['season'].unique(), reverse=True)[0]
    else:
        season_filter_matches = season_filter[0]

    date_standings = max(standings_filtered['date'].dt.strftime('%Y-%m-%d'))

with col2:
    if len(season_filter) == 1:
        standings_filtered = standings[standings['season'] == season_filter_matches]
        date_standings = st.date_input("Wybierz datę tabeli",
            min_value = min(standings_filtered['date']),
            max_value = max(standings_filtered['date']),
            value = max(standings_filtered['date']))

    possible_date = max(standings_filtered[standings_filtered["date"] <= pd.to_datetime(date_standings)]["date"].unique())
    standings_filtered = standings_filtered[standings_filtered["date"] == possible_date]

# Filtrowanie i wyświetlanie tabeli
st.subheader(f"Tabela Premier League w sezonie {season_filter_matches}")
st.caption(f"Stan na: {date_standings}")

selected_columns_standings = ['team', 'matches_played', 'wins', 'draws', 'defeats', 'goal_difference', 'goals', 'goals_conceded', 'points']
table = standings_filtered[selected_columns_standings]
table['place'] = range(1, len(table) + 1)
table = table.set_index('place')
table.columns = [ 'Zespół', 'Mecze rozegrane', 'Wygrane', 'Remisy', 'Porażki', 'Różnica bramek', 'Bramki strzelone', 'Bramki stracone', 'Punkty']
col1, col2, col3 = st.columns([1,5,1])
with col2:
    st.table(table)

# Filtry dla meczów
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
df_filtered.sort_values(by="round", ascending=False, inplace=True)

# Wypisywanie danych
for i in range(min(number_of_matches, df_filtered['home_team'].count())):
    col1, col2, col3, col4, col5, col6 = st.columns([3,5,2,5,2,2])
    with col1:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['date']} {df_filtered.iloc[i]['time']}
                </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['home_team']}
                </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['home_goals']} - {df_filtered.iloc[i]['away_goals']}
                </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['away_team']}
                </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
                <div style="text-align: center; font-size: 15px;
                    background-color: #f8f9ab; 
                    padding: 20px 0;
                    margin: 10px;
                    margin-top: 0;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">Kolejka {df_filtered.iloc[i]['round']} 
                </div>""", unsafe_allow_html=True)
    with col6:
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
            "Spalone", "Faule", "Żółte kartki", "Czerwone kartki", "Podania", "Celne podania"]
        # trzeba będzie dodać Ball Possession jako pierwsze
        values_home = ["54", df_filtered.iloc[i]["home_shots"],
            df_filtered.iloc[i]["home_shots_on_target"], df_filtered.iloc[i]["home_fouled"],
            df_filtered.iloc[i]["home_corner_kicks"], df_filtered.iloc[i]["home_offsides"], df_filtered.iloc[i]["home_fouls"],
            df_filtered.iloc[i]["home_cards_yellow"], df_filtered.iloc[i]["home_cards_red"],
            df_filtered.iloc[i]["home_passes"], df_filtered.iloc[i]["home_passes_completed"]]
        values_away = ["46", df_filtered.iloc[i]["away_shots"],
            df_filtered.iloc[i]["away_shots_on_target"], df_filtered.iloc[i]["away_fouled"],
            df_filtered.iloc[i]["away_corner_kicks"], df_filtered.iloc[i]["away_offsides"], df_filtered.iloc[i]["away_fouls"],
            df_filtered.iloc[i]["away_cards_yellow"], df_filtered.iloc[i]["away_cards_red"],
            df_filtered.iloc[i]["away_passes"], df_filtered.iloc[i]["away_passes_completed"]]

        values_home = [int(v) for v in values_home]
        values_away = [int(v) for v in values_away]

        # data = """
        #     <div style="text-align: center; font-size: 15px;
        #         background-color: #f8f9fa; 
        #         border-radius: 10px; 
        #         padding: 20px; 
        #         margin: 20px;
        #         box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
        #     """
        
        # for j in range(len(categories)):
        #     data += f"""
        #     <div style="display: flex; justify-content: center; margin-bottom: 15px;">
        #         <div style="width: 100px;">{values_home[j]}</div>
        #         <div style="width: 200px;">{categories[j]}</div>
        #         <div style="width: 100px;">{values_away[j]}</div>
        #     </div>
        #     """
        # Funkcja do rysowania pojedynczego wykresu dla każdej statystyki
        col1, col2, col3 = st.columns([2,9,2])
        with col2:
            max_values = np.maximum(values_home, values_away)
            normalized_home = np.array(values_home) / max_values
            normalized_away = np.array(values_away) / max_values

            # Parametry wykresu
            bar_height = 0.8
            bar_spacing = 1.2
            index = np.arange(len(categories) * 2, step=2)

            fig, ax = plt.subplots(figsize=(10, 7))

            # Rysowanie slupkow
            ax.barh(index, normalized_home, height=bar_height, color='crimson', label=df_filtered.iloc[i]['home_team'], align='center')
            ax.barh(index + bar_height, normalized_away, height=bar_height, color='darkblue', label=df_filtered.iloc[i]['away_team'], align='center')

            # Dodanie wartosci tekstowych na wykresie
            for s, (v_home, v_away) in enumerate(zip(values_home, values_away)):
                ax.text(v_home / max_values[s] + 0.02, index[s], str(v_home), va='center', color='black')
                ax.text(v_away / max_values[s] + 0.02, index[s] + bar_height, str(v_away), va='center', color='black')

            # Ustawienia osi
            ax.set_yticks(index + bar_height / 2)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels(categories)
            ax.invert_yaxis()  # Odwrocenie osi Y
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
            ax.xaxis.set_visible(False)  # Usunięcie podziałki osi x

            st.pyplot(plt)
        

        # data += "</div>"
        # st.markdown(data, unsafe_allow_html=True)
        if st.button(
            "Pokaż więcej statystyk",
            key=f"show_stats_button_{i}",
            args=(i,),
        ):
            st.session_state["stats_id"] = df_filtered.iloc[i]
            st.switch_page("pagesHid/Statystyki Premier League.py")

st.write(st.session_state)