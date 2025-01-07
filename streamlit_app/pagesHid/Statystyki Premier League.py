import streamlit as st
import numpy as np
import pandas as pd
import runpy
import random
from mplsoccer import Pitch
import matplotlib.pyplot as plt


def navbar():
    cols = st.columns(6)
    with cols[0]:
        if st.button(
            "Strona Główna",
            key=f"HomeSPL"
        ):
            st.switch_page("Kursomat.py")
    with cols[1]:
        if st.button(
            "Premier League",
            key=f"PremierLeagueSPL"
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


def create_team_structure(idx, formation, starting_eleven):
    
    formation_parts = list(map(int, formation.split('-')))
    used_players = set()
    starting_eleven["main_pos"] = starting_eleven["position_x"].apply(lambda x: x.split(",")[0])
    starting_eleven["number_of_positions"] = starting_eleven["position_x"].apply(lambda x: len(x.split(",")))
    formation_array = [[starting_eleven[starting_eleven["main_pos"] == "GK"].iloc[0]["player"]]]
    
    for parts in formation_parts:
        formation_array.append(parts * [None])

    used_players.add(formation_array[0][0])
    
    # Central Backs
    cbs = starting_eleven[starting_eleven["main_pos"] == "CB"]
    cbs2 = starting_eleven[(starting_eleven["main_pos"] == "CB") & (starting_eleven["number_of_positions"] == 1)]
    cbs3 = starting_eleven[starting_eleven["position_x"].str.contains("CB")]
    if formation_parts[0] == 3:
        if len(cbs) == 3:
            formation_array[1][0] = cbs.iloc[0]["player"]
            formation_array[1][1] = cbs.iloc[1]["player"]
            formation_array[1][2] = cbs.iloc[2]["player"]
        elif len(cbs2) == 3:
            formation_array[1][0] = cbs2.iloc[0]["player"]
            formation_array[1][1] = cbs2.iloc[1]["player"]
            formation_array[1][2] = cbs2.iloc[2]["player"]
        elif len(cbs3) == 3:
            formation_array[1][0] = cbs3.iloc[0]["player"]
            formation_array[1][1] = cbs3.iloc[1]["player"]
            formation_array[1][2] = cbs3.iloc[2]["player"]
        used_players.add(formation_array[1][0])
        used_players.add(formation_array[1][1])
        used_players.add(formation_array[1][2])
    elif formation_parts[0] == 5:
        if len(cbs) == 3:
            formation_array[1][1] = cbs.iloc[0]["player"]
            formation_array[1][2] = cbs.iloc[1]["player"]
            formation_array[1][3] = cbs.iloc[2]["player"]
        elif len(cbs2) == 3:
            formation_array[1][1] = cbs2.iloc[0]["player"]
            formation_array[1][2] = cbs2.iloc[1]["player"]
            formation_array[1][3] = cbs2.iloc[2]["player"]
        elif len(cbs3) == 3:
            formation_array[1][1] = cbs3.iloc[0]["player"]
            formation_array[1][2] = cbs3.iloc[1]["player"]
            formation_array[1][3] = cbs3.iloc[2]["player"]
        used_players.add(formation_array[1][1])
        used_players.add(formation_array[1][2])
        used_players.add(formation_array[1][3])
    else:
        if len(cbs) == 2:
            formation_array[1][1] = cbs.iloc[0]["player"]
            formation_array[1][2] = cbs.iloc[1]["player"]
        elif len(cbs2) == 2:
            formation_array[1][1] = cbs2.iloc[0]["player"]
            formation_array[1][2] = cbs2.iloc[1]["player"]
        elif len(cbs3) == 2:
            formation_array[1][1] = cbs3.iloc[0]["player"]
            formation_array[1][2] = cbs3.iloc[1]["player"]
        used_players.add(formation_array[1][1])
        used_players.add(formation_array[1][2])

    # Left/Right backs for 4 defenders formation
    lbs = starting_eleven[starting_eleven["main_pos"] == "LB"]
    lbs2 = starting_eleven[(starting_eleven["main_pos"] == "LB") & (starting_eleven["number_of_positions"] == 1)]
    lbs3 = starting_eleven[starting_eleven["position_x"].str.contains("LB")]
    rbs = starting_eleven[starting_eleven["main_pos"] == "RB"]
    rbs2 = starting_eleven[(starting_eleven["main_pos"] == "RB") & (starting_eleven["number_of_positions"] == 1)]
    rbs3 = starting_eleven[starting_eleven["position_x"].str.contains("RB")]

    if formation_parts[0] == 4:
        if len(lbs) == 1:
            formation_array[1][0] = lbs.iloc[0]["player"]
        elif len(lbs2) == 1:
            formation_array[1][0] = lbs2.iloc[0]["player"]
        elif len(lbs3) == 1:
            formation_array[1][0] = lbs3.iloc[0]["player"]
        elif len(lbs3) > 1:
            min_place = 100
            name = "none"
            for indx, player in lbs3.iterrows():
                place = player["position_x"].split(",").index("LB")
                if place < min_place:
                    min_place = place
                    name = player["player"]
            formation_array[1][0] = name
        used_players.add(formation_array[1][0])

        if len(rbs) == 1:
            formation_array[1][3] = rbs.iloc[0]["player"]
        elif len(rbs2) == 1:
            formation_array[1][3] = rbs2.iloc[0]["player"]
        elif len(rbs3) == 1:
            formation_array[1][3] = rbs3.iloc[0]["player"]
        elif len(rbs3) > 1:
            min_place = 100
            name = "none"
            for indx, player in rbs3.iterrows():
                place = player["position_x"].split(",").index("RB")
                if place < min_place:
                    min_place = place
                    name = player["player"]
            formation_array[1][3] = name
        used_players.add(formation_array[1][3])

    elif formation_parts[0] == 5:
        if len(lbs) == 1:
            formation_array[1][0] = lbs.iloc[0]["player"]
        elif len(lbs2) == 1:
            formation_array[1][0] = lbs2.iloc[0]["player"]
        elif len(lbs3) == 1:
            formation_array[1][0] = lbs3.iloc[0]["player"]
        elif len(lbs3) > 1:
            min_place = 100
            name = "none"
            for indx, player in lbs3.iterrows():
                place = player["position_x"].split(",").index("LB")
                if place < min_place:
                    min_place = place
                    name = player["player"]
            formation_array[1][0] = name
        used_players.add(formation_array[1][0])

        if len(rbs) == 1:
            formation_array[1][4] = rbs.iloc[0]["player"]
        elif len(rbs2) == 1:
            formation_array[1][4] = rbs2.iloc[0]["player"]
        elif len(rbs3) == 1:
            formation_array[1][4] = rbs3.iloc[0]["player"]
        elif len(rbs3) > 1:
            min_place = 100
            name = "none"
            for indx, player in rbs3.iterrows():
                place = player["position_x"].split(",").index("RB")
                if place < min_place:
                    min_place = place
                    name = player["player"]
            formation_array[1][4] = name
        used_players.add(formation_array[1][4])

    # Wingers in formation with 3 defenders

    # Forwards
    fws = starting_eleven[starting_eleven["main_pos"] == "FW"]
    fws2 = starting_eleven[(starting_eleven["main_pos"] == "FW") & (starting_eleven["number_of_positions"] == 1)]
    fws3 = starting_eleven[starting_eleven["position_x"].str.contains("FW")]

    if formation_parts[-1] == 2:
        if len(fws) == 2:
            formation_array[-1][0] = fws.iloc[0]["player"]
            formation_array[-1][1] = fws.iloc[1]["player"]
        elif len(fws2) == 2:
            formation_array[-1][0] = fws2.iloc[0]["player"]
            formation_array[-1][1] = fws2.iloc[1]["player"]
        elif len(fws3) == 2:
            formation_array[-1][0] = fws3.iloc[0]["player"]
            formation_array[-1][1] = fws3.iloc[1]["player"]
        elif len(fws3) > 2:
            min_place1 = 100
            min_place2 = 100
            name1 = "none"
            name2 = "none"
            for indx, player in fws3.iterrows():
                place = player["position_x"].split(",").index("FW")
                if place < min_place1:
                    min_place2 = min_place1
                    min_place1 = place
                    name1 = player["player"]
                    name2 = name1
                elif place < min_place2:
                    min_place2 = place
                    name2 = player["player"]
            formation_array[-1][0] = name1
            formation_array[-1][1] = name2
        used_players.add(formation_array[-1][0])
        used_players.add(formation_array[-1][1])

    elif formation_parts[-1] == 1:
        if len(fws) == 1:
            formation_array[-1][0] = fws.iloc[0]["player"]
        elif len(fws2) == 1:
            formation_array[-1][0] = fws2.iloc[0]["player"]
        elif len(fws3) == 1:
            formation_array[-1][0] = fws3.iloc[0]["player"]
        elif len(fws3) > 1:
            min_place = 100
            name = "none"
            for indx, player in fws3.iterrows():
                place = player["position_x"].split(",").index("FW")
                if place < min_place:
                    min_place = place
                    name = player["player"]
            formation_array[-1][0] = name
        used_players.add(formation_array[-1][0])

    elif formation_parts[-1] == 3:
        if len(fws) == 1:
            formation_array[-1][1] = fws.iloc[0]["player"]
        elif len(fws2) == 1:
            formation_array[-1][1] = fws2.iloc[0]["player"]
        elif len(fws3) == 1:
            formation_array[-1][1] = fws3.iloc[0]["player"]
        elif len(fws3) > 1:
            min_place = 100
            name = "none"
            for indx, player in fws3.iterrows():
                place = player["position_x"].split(",").index("FW")
                if place < min_place:
                    min_place = place
                    name = player["player"]
            formation_array[-1][1] = name
        used_players.add(formation_array[-1][1])


    # add the rest randomly
    all_players = starting_eleven['player'].tolist()
    unassigned_players = list(set(all_players) - used_players)
    for arr in formation_array:
        for i, el in enumerate(arr):
            if el is None and unassigned_players:
                arr[i] = random.choice(unassigned_players)
                unassigned_players.remove(arr[i])


    return formation_array

def get_starters(group):
    starters = []
    group = group.sort_index()
    used_indices = set()
    for idx, row in group.iterrows():
        if idx in used_indices:
            continue

        if row['minutes'] == 90:
            starters.append(group.index.get_loc(idx))
            used_indices.add(idx)
        elif row['minutes'] < 90:
            starters.append(group.index.get_loc(idx))
            used_indices.add(idx)
            minutes_sum = row['minutes']
            next_row = row
            next_idx_global = idx
            while minutes_sum < 90 and next_row['cards_red'] < 1:
                next_idx = group.index.get_loc(next_idx_global) + 1
                next_idx_global = next_idx_global + 1
                if next_idx < len(group):
                    next_row = group.iloc[next_idx]
                    minutes_sum += next_row['minutes']
                    if minutes_sum > 91:
                        starters.append(next_idx)
                    used_indices.add(next_idx_global)
                else:
                    minutes_sum = 90
                    
    group = group.iloc[starters]
    return group


players_23_24 = pd.read_csv("../fbref/data/players_pl_23-24_fbref.csv")
players_22_23 = pd.read_csv("../fbref/data/players_pl_22-23_fbref.csv")
players_21_22 = pd.read_csv("../fbref/data/players_pl_21-22_fbref.csv")
players_20_21 = pd.read_csv("../fbref/data/players_pl_20-21_fbref.csv")
players_19_20 = pd.read_csv("../fbref/data/players_pl_19-20_fbref.csv")
players_18_19 = pd.read_csv("../fbref/data/players_pl_18-19_fbref.csv")
players = pd.concat([players_23_24, players_22_23, players_21_22, players_20_21, players_19_20, players_18_19], ignore_index=True)
players = players.rename(columns={"position": "position_x"})

matches = pd.read_csv("../fbref/data/matches_with_rolling_stats_pl.csv")
matches["formation_home"] = matches["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
matches["formation_away"] = matches["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
matches["formation_home"] = matches["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
matches["formation_away"] = matches["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)


odds_23_24 = pd.read_csv("../fbref/data/E0_23-24.csv")
odds_22_23 = pd.read_csv("../fbref/data/E0_22-23.csv")
odds_21_22 = pd.read_csv("../fbref/data/E0_21-22.csv")
odds_20_21 = pd.read_csv("../fbref/data/E0_20-21.csv")
odds_19_20 = pd.read_csv("../fbref/data/E0_19-20.csv")
odds_18_19 = pd.read_csv("../fbref/data/E0_18-19.csv")
odds = pd.concat([odds_23_24, odds_22_23, odds_21_22, odds_20_21, odds_19_20, odds_18_19], ignore_index=True)

team_name_mapping = {
    'Burnley': 'Burnley',
    'Arsenal': 'Arsenal',
    'Bournemouth': 'Bournemouth',
    'Brighton': 'Brighton & Hove Albion',
    'Everton': 'Everton',
    'Sheffield United': 'Sheffield United',
    'Newcastle': 'Newcastle United',
    'Brentford': 'Brentford',
    'Chelsea': 'Chelsea',
    'Man United': 'Manchester United',
    "Nott'm Forest": 'Nottingham Forest',
    'Fulham': 'Fulham',
    'Liverpool': 'Liverpool',
    'Wolves': 'Wolverhampton Wanderers',
    'Tottenham': 'Tottenham Hotspur',
    'Man City': 'Manchester City',
    'Aston Villa': 'Aston Villa',
    'West Ham': 'West Ham United',
    'Crystal Palace': 'Crystal Palace',
    'Luton': 'Luton Town',
    'Leeds': 'Leeds United',
    'Leicester': 'Leicester City',
    'Southampton': 'Southampton',
    'Watford': 'Watford',
    'Norwich': 'Norwich City',
    'West Brom': 'West Bromwich Albion',
    'Huddersfield': 'Huddersfield Town',
    'Cardiff': 'Cardiff City'
}

odds['HomeTeam'] = odds['HomeTeam'].map(team_name_mapping)
odds['AwayTeam'] = odds['AwayTeam'].map(team_name_mapping)

odds['Date'] = pd.to_datetime(odds['Date'], format='%d/%m/%Y')
matches['date'] = pd.to_datetime(matches['date'])
odds = odds[["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]]
odds.rename(columns={
    'HomeTeam': 'home_team',
    'AwayTeam': 'away_team'
}, inplace=True)

merged_df = pd.merge(
    matches,
    odds,
    how='inner',
    left_on=['home_team', 'away_team', 'date'],
    right_on=['home_team', 'away_team', 'Date']
)
merged_df.drop(columns=['Date'], inplace=True)
merged_df["B365probsH"] = 1 / merged_df["B365H"] / (1 / merged_df["B365H"] + 1 / merged_df["B365D"] + 1 / merged_df["B365A"])
merged_df["B365probsD"] = 1 / merged_df["B365D"] / (1 / merged_df["B365H"] + 1 / merged_df["B365D"] + 1 / merged_df["B365A"])
merged_df["B365probsA"] = 1 / merged_df["B365A"] / (1 / merged_df["B365H"] + 1 / merged_df["B365D"] + 1 / merged_df["B365A"])


date = st.session_state['stats_id']["date"]
home_team = st.session_state['stats_id']["home_team"]
away_team = st.session_state['stats_id']["away_team"]

home_goals = st.session_state['stats_id']["home_goals"]
away_goals = st.session_state['stats_id']["away_goals"]

formation_home = st.session_state['stats_id']["formation_home"]
formation_away = st.session_state['stats_id']["formation_away"]

home_team_players = players[(players["date"]==date) & (players["team"]==home_team)]
away_team_players = players[(players["date"]==date) & (players["team"]==away_team)]

structure_away = create_team_structure(0, formation_away, get_starters(players[(players["team"] == away_team) & (players["date"] == date)]))
structure_home = create_team_structure(0, formation_home, get_starters(players[(players["team"] == home_team) & (players["date"] == date)]))


width = 100
pitch = Pitch(pitch_color='grass', line_color='white', stripe=False)
fig, ax = pitch.draw(figsize=(16, 8))

width = 120
height = 80
size = 36*36
y_shift_length = 15

color1 = '#2b83ba'
color2 = '#d7191c'

formation_3_x = [6, 24, 37, 50]
formation_4_x = [6, 21, 32, 43, 54]

formation_3_y = [0, 18, 18, 12]
formation_4_y = [0, 18, 18, 18, 12]


if len(structure_home) == 4:
    formation_x = formation_3_x
    formation_y = formation_3_y
else:
    formation_x = formation_4_x
    formation_y = formation_4_y
for i in range(len(structure_home)):
    arr = structure_home[i]
    y_shift_length = formation_y[i]
    y_start = 40 - (len(arr) - 1) * y_shift_length / 2
    for j in range(len(arr)):
        ax.scatter(formation_x[i], y_start + y_shift_length*j, c=color1, s=size, edgecolors='black', label='Team A')
        ax.text(formation_x[i], y_start + y_shift_length*j + 5.5, arr[j].split()[-1], horizontalalignment='center', fontsize = 13)

if len(structure_away) == 4:
    formation_x = formation_3_x
    formation_y = formation_3_y
else:
    formation_x = formation_4_x
    formation_y = formation_4_y
for i in range(len(structure_away)):
    arr = structure_away[i]
    y_shift_length = formation_y[i]
    if i == len(structure_away) - 1 and len(arr) > 2:
        y_shift_length = 28
    y_start = 40 - (len(arr) - 1) * y_shift_length / 2
    for j in range(len(arr)):
        ax.scatter(width - formation_x[i], y_start + y_shift_length*j, c=color2, s=size, edgecolors='black', label='Team A')
        ax.text(width - formation_x[i], y_start + y_shift_length*j + 5.5, arr[j].split()[-1], horizontalalignment='center', fontsize = 13)

plt.show()

st.markdown(f"""
                <p style='text-align: center; font-size: 40px;'>{home_team}  {home_goals} - {away_goals}  {away_team}</p>
                """, unsafe_allow_html=True)


categories = ['Home Win', 'Draw', 'Away Win']
probabilities = [merged_df[(merged_df["date"]==date) & (merged_df["home_team"] == home_team)]["B365probsH"], merged_df[(merged_df["date"]==date) & (merged_df["home_team"] == home_team)]["B365probsD"], merged_df[(merged_df["date"]==date) & (merged_df["home_team"] == home_team)]["B365probsA"]]
colors = ['green', 'gray', 'blue']

fig2, ax = plt.subplots(figsize=(6, 1))
start = 0

for prob, color in zip(probabilities, colors):
    ax.barh(0, prob, left=start, color=color, edgecolor='none', height=0.5)
    start += prob

start = 0
for prob, color in zip(probabilities, colors):
    ax.text(start + prob / 2, 0, f"{int(prob * 100)}%", color='black', va='center', ha='center', fontsize=10)
    start += prob

ax.set_xlim(0, 1)
ax.axis('off')  # Turn off the axis
plt.title('Prawdopodbieństwo zdarzeń', pad=10)
plt.show()
plt.tight_layout()

col1, col2, col3, col4 = st.columns([1,4,4,1])

st.write("<div style='display: block; justify-content: center; width:60%'>", unsafe_allow_html=True)
with col2:
    st.pyplot(fig2)
    st.write("</div>", unsafe_allow_html=True)

    st.write(fig)

with col3:
    filtr1, filtr2 = st.columns(2)

    css = """
    .st-key-team_filter *, .st-key-stat_filter *, .st-key-team_filter, .st-key-stat_filter{
                cursor: pointer;
            }
    """

    st.html(f"<style>{css}</style>")

    with filtr1:
        team_filter = st.selectbox("Wybierz drużynę", options=[home_team, away_team], key="team_filter")
    with filtr2:
        stat_filter = st.selectbox("Wybierz statystykę", options=["goals", "corner_kicks"], key="stat_filter")

    def select_last_matches(df, team, date, n, where="all"):
        if where == "home":
            team_matches = df[(df["home_team"] == team) & (df["date"]<date)]
        elif where == "away":
            team_matches = df[(df["away_team"] == team) & (df["date"]<date)]
        else:
            team_matches = df[((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"]<date)]
        team_matches_sorted = team_matches.sort_values(by="date", ascending=False)
        return team_matches_sorted.head(n)

    def get_stat(df, team, stat):
        df[stat] = df.apply(lambda x: x["home_" + stat] if x["home_team"] == team else x["away_" + stat], axis=1)
        df["new_date"] = df["date"].apply(lambda x: str(x)[5:7]+"."+str(x)[8:10])
        return df[[stat, "new_date"]]

    team = team_filter
    stat = stat_filter
    date = pd.to_datetime("2024-05-11")
    n = 10
    last_matches = select_last_matches(matches, team, date, n)
    stat_df = get_stat(last_matches, team, stat)
    threshold = 1.5

    # Set colors: green if above threshold, red otherwise
    colors = ["green" if val > threshold else "red" for val in stat_df[stat]]

    # Plot the bar chart
    fig3 = plt.figure(figsize=(10, 6))
    bars = plt.bar(stat_df["new_date"], stat_df[stat], color=colors)

    # Add threshold line
    plt.axhline(y=threshold, color="gray", linestyle="--", label=f"Linia = {threshold}")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                ha="center", va="bottom", fontsize=20)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height-0.4, str(height),
                ha="center", va="bottom", fontsize=20)

    # Chart styling
    plt.title(stat.capitalize() + " dla " + team + " w ostatnich " + str(n) + " meczach")
    plt.xlabel("Mecze")
    plt.ylabel(stat)
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()
    st.write(fig3)
