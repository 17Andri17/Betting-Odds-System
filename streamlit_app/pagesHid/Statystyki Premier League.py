import streamlit as st
import numpy as np
import pandas as pd
import runpy
import random
from mplsoccer import Pitch
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import joblib
from models import FootballMatchPredictor, FootballMatchPredictorOutcome

import math
from scipy.optimize import minimize_scalar
from IPython.core.display import HTML

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def squads(players, date, home_team, away_team, formation_home, formation_away):
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
    formation_4_y = [0, 16, 18, 18, 12]


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
    st.write(fig)

@st.cache_data
def statsGraph(home_stats, away_stats, categories):
    total_stats = np.array(home_stats) + np.array(away_stats)
    home_ratios = np.array(home_stats) / total_stats
    away_ratios = np.array(away_stats) / total_stats

    fig, ax = plt.subplots(figsize=(8, len(categories) * 0.62))
    ax.set_facecolor("#1A1A1A") 

    for j, (category, home_ratio, away_ratio) in enumerate(zip(categories, home_ratios, away_ratios)):
        y_position = len(categories) - j  # Pozycja w osi Y (odwracamy kolejność)
        home_color = "#001e28"
        away_color = "#001e28"
        if home_ratio > away_ratio:
            home_color = "#ff0046"
        elif home_ratio < away_ratio:
            away_color = "#ff0046"

        ax.text(0, y_position + 0.4, category, ha="center", va="center", fontsize=12, color="#001e28", weight="bold")

        ax.barh(
            y_position, 0.993, left=0.007, height=0.18, color="#eee", align="center", 
            zorder=3, linewidth=1.5
        )

        ax.barh(
            y_position, -0.993, left=-0.007, height=0.18, color="#eee", align="center", 
            zorder=3, linewidth=1.5
        )
                
        ax.barh(
            y_position, min(-home_ratio+0.007, 0), left=-0.007, height=0.18, color=home_color, align="center", 
            zorder=3, linewidth=1.5
        )
        ax.barh(
            y_position, max(away_ratio-0.007, 0), height=0.18, left=0.007, color=away_color, align="center", 
            zorder=3, linewidth=1.5
        )
        if j==0:
            ax.text(-0.999, y_position + 0.4, f"{home_stats[j]}%", ha="left", va="center", fontsize=11, color="#001e28", weight="bold")
            ax.text(0.999, y_position + 0.4, f"{away_stats[j]}%", ha="right", va="center", fontsize=11, color="#001e28", weight="bold")
        else:
            ax.text(-0.999, y_position + 0.4, f"{home_stats[j]}", ha="left", va="center", fontsize=11, color="#001e28", weight="bold")
            ax.text(0.999, y_position + 0.4, f"{away_stats[j]}", ha="right", va="center", fontsize=11, color="#001e28", weight="bold")

    ax.set_xlim(-1, 1)  # Oś X od -1 do 1 (po równo na obie strony)
    ax.set_ylim(0.5, len(categories) + 0.5)  # Oś Y dla odpowiedniego rozmieszczenia
    ax.axis("off")  # Usunięcie osi, ponieważ nie są potrzebne

    plt.tight_layout()
    st.pyplot(plt)

def poisson_cdf(lmbda, k):
    return sum((lmbda ** i) * math.exp(-lmbda) / math.factorial(i) for i in range(k + 1))

def solve_lambda(p_over, line):
    def objective(lmbda):
        p_leq = poisson_cdf(lmbda, line)
        return abs(1 - p_leq - p_over)

    result = minimize_scalar(objective, bounds=(0, 10), method='bounded')
    return result.x

def poisson_probability(lmbda, line, over=True):
    if over:
        return 1 - poisson_cdf(lmbda, math.floor(line))
    else:
        return poisson_cdf(lmbda, math.floor(line))

def exact_score_probability(home_lambda, away_lambda, home_goals, away_goals):
    p_home = (home_lambda ** home_goals) * math.exp(-home_lambda) / math.factorial(home_goals)
    p_away = (away_lambda ** away_goals) * math.exp(-away_lambda) / math.factorial(away_goals)
    return p_home * p_away

def get_probabilities(all_fetures):
    all_features_scaled = scaler.transform([all_features])
    input_features = all_features_scaled[:, [filtered_matches.columns.get_loc(col) for col in selected_features]]

    all_features_scaled_home = scaler_home.transform([all_features])
    input_features_home = all_features_scaled_home[:, [filtered_matches.columns.get_loc(col) for col in selected_features_home]]

    all_features_scaled_away = scaler_home.transform([all_features])
    input_features_away = all_features_scaled_away[:, [filtered_matches.columns.get_loc(col) for col in selected_features_away]]

    all_features_scaled_outcome = scaler_outcome.transform([all_features])
    input_features_outcome = all_features_scaled_outcome[:, [filtered_matches.columns.get_loc(col) for col in selected_features_outcome]]

    probabilities = {}

    probabilities["under25"], probabilities["over25"] = predict_goals(input_features, model)
    probabilities["home_under15"], probabilities["home_over15"] = predict_goals(input_features_home, model_home)
    probabilities["away_under15"], probabilities["away_over15"] = predict_goals(input_features_away, model_away)
    probabilities["draw"], probabilities["home_win"], probabilities["away_win"] = predict_outcome(input_features_outcome, model_outcome)
    probabilities["lambda_goals"] = solve_lambda(probabilities["over25"], 2)
    probabilities["lambda_home_goals"] = solve_lambda(probabilities["home_over15"], 1)
    probabilities["lambda_away_goals"] = solve_lambda(probabilities["away_over15"], 1)

    probabilities["1x"] = probabilities["draw"] + probabilities["home_win"]
    probabilities["x2"] = probabilities["draw"] + probabilities["away_win"]
    probabilities["12"] = probabilities["home_win"] + probabilities["away_win"]

    probabilities["no_draw_home_win"] = probabilities["home_win"] / (probabilities["home_win"] + probabilities["away_win"])
    probabilities["no_draw_away_win"] = probabilities["away_win"] / (probabilities["home_win"] + probabilities["away_win"])

    probabilities["btts"] = poisson_probability(probabilities["lambda_home_goals"], 0.5, over=True)*poisson_probability(probabilities["lambda_away_goals"], 0.5, over=True)
    probabilities["no_btts"] = 1 - poisson_probability(probabilities["lambda_home_goals"], 0.5, over=True)*poisson_probability(probabilities["lambda_away_goals"], 0.5, over=True)

    probabilities["under05"] = poisson_probability(probabilities["lambda_goals"], 0.5, over=False)
    probabilities["over05"] = poisson_probability(probabilities["lambda_goals"], 0.5, over=True)
    probabilities["under15"] = poisson_probability(probabilities["lambda_goals"], 1.5, over=False)
    probabilities["over15"] = poisson_probability(probabilities["lambda_goals"], 1.5, over=True)
    probabilities["under35"] = poisson_probability(probabilities["lambda_goals"], 3.5, over=False)
    probabilities["over35"] = poisson_probability(probabilities["lambda_goals"], 3.5, over=True)

    probabilities["home_under05"] = poisson_probability(probabilities["lambda_home_goals"], 0.5, over=False)
    probabilities["home_over05"] = poisson_probability(probabilities["lambda_home_goals"], 0.5, over=True)
    probabilities["away_under05"] = poisson_probability(probabilities["lambda_away_goals"], 0.5, over=False)
    probabilities["away_over05"] = poisson_probability(probabilities["lambda_away_goals"], 0.5, over=True)
    probabilities["home_under25"] = poisson_probability(probabilities["lambda_home_goals"], 2.5, over=False)
    probabilities["home_over25"] = poisson_probability(probabilities["lambda_home_goals"], 2.5, over=True)
    probabilities["away_under25"] = poisson_probability(probabilities["lambda_away_goals"], 2.5, over=False)
    probabilities["away_over25"] = poisson_probability(probabilities["lambda_away_goals"], 2.5, over=True)

    probabilities["exact_11"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 1, 1)
    probabilities["exact_00"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 0, 0)
    probabilities["exact_22"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 2, 2)
    probabilities["exact_10"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 1, 0)
    probabilities["exact_20"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 2, 0)
    probabilities["exact_21"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 2, 1)
    probabilities["exact_01"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 0, 1)
    probabilities["exact_02"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 0, 2)
    probabilities["exact_12"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 1, 2)

    probabilities["exact_33"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 3, 3)
    probabilities["exact_30"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 3, 0)
    probabilities["exact_31"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 3, 1)
    probabilities["exact_32"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 3, 2)
    probabilities["exact_03"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 0, 3)
    probabilities["exact_13"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 1, 3)
    probabilities["exact_23"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 2, 3)
    probabilities["exact_40"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 4, 0)
    probabilities["exact_41"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 4, 1)
    probabilities["exact_42"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 4, 2)
    probabilities["exact_43"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 4, 3)
    probabilities["exact_44"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 4, 4)
    probabilities["exact_34"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 3, 4)
    probabilities["exact_24"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 2, 4)
    probabilities["exact_14"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 1, 4)
    probabilities["exact_04"] = exact_score_probability(probabilities["lambda_home_goals"], probabilities["lambda_away_goals"], 0, 4)
    
    return probabilities

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler

def load_selected_fetures(selected_features_path):
    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    return selected_features

def predict_goals(input_features, model):
    with torch.no_grad():
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        prediction = model(input_tensor)
        return prediction.squeeze()[0].item(), prediction.squeeze()[1].item()
    
def predict_outcome(input_features, model):
    with torch.no_grad():
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        prediction = model(input_tensor)
        return prediction.squeeze()[0].item(), prediction.squeeze()[1].item(), prediction.squeeze()[2].item()

def load_data():
    home_team = st.query_params["home_team"]
    date = pd.to_datetime(st.query_params["date"])
    league = st.query_params["league"]
    odds = pd.read_csv("../odds.csv")
    standings = pd.read_csv("../standings_with_new.csv")
    standings['date']=pd.to_datetime(standings['date'])
    standings['goal_difference'] = standings['goal_difference'].astype(int)
    standings['goals'] = standings['goals'].astype(int)
    standings['goals_conceded'] = standings['goals_conceded'].astype(int)
    standings = standings[standings["league"] == league]
    current_standings_date = standings[standings['date'] < date]["date"].tail(1).iloc[0]
    standings = standings[standings["date"] == current_standings_date]
    players = pd.read_csv("../players_pl.csv")
    players_new = pd.read_csv("../new_players.csv")
    players = pd.concat([players, players_new], ignore_index=True)
    players["date"] = pd.to_datetime(players["date"])
    matches = pd.read_csv("../final_prepared_data_with_new.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    players = players.rename(columns={"position": "position_x"})
    return players, matches, odds, home_team, date, standings

def getCourse(prob):
    return round(1 / prob, 2)

def select_last_matches(df, team, date, n, where="all"):
        if where == "home":
            team_matches = df[(df["home_team"] == team) & (df["date"]<date)]
        elif where == "away":
            team_matches = df[(df["away_team"] == team) & (df["date"]<date)]
        else:
            team_matches = df[((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"]<date)]
        team_matches_sorted = team_matches.sort_values(by="date", ascending=False)
        return team_matches_sorted.head(n)

def get_stat(df, team, stat, other_team = False, sum = False):
        df[stat] = df.apply(lambda x: x["home_" + stat] if x["home_team"] == team else x["away_" + stat], axis=1)
        if other_team:
            df[stat] = df.apply(lambda x: x["away_" + stat] if x["home_team"] == team else x["home_" + stat], axis=1)
        if sum:
            df[stat] = df.apply(lambda x: x["home_" + stat] + x["away_" + stat], axis=1)
        df["new_date"] = df["date"].apply(lambda x: str(x)[5:7]+"."+str(x)[8:10])
        return df[[stat, "new_date"]]

def generate_html_table(teams_stats):
    html_template = """
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
                text-align: center;
                background-color: #f9f9f9;
                color:black;
                border: 0px solid rgba(34, 34, 38, 0.25);
                font: Arial;
            th, td {{
                padding: 5px;
                border: 0px solid rgba(34, 34, 38, 0.25);
                text-align: center;
                width: 1%;
            }}
            th {{
                background-color: #f9f9f9;
                color: rgba(34, 34, 38, 0.45);
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #e6e6e6;
            }}
            td {{
                line-height: 25px;
                padding-top: 2px;
                padding-bottom: 2px;
            }}
            th:last-child, td:last-child {{
                font-weight: bold;
            }}
            th:nth-child(2), td:nth-child(2) {{
                width: 15%;
                text-align: left;
            }}
            th:nth-child(1), td:nth-child(1) {{
                width: 0%;
            }}
            .highlight-green td:nth-child(1) span {{
                display: inline-block;
                width: 25px;
                height: 25px;
                line-height: 25px;
                border-radius: 50%;
                background-color: #26943b;
                color: white;
                font-weight: bold;
            }}
            .highlight-red td:nth-child(1) span {{
                display: inline-block;
                width: 25px;
                height: 25px;
                line-height: 25px;
                border-radius: 50%;
                background-color: #c1262d;
                color: white;
                font-weight: bold;
            }}
        </style>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Zespół</th>
                    <th>M</th>
                    <th>W</th>
                    <th>R</th>
                    <th>P</th>
                    <th>+/-</th>
                    <th>+</th>
                    <th>-</th>
                    <th>Pkt</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """

    rows = ""
    for team in teams_stats:
        row_class = (
            "highlight-green" if team["highlight"] == "green" else "highlight-red" if team["highlight"] == "red" else ""
        )
        rows += f"""
        <tr class="{row_class}">
            <td><span>{team["position"]}</span></td>
            <td>{team["name"]}</td>
            <td>{team["played"]}</td>
            <td>{team["wins"]}</td>
            <td>{team["draws"]}</td>
            <td>{team["losses"]}</td>
            <td>{team["diff"]}</td>
            <td>{team["goals_scored"]}</td>
            <td>{team["goals_conceded"]}</td>
            <td>{team["points"]}</td>
        </tr>
        """
    return html_template.format(rows=rows)


players, matches, odds, home_team, date, standings = load_data()
curr_match = matches[(matches["date"] == date) & (matches["home_team"] == home_team)].iloc[0]
matches2 = matches.copy()

# Load models
scaler = load_scaler("../models/goals_scaler_v1.pkl")
selected_features = load_selected_fetures("../models/goals_features_v1.json")
model = load_model("../models/goals_predictor_v1.pth")

scaler_home = load_scaler("../models/goals_scaler_home_goals_v1.pkl")
selected_features_home = load_selected_fetures("../models/home_goals_features_v1.json")
model_home = load_model("../models/goals_home_predictor_v1.pth")

scaler_away = load_scaler("../models/goals_scaler_away_goals_v1.pkl")
selected_features_away = load_selected_fetures("../models/away_goals_features_v1.json")
model_away = load_model("../models/goals_away_predictor_v1.pth")

scaler_outcome = load_scaler("../models/outcome_scaler.pkl")
selected_features_outcome = load_selected_fetures("../models/outcome_features.json")
model_outcome = load_model("../models/football_match_predictor_v1.pth")

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

odds['home_team'] = odds['home_team'].map(team_name_mapping)
odds['away_team'] = odds['away_team'].map(team_name_mapping)

odds['Date'] = pd.to_datetime(odds['Date'], format='%d/%m/%Y')
matches['date'] = pd.to_datetime(matches['date'])
odds = odds[["Date", "home_team", "away_team", "B365H", "B365D", "B365A"]]

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


away_team = curr_match["away_team"]
home_goals = curr_match["home_goals"]
away_goals = curr_match["away_goals"]
formation_home = curr_match["formation_home"]
formation_away = curr_match["formation_away"]

filtered_matches = matches[(matches["date"] == date) & (matches["home_team"] == home_team)]

filtered_matches = filtered_matches[[col for col in matches.columns if 'last5' in col or 'matches_since' in col or 'overall' in col or 'tiredness' in col]]
filtered_matches = filtered_matches.drop(columns = ["home_last5_possession", "away_last5_possession"])
filtered_matches = filtered_matches[~filtered_matches.isna().any(axis=1)]
all_features = filtered_matches.iloc[0]

match_probabilities = get_probabilities(all_features)


st.markdown(f"""
                <p style='text-align: center; font-size: 40px;'>{home_team}  {int(home_goals)} - {int(away_goals)}  {away_team}</p>
                """, unsafe_allow_html=True)



categories = ['Home Win', 'Draw', 'Away Win']
probabilities2 = [round(match_probabilities["home_win"], 2), round(match_probabilities["draw"], 2), 1 - round(match_probabilities["home_win"], 2) - round(match_probabilities["draw"], 2)]

if len(merged_df[(merged_df["date"] == date) & (merged_df["home_team"] == home_team)]) > 0:
    probabilities = [merged_df[(merged_df["date"]==date) & (merged_df["home_team"] == home_team)]["B365probsH"].iloc[0], merged_df[(merged_df["date"]==date) & (merged_df["home_team"] == home_team)]["B365probsD"].iloc[0], merged_df[(merged_df["date"]==date) & (merged_df["home_team"] == home_team)]["B365probsA"].iloc[0]]
    probabilities = [round(probabilities[0], 2), round(probabilities[1], 2), 1 - round(probabilities[0], 2) - round(probabilities[1], 2)]
else:
    probabilities = []
colors = ['green', 'gray', 'blue']

fig21, ax = plt.subplots(figsize=(6, 1))
start = 0

for prob, color in zip(probabilities2, colors):
    ax.barh(0, prob, left=start, color=color, edgecolor='none', height=0.5)
    start += prob

start = 0
sumi = 0
for prob, color in zip(probabilities2, colors):
    ax.text(start + prob / 2, 0, f"{int(prob * 100)}%", color='black', va='center', ha='center', fontsize=10)
    start += prob

ax.set_xlim(0, 1)
ax.axis('off')  # Turn off the axis
plt.title('Prawdopodbieństwo zdarzeń modelu', pad=10)
plt.show()
plt.tight_layout()

if len(probabilities) > 0:
    fig22, ax = plt.subplots(figsize=(6, 1))
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
    plt.title('Prawdopodbieństwo zdarzeń bukmacherów', pad=10)
    plt.show()
    plt.tight_layout()

categories = ["Posiadanie piłki", "Strzały", "Strzały na bramkę", "Rzuty wolne", "Rzuty rożne",
                "Spalone", "Faule", "Żółte kartki", "Czerwone kartki", "Podania", "Celne podania"]

home_stats = [int(100*curr_match["home_possession"]), curr_match["home_shots"],
                curr_match["home_shots_on_target"], curr_match["home_fouled"],
                curr_match["home_corner_kicks"], curr_match["home_offsides"], curr_match["home_fouls"],
                curr_match["home_cards_yellow"], curr_match["home_cards_red"],
                curr_match["home_passes"], curr_match["home_passes_completed"]]
away_stats = [int(100*curr_match["away_possession"]), curr_match["away_shots"],
                curr_match["away_shots_on_target"], curr_match["away_fouled"],
                curr_match["away_corner_kicks"], curr_match["away_offsides"], curr_match["away_fouls"],
                curr_match["away_cards_yellow"], curr_match["away_cards_red"],
                curr_match["away_passes"], curr_match["away_passes_completed"]]
home_stats = [int(v) for v in home_stats]
away_stats = [int(v) for v in away_stats]


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Informacje", "Statystyki", "Składy", "Kursy bukmacherskie", "Inne"])

with tab2:
    col1, col2, col3 = st.columns([4, 11, 4])
    with col2:
        statsGraph(home_stats, away_stats, categories)

with tab3:
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        with st.spinner("Generowanie składów"):
            squads(players, date, home_team, away_team, formation_home, formation_away)



home_matches = matches[(matches["date"] < date) &
    ((matches["home_team"] == home_team) | (matches["away_team"] == home_team)) ]
away_matches = matches[(matches["date"] < date) &
    ((matches["home_team"] == away_team) | (matches["away_team"] == away_team)) ]
home_matches = home_matches[["date", "time", "home_team", "home_goals", "away_goals", "away_team"]]
home_matches = home_matches.sort_values(by="date", ascending=False)
away_matches = away_matches[["date", "time", "home_team", "home_goals", "away_goals", "away_team"]]
away_matches = away_matches.sort_values(by="date", ascending=False)

last_home = f"""
    <div style="text-align: center; font-size: 15px;
        background-color: #f8f9fa; 
        border-radius: 10px; 
        padding: 5px; 
        margin: 5px;
        box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
    <div style="display: flex; justify-content: center; margin-bottom: 5px; font-size: 25px; font-weight:bold;">Ostatnie mecze {home_team}</div>
    """

if len(home_matches) == 0:
    last_home += "<div style='font-size: 20px;'>Brak danych na temat ostatnich meczów</div>"

for j in range(min(5, len(home_matches))):
    last_home += f"""
    <div style="display: flex; justify-content: center; margin-bottom: 5px;">
        <div style="width: 120px;">{home_matches.iloc[j]['date'].date()} {home_matches.iloc[j]['time']}</div>
        <div style="width: 200px;">{home_matches.iloc[j]['home_team']}</div>
        <div style="width: 100px;">{int(home_matches.iloc[j]['home_goals'])} - {int(home_matches.iloc[j]['away_goals'])}</div>
        <div style="width: 200px;">{home_matches.iloc[j]['away_team']}</div>
    """
    if home_matches.iloc[j]['home_team'] == home_team:
        if home_matches.iloc[j]['home_goals'] > home_matches.iloc[j]['away_goals']:
            last_home += "<div style='width: 50px; background-color: green;'>W</div>"
        elif home_matches.iloc[j]['home_goals'] < home_matches.iloc[j]['away_goals']:
            last_home += "<div style='width: 50px; background-color: red;'>L</div>"
        else:
            last_home += "<div style='width: 50px; background-color: gray;'>D</div>"
    else:
        if home_matches.iloc[j]['home_goals'] > home_matches.iloc[j]['away_goals']:
            last_home += "<div style='width: 50px; background-color: red;'>L</div>"
        elif home_matches.iloc[j]['home_goals'] < home_matches.iloc[j]['away_goals']:
            last_home += "<div style='width: 50px; background-color: green;'>W</div>"
        else:
            last_home += "<div style='width: 50px; background-color: gray;'>D</div>"
    last_home+="</div>"
last_home += "</div>"

last_away = f"""
            <div style="text-align: center; font-size: 15px;
                background-color: #f8f9fa; 
                border-radius: 10px; 
                padding: 5px; 
                margin: 5px;
                box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
            <div style="display: flex; justify-content: center; margin-bottom: 5px; font-size: 25px; font-weight:bold;">Ostatnie mecze {away_team}</div>
            """

if len(away_matches) == 0:
    last_away += "<div style='font-size: 20px;'>Brak danych na temat ostatnich meczów</div>"

for j in range(min(5, len(away_matches))):
    last_away += f"""
    <div style="display: flex; justify-content: center; margin-bottom: 5px;">
        <div style="width: 120px;">{away_matches.iloc[j]['date'].date()} {away_matches.iloc[j]['time']}</div>
        <div style="width: 200px;">{away_matches.iloc[j]['home_team']}</div>
        <div style="width: 100px;">{int(away_matches.iloc[j]['home_goals'])} - {int(away_matches.iloc[j]['away_goals'])}</div>
        <div style="width: 200px;">{away_matches.iloc[j]['away_team']}</div>
    """
    if away_matches.iloc[j]['home_team'] == away_team:
        if away_matches.iloc[j]['home_goals'] > away_matches.iloc[j]['away_goals']:
            last_away += "<div style='width: 50px; background-color: green;'>W</div>"
        elif away_matches.iloc[j]['home_goals'] < away_matches.iloc[j]['away_goals']:
            last_away += "<div style='width: 50px; background-color: red;'>L</div>"
        else:
            last_away += "<div style='width: 50px; background-color: gray;'>D</div>"
    else:
        if away_matches.iloc[j]['home_goals'] > away_matches.iloc[j]['away_goals']:
            last_away += "<div style='width: 50px; background-color: red;'>L</div>"
        elif away_matches.iloc[j]['home_goals'] < away_matches.iloc[j]['away_goals']:
            last_away += "<div style='width: 50px; background-color: green;'>W</div>"
        else:
            last_away += "<div style='width: 50px; background-color: gray;'>D</div>"
    last_away+="</div>"
last_away += "</div>"


with tab5:
    col2, col3 = st.columns(2)
    with col2:        
        st.markdown(last_home, unsafe_allow_html=True)

    with col3:        
        st.markdown(last_away, unsafe_allow_html=True)
        

    
    filtr1, filtr2 = st.columns(2)

    css = """
    .st-key-team_filter *, .st-key-stat_filter *, .st-key-team_filter, .st-key-stat_filter {
        cursor: pointer;
    }
    .stMainBlockContainer {
        padding-top: 3rem !important;
    }
    """
    st.html(f"<style>{css}</style>")
    with filtr1:
        team_filter = st.selectbox("Wybierz drużynę", options=[home_team, away_team], key="team_filter")
    with filtr2:
        stat_filter = st.selectbox("Wybierz statystykę", options=["Bramki w meczu", "Strzelone bramki", "Stracone bramki"], key="stat_filter")
    

    team = team_filter
    stat_name = stat_filter
    n = 10
    last_matches = select_last_matches(matches, team, date, n)
    if stat_name == "Strzelone bramki":
        stat = "goals"
        stat_df = get_stat(last_matches, team, stat)
        threshold = 1.5
    if stat_name == "Stracone bramki":
        stat = "goals"
        stat_df = get_stat(last_matches, team, stat, True)
        threshold = 1.5
    if stat_name == "Bramki w meczu":
        stat = "goals"
        stat_df = get_stat(last_matches, team, stat, sum = True)
        threshold = 2.5

    # Set colors: green if above threshold, red otherwise
    colors = ["green" if val > threshold else "red" for val in stat_df[stat]]

    # Plot the bar chart
    fig3 = plt.figure(figsize=(10, 6))
    bars = plt.bar(stat_df["new_date"], stat_df[stat], color=colors)

    # Add threshold line
    plt.axhline(y=threshold, color="gray", linestyle="--", label=f"Linia = {threshold}")

    # Add value labels on top of bars
    for bar in bars:
        height = int(bar.get_height())
        if height == 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                ha="center", va="bottom", fontsize=20)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height-0.4, str(height),
                ha="center", va="bottom", fontsize=20)

    # Chart styling
    plt.title(stat_name.capitalize() + " " + team + " w ostatnich " + str(n) + " meczach")
    plt.xlabel("Mecze")
    plt.ylabel(stat)
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()
    st.write(fig3)

    h2h = matches2[((matches2["home_team"] == home_team) & (matches2["away_team"] == away_team)) |
            ((matches2["home_team"] == away_team) & (matches2["away_team"] == home_team))]
    h2h = h2h[["date", "time", "home_team", "home_goals", "away_goals", "away_team"]]
    h2h = h2h.sort_values(by="date", ascending=False)
    data = """
                <div style="text-align: center; font-size: 15px;
                    border-radius: 10px; 
                    padding: 5px; 
                    margin: 5px;
                    box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
                <div style="display: flex; justify-content: center; margin-bottom: 5px; font-size: 25px; font-weight:bold;">Mecze pomiędzy drużynami</div>
                """
    
    if len(h2h) == 0:
        data += "<div style='font-size: 20px;'>Brak danych na temat meczów pomiędzy drużynami</div>"
    for j in range(len(h2h)):
        if h2h.iloc[j]['date'] == date:
            data += """<div style="display: flex; justify-content: center; margin-bottom: 5px; font-size: 15px; background-color:Aqua;">"""
        elif h2h.iloc[j]['date'] > date:
            data += """<div style="display: flex; justify-content: center; margin-bottom: 5px; font-size: 15px; background-color:LightGreen;">"""
        else:
            data += """<div style="display: flex; justify-content: center; margin-bottom: 5px; font-size: 15px; background-color:LightCoral;">"""
        data += f"""
            <div style="width: 120px;">{h2h.iloc[j]['date']} {h2h.iloc[j]['time']}</div>
            <div style="width: 200px;">{h2h.iloc[j]['home_team']}</div>
            <div style="width: 100px;">{int(h2h.iloc[j]['home_goals'])} - {int(h2h.iloc[j]['away_goals'])}</div>
            <div style="width: 200px;">{h2h.iloc[j]['away_team']}</div>
        </div>
        """
    data += "</div>"
    st.markdown(data, unsafe_allow_html=True)


############ Zakładka z informacjami ############
selected_columns_standings = ['team', 'matches_played', 'wins', 'draws', 'defeats', 'goal_difference', 'goals', 'goals_conceded', 'points']
table = standings[selected_columns_standings]
table = table.sort_values(["points", "goal_difference", "goals"], ascending=False)
table['place'] = range(1, len(table) + 1)
table = table.set_index('place')
standings_data = []
for i, row in table.iterrows():
    if row["team"] == home_team or row["team"] == away_team:
        team_stats = {}
        team_stats["position"] = i
        team_stats["highlight"] = ""
        if i<5:
            team_stats["highlight"] = "green"
        if i>17:
            team_stats["highlight"] = "red"
        team_stats["name"] = row['team']
        team_stats["played"] = row['matches_played']
        team_stats["wins"] = row['wins']
        team_stats["draws"] = row['draws']
        team_stats["losses"] = row['defeats']
        team_stats["diff"] = row['goal_difference']
        if team_stats["diff"] > 0:
            team_stats["diff"] = "+" + str(row['goal_difference'])
        team_stats["goals_scored"] = row['goals']
        team_stats["goals_conceded"] = row['goals_conceded']
        team_stats["points"] = row['points']
        standings_data.append(team_stats)

html_table = generate_html_table(standings_data)
html_table_final = """
<style>
    .tab {
        border: 1px solid #ddd;
        border-radius: 8px;
        width: 90%;
        padding: 16px;
        background-color: #f9f9f9;
        margin: auto;
    }
    .tab_title {
        font-size: 22px;
        font-weight: bold;
        color: #333;
        width: 100%;
        text-align: center;
        margin-bottom: 12px;
    }
</style> """ + f"""
<div class="tab">
<div class="tab_title">Klasyfikacja przedmeczowa</div>
{html_table}
</div>
"""

st.markdown(
    """
    <style>
    .tab {
        border: 1px solid #ddd;
        border-radius: 8px;
        width: 400px;
        padding: 16px;
        background-color: #f9f9f9;
        margin: auto;
    }
    .row {
        margin-bottom: 8px;
    }
    .small_title {
        font-size: 12px;
        color: #777;
        margin-bottom: -2px;
    }
    .text {
        font-size: 16px;
        font-weight: bold;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

league_map = {
    "pl": "Premier League",
    "ll": "La Liga",
    "bl": "Bundesliga",
    "l1": "Ligue 1",
    "sa": "Serie A",
}

ref_df = matches[matches["referee"] == curr_match["referee"]][["home_cards_yellow", "home_cards_red", "away_cards_yellow", "away_cards_red"]]
ref_df.dropna(how='all')
ref_df["sum_yellow"] = ref_df["home_cards_yellow"] + ref_df["away_cards_yellow"]
ref_df["sum_red"] = ref_df["home_cards_red"] + ref_df["away_cards_red"]
yellow_avg = ref_df["sum_yellow"].mean()
red_avg = ref_df["sum_red"].mean()

# HTML structure for the tab
tab_html = f"""
<div class="tab">
    <div class="row">
        <div class="small_title">Data i godzina</div>
        <div class="text">{curr_match['date'].date().strftime('%d.%m.%Y')} • {curr_match['time']}</div>
    </div>
    <div class="row">
        <div class="small_title">Rozgrywki</div>
        <div class="text">{league_map[curr_match["league"]]}, Kolejka {curr_match["round"]}</div>
    </div>
    <div class="row">
        <div class="small_title">Stadion</div>
        <div class="text">{curr_match["Stadium"]}</div>
    </div>
    <div class="row">
        <div class="small_title">Lokalizacja</div>
        <div class="text">{curr_match["City"]}, {curr_match["Country"]}</div>
    </div>
    <div class="row">
        <div class="small_title">{"Frekwencja" if curr_match["attendance_value"] > 0 else "Pojemność"}</div>
        <div class="text">{int(curr_match["attendance_value"]) if curr_match["attendance_value"] > 0 else curr_match["Capacity"]}</div>
    </div>
    <div class="row">
        <div class="small_title">Sędzia</div>
        <div class="text">{curr_match["referee"] if curr_match["referee"] != "" else "-"}</div>
    </div>
    <div class="row">
        <div class="small_title">Średnie kartki</div>
        <div class="text">
            <span><span class="emoji">🟨</span>{yellow_avg:.2f}</span>
            <span><span class="emoji">🟥</span>{red_avg:.2f}</span>
        </div>
    </div>
</div>
"""



with tab1:
    col1, col2 = st.columns([1,1])
    with col1:
        st.components.v1.html(html_table_final, height=280)
        # st.pyplot(fig21)
        # if (len(probabilities)>0):
        #     st.pyplot(fig22)
    with col2:
        st.markdown(tab_html, unsafe_allow_html=True)

############ Zakładka z kursami ############
wyniki = [
    ["1:0", getCourse(match_probabilities["exact_10"])], ["2:0", getCourse(match_probabilities["exact_20"])], ["2:1", getCourse(match_probabilities["exact_21"])],
    ["3:0", getCourse(match_probabilities["exact_30"])], ["3:1", getCourse(match_probabilities["exact_31"])], ["3:2", getCourse(match_probabilities["exact_32"])],
    ["0:1", getCourse(match_probabilities["exact_01"])], ["0:2", getCourse(match_probabilities["exact_02"])], ["1:2", getCourse(match_probabilities["exact_12"])],
    ["0:3", getCourse(match_probabilities["exact_03"])], ["1:3", getCourse(match_probabilities["exact_13"])], ["2:3", getCourse(match_probabilities["exact_23"])],
    ["0:0", getCourse(match_probabilities["exact_00"])], ["1:1", getCourse(match_probabilities["exact_11"])], ["2:2", getCourse(match_probabilities["exact_22"])], 
    ["3:3", getCourse(match_probabilities["exact_33"])], ["4:0", getCourse(match_probabilities["exact_40"])], ["0:4", getCourse(match_probabilities["exact_04"])],
    ["4:1", getCourse(match_probabilities["exact_41"])], ["4:2", getCourse(match_probabilities["exact_42"])], ["4:3", getCourse(match_probabilities["exact_43"])],
    ["3:4", getCourse(match_probabilities["exact_34"])], ["2:4", getCourse(match_probabilities["exact_24"])], ["1:4", getCourse(match_probabilities["exact_14"])],
]

# Konwersja danych do macierzy
rows, cols = 4, 3  # Liczba wierszy i kolumn
# table_data = [wyniki[i:i + cols] for i in range(0, len(wyniki), cols)]
table_data = [wyniki[i:i + cols] for i in range(0, len(wyniki), cols)]
match_score = str(curr_match["home_goals"]) + ":" + str(curr_match["away_goals"])

correct_scores = ""
for row in table_data:
    for cell in row:
        wynik, kurs = cell
        if wynik == match_score:
            correct_scores += f"""
            <div class="cell correct">
                <span class="result">{wynik}</span>
                <span class="odds">{kurs:.2f}</span>
            </div>
            """
        else:
            correct_scores += f"""
            <div class="cell">
                <span class="result">{wynik}</span>
                <span class="odds">{kurs:.2f}</span>
            </div>
            """

odds_style = """
<style>
    .odds-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        max-width: 1000px;
        width: 90%;
        margin: 20px auto;
        font-family: Arial, sans-serif;
    }

    .odds-group {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #f5f5f5;
        color: #333;
        width: 100%;
    }

    .odds-group h3 {
        margin-top: 0;
        margin-bottom: 18px;
        font-size: 24px;
        text-align: center;
        color: #111;
    }

    .table-container {
        display: flex;
        flex-wrap: wrap;
        gap: 25px;
        align-items: center;
        justify-content: center;
    }

    .cell {
        width: calc(25% - 25px);
        height: 62px;
        background-color: #333333;
        border-radius: 12px;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        padding: 0 15px;
        color: white;
    }

    .correct {
        background-color: #26943b;
    }

    .result {
        font-size: 13px;
        text-align: left;
    }

    .odds {
        font-size: 17px;
        font-weight: bold;
        text-align: right;
    }
    .two-column .cell {
    width: calc(45% - 25px); /* Two cells per row with gap adjustment */
    box-sizing: border-box; /* Ensure padding doesn't affect width */
}
</style>
"""
odds_content = f"""<div class="odds-container">
    <!-- 1x2 Odds Group -->
    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Wynik meczu</h3>
        <div class="table-container">
            <!-- Example cells for 1x2 odds -->
            <div class="cell{" correct" if curr_match["home_goals"] > curr_match["away_goals"] else ""}">
                <span class="result">1</span>
                <span class="odds">{getCourse(match_probabilities["home_win"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] == curr_match["away_goals"] else ""}">
                <span class="result">X</span>
                <span class="odds">{getCourse(match_probabilities["draw"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] < curr_match["away_goals"] else ""}">
                <span class="result">2</span>
                <span class="odds">{getCourse(match_probabilities["away_win"]):.2f}</span>
            </div>
        </div>
    </div>

    <!-- No Draw Odds Group -->
    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Mecz bez remisu</h3>
        <div class="table-container two-column">
            <!-- Example cells for no draw odds -->
            <div class="cell{" correct" if curr_match["home_goals"] > curr_match["away_goals"] else ""}">
                <span class="result">1</span>
                <span class="odds">{getCourse(match_probabilities["no_draw_home_win"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] < curr_match["away_goals"] else ""}">
                <span class="result">2</span>
                <span class="odds">{getCourse(match_probabilities["no_draw_away_win"]):.2f}</span>
            </div>
        </div>
    </div>

    <!-- Double Chance Odds Group -->
    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Podwójna szansa</h3>
        <div class="table-container">
            <!-- Example cells for double chance odds -->
            <div class="cell{" correct" if curr_match["home_goals"] >= curr_match["away_goals"] else ""}">
                <span class="result">1X</span>
                <span class="odds">{getCourse(match_probabilities["1x"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] <= curr_match["away_goals"] else ""}">
                <span class="result">X2</span>
                <span class="odds">{getCourse(match_probabilities["x2"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] != curr_match["away_goals"] else ""}">
                <span class="result">12</span>
                <span class="odds">{getCourse(match_probabilities["12"]):.2f}</span>
            </div>
        </div>
    </div>

    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Obie drużyny strzelą bramkę</h3>
        <div class="table-container two-column">
            <div class="cell{" correct" if curr_match["home_goals"] > 0 and curr_match["away_goals"] > 0 else ""}">
                <span class="result">tak</span>
                <span class="odds">{getCourse(match_probabilities["btts"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] < 1 or curr_match["away_goals"] < 1 else ""}">
                <span class="result">nie</span>
                <span class="odds">{getCourse(match_probabilities["no_btts"]):.2f}</span>
            </div>
        </div>
    </div>

    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Liczba bramek w meczu</h3>
        <div class="table-container two-column">
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] > 0.5 else ""}">
                <span class="result">+0.5</span>
                <span class="odds">{getCourse(match_probabilities["over05"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] < 0.5 else ""}">
                <span class="result">-0.5</span>
                <span class="odds">{getCourse(match_probabilities["under05"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] > 1.5 else ""}">
                <span class="result">+1.5</span>
                <span class="odds">{getCourse(match_probabilities["over15"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] < 1.5 else ""}">
                <span class="result">-1.5</span>
                <span class="odds">{getCourse(match_probabilities["under15"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] > 2.5 else ""}">
                <span class="result">+2.5</span>
                <span class="odds">{getCourse(match_probabilities["over25"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] < 2.5 else ""}">
                <span class="result">-2.5</span>
                <span class="odds">{getCourse(match_probabilities["under25"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] > 3.5 else ""}">
                <span class="result">+3.5</span>
                <span class="odds">{getCourse(match_probabilities["over35"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] + curr_match["away_goals"] < 3.5 else ""}">
                <span class="result">-3.5</span>
                <span class="odds">{getCourse(match_probabilities["under35"]):.2f}</span>
            </div>
        </div>
    </div>
    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Liczba bramek {home_team}</h3>
        <div class="table-container two-column">
            <div class="cell{" correct" if curr_match["home_goals"] > 0.5 else ""}">
                <span class="result">+0.5</span>
                <span class="odds">{getCourse(match_probabilities["home_over05"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] < 0.5 else ""}">
                <span class="result">-0.5</span>
                <span class="odds">{getCourse(match_probabilities["home_under05"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] > 1.5 else ""}">
                <span class="result">+1.5</span>
                <span class="odds">{getCourse(match_probabilities["home_over15"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] < 1.5 else ""}">
                <span class="result">-1.5</span>
                <span class="odds">{getCourse(match_probabilities["home_under15"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] > 2.5 else ""}">
                <span class="result">+2.5</span>
                <span class="odds">{getCourse(match_probabilities["home_over25"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["home_goals"] < 2.5 else ""}">
                <span class="result">-2.5</span>
                <span class="odds">{getCourse(match_probabilities["home_under25"]):.2f}</span>
            </div>
        </div>
    </div>
    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Liczba bramek {away_team}</h3>
        <div class="table-container two-column">
            <div class="cell{" correct" if curr_match["away_goals"] > 0.5 else ""}">
                <span class="result">+0.5</span>
                <span class="odds">{getCourse(match_probabilities["away_over05"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["away_goals"] < 0.5 else ""}">
                <span class="result">-0.5</span>
                <span class="odds">{getCourse(match_probabilities["away_under05"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["away_goals"] > 1.5 else ""}">
                <span class="result">+1.5</span>
                <span class="odds">{getCourse(match_probabilities["away_over15"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["away_goals"] < 1.5 else ""}">
                <span class="result">-1.5</span>
                <span class="odds">{getCourse(match_probabilities["away_under15"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["away_goals"] > 2.5 else ""}">
                <span class="result">+2.5</span>
                <span class="odds">{getCourse(match_probabilities["away_over25"]):.2f}</span>
            </div>
            <div class="cell{" correct" if curr_match["away_goals"] < 2.5 else ""}">
                <span class="result">-2.5</span>
                <span class="odds">{getCourse(match_probabilities["away_under25"]):.2f}</span>
            </div>
        </div>
    </div>
    <div class="odds-group" style="background-color: #d1e7dd;">
        <h3>Dokładny Wynik</h3>
        <div class="table-container">
            {correct_scores}
        </div>
    </div>
</div>
"""

odds_html = odds_style + odds_content
# Wyświetlenie tabeli w Streamlit
with tab4:
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.components.v1.html(odds_html, height=2580)