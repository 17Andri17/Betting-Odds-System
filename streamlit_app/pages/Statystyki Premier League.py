import streamlit as st
import numpy as np
import pandas as pd
import runpy


df=pd.read_csv("../premier_league_21_24_stadiums_weather.csv")
df['date_x']=pd.to_datetime(df['date_x'])
df['date_x'] = df['date_x'].dt.strftime('%Y-%m-%d %H:%M')

st.write(df[df["Unnamed: 0"]==st.session_state['stats_id']])