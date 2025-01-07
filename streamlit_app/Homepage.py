import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

pg = st.navigation([st.Page("Kursomat.py"), st.Page("pagesVis/Premier League.py"), st.Page("pagesHid/Statystyki Premier League.py")])
pg.run()