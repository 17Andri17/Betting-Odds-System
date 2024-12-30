import streamlit as st
import numpy as np
import pandas as pd

pg = st.navigation([st.Page("Kursomat.py"), st.Page("pagesVis/Premier League.py"), st.Page("pagesHid/Statystyki Premier League.py")], position="hidden")
pg.run()