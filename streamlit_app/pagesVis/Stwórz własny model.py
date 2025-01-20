import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_poisson_prediction():
    st.title("Własny model Poissona dla przewidywania wyników meczów")

    st.header("Rozkład Poissona")

    st.subheader("Opis modelu")
    st.write(
        """Rozkład Poissona jest dyskretnym rozkładem prawdopodobieństwa, który wyraża prawdopodobieństwo uzyskania określonej liczby zdarzeń w danym przedziale czasowym, 
        pod warunkiem, że te zdarzenia zachodzą niezależnie i w stałym tempie. Prawdopodobieństwo wystąpienia \(k\) zdarzeń oblicza się wzorem:
        """
    )
    latex_expr = r"P(k) = \frac{e^{-\lambda} \lambda^k}{k!}"
    st.write(f"$$ {latex_expr} $$")

    st.write(
        """Liczba bramek strzelanych przez drużynę w meczu, traktowanych jako niezależne i występujące z określoną średnią intensywnością, 
        może być modelowana za pomocą tego rozkładu. Dla meczu między drużyną gospodarzy H a drużyną gości A, 
        prawdopodobieństwo konkretnego wyniku jest iloczynem prawdopodobieństw dla obu drużyn:
        """
    )
    latex_expr = r"P(k, h) = \frac{\lambda_{\text{H}}^k e^{-\lambda_{\text{H}}}}{k!} \times \frac{\lambda_{\text{A}}^h e^{-\lambda_{\text{A}}}}{h!}"
    st.write(f"$$ {latex_expr} $$")

    st.write("### Przykładowa tabela rozkładu wyników meczów")
    data = {
        'H\\A': ['0', '1', '2', '3+'],
        '0': [0.05, 0.08, 0.12, 0.0],
        '1': [0.1, 0.1, 0.1, 0.1],
        '2': [0.05, 0.1, 0.05, 0.05],
        '3+': [0.01, 0.02, 0.03, 0.02],
    }
    df = pd.DataFrame(data)
    st.table(df)

    st.write("Zwycięstwo drużyny H możemy obliczyć jako sumę liczb prawdopodobieństw pod diagonalą.")


    st.subheader("Wzory na obliczanie lambd")
    st.write("Parametrami naszego modelu są liczby $$\lambda_{H}$$ i $$\lambda_{A}$$. Wyliczamy je przy użyciu następujących wzorów:")

    latex_expr_h = r"\lambda_{\text{H}} = \text{SO}_{\text{H}} \cdot \text{SD}_{\text{A}} \cdot \bar{g}_{\text{H}}"
    latex_expr_a = r"\lambda_{\text{A}} = \text{SO}_{\text{A}} \cdot \text{SD}_{\text{H}} \cdot \bar{g}_{\text{A}}"
    st.write(f"$$ {latex_expr_h} $$")
    st.write(f"$$ {latex_expr_a} $$")

    st.write(
        """Gdzie:
        - $$\lambda_{H}$$ – oczekiwana liczba goli zdobytych przez drużynę gospodarzy,
        - $$\lambda_{A}$$ – oczekiwana liczba goli zdobytych przez drużynę gości,
        - $${SO}_{H}$$ – siła ofensywna drużyny gospodarzy,
        - $${SO}_{A}$$ – siła ofensywna drużyny gości,
        - $${SD}_{H}$$ – siła defensywna drużyny gospodarzy,
        - $${SD}_{A}$$ – siła defensywna drużyny gości,
        - $${g}_{H}$$– średnia liczba goli strzelanych przez gospodarzy u siebie,
        - $${g}_{A}$$– średnia liczba goli strzelanych przez gości na wyjeździe.
        """
    )


    st.header("Wprowadź dane drużyn")
    
    home_team = st.text_input("Nazwa drużyny gospodarzy", "Arsenal")
    away_team = st.text_input("Nazwa drużyny gości", "Chelsea")

    st.subheader("Statystyki drużyny gospodarzy")
    home_avg_goals = st.number_input("Średnia liczba bramek w ostatnich 5 meczach", min_value=0.0, value=1.5, step=0.1)
    home_offensive_power = st.number_input("Siła ofensywna", min_value=0.0, value=1.2, step=0.1)
    home_defensive_power = st.number_input("Siła defensywna", min_value=0.0, value=0.8, step=0.1)

    st.subheader("Statystyki drużyny gości")
    away_avg_goals = st.number_input("Średnia liczba bramek w ostatnich 5 meczach", min_value=0.0, value=1.3, step=0.1)
    away_offensive_power = st.number_input("Siła ofensywna", min_value=0.0, value=1.1, step=0.1)
    away_defensive_power = st.number_input("Siła defensywna", min_value=0.0, value=0.9, step=0.1)


    home_lambda = home_offensive_power * away_defensive_power * home_avg_goals
    away_lambda = away_offensive_power * home_defensive_power * away_avg_goals


    st.write(f"### Obliczone lambdy:")
    st.write(f"- Drużyna gospodarzy ({home_team}): {home_lambda:.2f}")
    st.write(f"- Drużyna gości ({away_team}): {away_lambda:.2f}")


    st.subheader("Macierz prawdopodobieństw")
    max_goals = st.slider("Maksymalna liczba bramek do wyświetlenia", min_value=3, max_value=10, value=5)

    home_goals = np.arange(0, max_goals + 1)
    away_goals = np.arange(0, max_goals + 1)

    probability_matrix = np.zeros((len(home_goals), len(away_goals)))

    for i, hg in enumerate(home_goals):
        for j, ag in enumerate(away_goals):
            probability_matrix[i, j] = poisson.pmf(hg, home_lambda) * poisson.pmf(ag, away_lambda)


    probability_data = pd.DataFrame(probability_matrix, index=home_goals, columns=away_goals)



    st.write(f"### Macierz prawdopodobieństwa dla gospodarzy {home_team} vs gości {away_team}")
    st.table(probability_data)

    home_win_prob = np.sum(np.tril(probability_matrix, -1))
    draw_prob = np.sum(np.diag(probability_matrix))
    away_win_prob = np.sum(np.triu(probability_matrix, 1))

    st.write("### Prawdopodobieństwo wyników")
    st.write(f"- Zwycięstwo gospodarzy ({home_team}): {home_win_prob:.2%}")
    st.write(f"- Remis: {draw_prob:.2%}")
    st.write(f"- Zwycięstwo gości ({away_team}): {away_win_prob:.2%}")

show_poisson_prediction()