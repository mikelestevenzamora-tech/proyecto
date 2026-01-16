import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 COMPARACIÓN DE JUGADORES (Para el Tab 3)
def compare_players(df, p1, p2):
    cols = ['xG', 'xAG', 'PrgP', 'Carries', 'Tkl+Int']
    if p1 not in df['Player'].values or p2 not in df['Player'].values:
        return pd.DataFrame()
    
    p1_data = df[df['Player'] == p1][cols].iloc[0]
    p2_data = df[df['Player'] == p2][cols].iloc[0]
    
    return pd.DataFrame({p1: p1_data, p2: p2_data})

# 🔹 JUGADORES SIMILARES (Corregido para la UI)
def jugadores_similares(df, jugador, n=4):
    base = df[df['Player'] == jugador]
    if base.empty:
        return pd.DataFrame()
    
    pos = base['Pos'].values[0]
    features = ['xG','xAG','Carries','PrgDist','PrgP']
    
    # Filtrar por misma posición y excluir al mismo jugador
    pool = df[(df['Pos'] == pos) & (df['Player'] != jugador)].copy()
    if pool.empty:
        return pd.DataFrame()
    
    # Calcular similitud de coseno
    sim = cosine_similarity(base[features].fillna(0), pool[features].fillna(0))
    pool['Similarity'] = sim[0]
    
    # Devolver los N más similares con las columnas necesarias para el app
    return pool.sort_values('Similarity', ascending=False).head(n)

# 🔹 ANÁLISIS DE EQUIPOS (Radar)
def club_dna_vector(df, team):
    metrics = ['xG','xAG','PrgP','PrgDist','Carries','Tkl+Int']
    team_df = df[df['Squad'] == team]
    if team_df.empty:
        return pd.Series([0]*len(metrics), index=metrics)
    return team_df[metrics].mean()

def radar_data(df, team):
    cdv = club_dna_vector(df, team)
    return cdv.index.tolist(), cdv.values.tolist()

# 🔹 PREDICCIÓN DE ENCUENTRO
def matchup_predictor(df, team1, team2):
    metrics = ['xG','xAG','PrgP','PrgDist','Carries','Tkl+Int']
    t1_score = df[df['Squad']==team1][metrics].mean().sum()
    t2_score = df[df['Squad']==team2][metrics].mean().sum()
    
    if t1_score > t2_score:
        return f"Ventaja táctica para {team1} (Basado en volumen de juego)"
    return f"Ventaja táctica para {team2} (Basado en volumen de juego)"

import plotly.graph_objects as go

# 🔹 COMPARACIÓN DE JUGADORES (Corregida para devolver un gráfico)
def compare_players(df, p1, p2):
    cols = ['xG', 'xAG', 'PrgP', 'Carries', 'Tkl+Int']
    
    if p1 not in df['Player'].values or p2 not in df['Player'].values:
        return None
    
    p1_data = df[df['Player'] == p1][cols].iloc[0].values.tolist()
    p2_data = df[df['Player'] == p2][cols].iloc[0].values.tolist()
    
    fig = go.Figure()

    # Añadir Jugador 1
    fig.add_trace(go.Scatterpolar(
        r=p1_data + [p1_data[0]],
        theta=cols + [cols[0]],
        fill='toself',
        name=p1,
        line_color="#bc6c25" # Copperwood
    ))

    # Añadir Jugador 2
    fig.add_trace(go.Scatterpolar(
        r=p2_data + [p2_data[0]],
        theta=cols + [cols[0]],
        fill='toself',
        name=p2,
        line_color="#dda15e" # Sunlit-clay
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=False),
            bgcolor="rgba(255, 255, 255, 0.05)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fefae0",
        showlegend=True,
        legend=dict(orientation="h", y=-0.2)
    )
    
    return fig