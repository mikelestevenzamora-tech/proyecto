import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from mplsoccer import Pitch
from utils import radar_data, matchup_predictor, jugadores_similares, compare_players

# --------------------------------------------------
# CONFIGURACIÓN PROFESIONAL
# --------------------------------------------------
st.set_page_config(page_title="Football Intel Pro", layout="wide", page_icon="⚽")

colors = {
    "olive-leaf": "#606c38",
    "black-forest": "#283618",
    "cornsilk": "#fefae0",
    "sunlit-clay": "#dda15e",
    "copperwood": "#bc6c25",
    "pitch-dark": "#1a1a1a"
}

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background-color: {colors['black-forest']}; color: {colors['cornsilk']}; }}
    .main-header {{ text-align: center; color: {colors['cornsilk']}; padding: 20px; border-bottom: 2px solid {colors['sunlit-clay']}; }}
    .player-card {{ background: {colors['olive-leaf']}; border-radius: 15px; padding: 20px; border: 1px solid {colors['sunlit-clay']}; text-align: center; }}
    .metric-box {{ background-color: {colors['cornsilk']}; padding: 15px; border-radius: 12px; text-align: center; border-left: 5px solid {colors['copperwood']}; margin-bottom: 10px; }}
    .metric-title {{ color: {colors['black-forest']}; font-size: 11px; font-weight: bold; text-transform: uppercase; }}
    .metric-value {{ color: {colors['copperwood']}; font-size: 24px; font-weight: 800; }}
    .stTabs [data-baseweb="tab"] {{ color: {colors['cornsilk']}; font-weight: bold; }}
    .stTabs [aria-selected="true"] {{ background-color: {colors['olive-leaf']} !important; border-radius: 10px 10px 0 0; }}
    /* Centrar el contenedor de las pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        justify-content: center;
        width: 100%;
    }}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("final.csv")

df = load_data()
def draw_tactical_pitch(df, team_left, team_right):
    fig = go.Figure()


def draw_mplsoccer_pitch_from_csv(df, team_left, team_right):
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a1a",
        line_color="#444444"
    )

    fig, ax = pitch.draw(figsize=(14, 9))
    fig.set_facecolor("#1a1a1a")

    # --- POSICIÓN X (NO CRUZA MITAD) ---
    def get_x(pos, side):
        x_map_left = {
            "GK": 6,
            "DF": 20,
            "MF": 38,
            "FW": 54
        }

        x_map_right = {
            "GK": 114,
            "DF": 100,
            "MF": 82,
            "FW": 66
        }

        pos_key = pos[:2]
        return x_map_left.get(pos_key, 38) if side == "left" else x_map_right.get(pos_key, 82)

    # --- POSICIÓN Y ---
    def get_y(idx, total):
        if total == 1:
            return 40
        step = 60 / (total - 1)
        return 10 + idx * step

    # --- DIBUJO DE JUGADORES ---
    for team, side in [(team_left, "left"), (team_right, "right")]:
        lineup = (
            df[df["Squad"] == team]
            .nlargest(11, "Min")
            .copy()
        )

        lineup["Line"] = lineup["Pos"].str[:2]
        counts = lineup["Line"].value_counts()
        idxs = {l: 0 for l in counts.index}

        for _, p in lineup.iterrows():
            x = get_x(p["Line"], side)
            y = get_y(idxs[p["Line"]], counts[p["Line"]])
            idxs[p["Line"]] += 1

            rating = p.get("Rating", 6.5)
            color = "#2ecc71" if rating >= 7 else "#f1c40f"

            pitch.scatter(
                x, y,
                s=700,
                color="white",
                edgecolors="#333",
                ax=ax,
                zorder=3
            )

            ax.text(
                x, y + 3.5,
                f"{rating:.2f}",
                ha="center", va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.25")
            )

            ax.text(
                x, y - 6,
                p["Player"].split()[-1],
                ha="center", va="center",
                color="white",
                fontsize=10,
                fontweight="bold"
            )

    return fig
# --------------------------------------------------
# INTERFAZ PRINCIPAL
# --------------------------------------------------
st.markdown('<h1 class="main-header">FOOTBALL INTEL PRO</h1>', unsafe_allow_html=True)
# Actualiza tu línea de tabs:
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Scouting Individual", 
    "Análisis de Equipos", 
    "Jugador vs Jugador",
    "Market Discovery",
    "Performance Labs"
])
# -----------------
# TAB 1: SCOUTING INDIVIDUAL
# -----------------
with tab1:
    col_sel, _ = st.columns([1, 2])
    p_name = col_sel.selectbox("Seleccionar Jugador", sorted(df["Player"].unique()), key="scout_p")
    row = df[df["Player"] == p_name].iloc[0]

    c1, c2 = st.columns([1.2, 3.5])
    with c1:
        st.markdown(f"""
        <div class="player-card">
            <img src="{row['PlayerImg']}" style="width:100%; border-radius:12px; border:2px solid {colors['cornsilk']}">
            <h2 style="margin-top:10px;">{row['Player']}</h2>
            <p style="color:{colors['sunlit-clay']}">{row['Squad']} | {row['Pos']} | {int(row['Age'])} años</p>
            <img src="{row['TeamImg']}" width="60">
        </div>
        """, unsafe_allow_html=True)
    with c2:
        m_c1, m_c2 = st.columns(2)
        metrics = [("Expected Goals (xG)", row["xG"]), ("Expected Assists (xAG)", row["xAG"]), 
                   ("Progression (m)", row["PrgDist"]), ("Fatigue Index", row["FatigueIndex"])]
        for i, (label, val) in enumerate(metrics):
            target_col = m_c1 if i % 2 == 0 else m_c2
            target_col.markdown(f'<div class="metric-box"><div class="metric-title">{label}</div><div class="metric-value">{val:.2f}</div></div>', unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; margin-bottom: 20px;">
                <span style="color: {colors['sunlit-clay']}; font-weight: bold;">Estilo de Juego:</span> {row['PlayerStyle']}
            </div>
        """, unsafe_allow_html=True)

        labels, values = radar_data(df[df['Player']==p_name], row['Squad'])
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]], 
            theta=labels + [labels[0]], 
            fill='toself', 
            fillcolor=f"rgba(188, 108, 37, 0.4)", 
            line=dict(color=colors['copperwood'], width=2)
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            polar=dict(
                bgcolor="rgba(254, 250, 224, 0.05)",
                radialaxis=dict(visible=True, gridcolor=colors['sunlit-clay'], showticklabels=False),
                angularaxis=dict(gridcolor=colors['sunlit-clay'])
            ), 
            font_color=colors['cornsilk'], 
            height=450,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Jugadores de Perfil Similar")
    sims = jugadores_similares(df, p_name)
    if not sims.empty:
        cols = st.columns(len(sims))
        for i, srow in enumerate(sims.iloc):
            with cols[i]:
                st.markdown(f"""
                <div style="background:{colors['olive-leaf']}; padding:15px; border-radius:10px; text-align:center; border:1px solid {colors['sunlit-clay']}">
                    <img src="{srow['PlayerImg']}" width="70" style="border-radius:50%">
                    <p style="margin:5px 0 0 0;"><b>{srow['Player']}</b></p>
                    <small style="color:{colors['sunlit-clay']}">{srow['Similarity']:.1%} Match</small>
                </div>
                """, unsafe_allow_html=True)
# -----------------
# TAB 2: ANÁLISIS DE EQUIPOS
# -----------------
with tab2:
    t1_col, t2_col = st.columns(2)
    teamA = t1_col.selectbox("Equipo Local (Izquierda)", sorted(df["Squad"].unique()), index=0, key="team_a_sel")
    teamB = t2_col.selectbox("Equipo Visitante (Derecha)", sorted(df["Squad"].unique()), index=1, key="team_b_sel")
    
    st.markdown(f"""
        <div style="text-align:center; padding:15px; background:rgba(221,161,94,0.1); border-radius:15px; border:1px solid {colors['sunlit-clay']}; margin-bottom:25px;">
            <h3 style='margin:0; color:{colors['sunlit-clay']};'>Predicción: {matchup_predictor(df, teamA, teamB)}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col_pitch, col_fatiga = st.columns([2, 1.2])
    
    with col_pitch:
        st.markdown(f"<p style='text-align:center; color:{colors['cornsilk']}'><b>Disposición Táctica y Calificaciones</b></p>", unsafe_allow_html=True)
        fig = draw_mplsoccer_pitch_from_csv(df, teamA, teamB)
        st.pyplot(fig)    

    with col_fatiga:
        # TÍTULO ESTILO APP
        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <span style="font-size: 30px; margin-right: 10px;">⚠️</span>
                <h2 style="margin: 0; color: white; font-size: 1.5em;">Riesgo de Lesión</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # FILTRADO DE JUGADORES POR LOS DOS EQUIPOS SELECCIONADOS
        fatigued_players = df[
            (df['Squad'].isin([teamA, teamB])) & 
            (df['FatigueIndex'] >= 1.5) # Umbral para mostrar en el ranking
        ].sort_values('FatigueIndex', ascending=False).head(6)

        if not fatigued_players.empty:
            for _, p in fatigued_players.iterrows():
                # Color de fondo dinámico según gravedad (basado en tus capturas)
                # Verde oscuro para fatiga moderada, más rojizo para crítica
                bg_color = "rgba(101, 115, 26, 0.8)" if p['FatigueIndex'] < 2.4 else "rgba(192, 57, 43, 0.4)"
                border_color = colors['sunlit-clay'] if p['Squad'] == teamA else "white"
                
                st.markdown(f"""
                    <div style="background:{bg_color}; padding:15px; border-radius:10px; border-left: 5px solid {border_color}; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:bold; color:white; font-size:0.95em;">{p['Player']}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:5px;">
                            <span style="color:rgba(255,255,255,0.7); font-size:0.8em;">{p['Squad']}</span>
                            <span style="color:white; font-size:0.85em;">Fatiga Crítica ({p['FatigueIndex']:.1f})</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style="color: grey; font-size: 0.8em; margin-top: 15px;">
                    Los jugadores en esta lista requieren rotación inmediata para evitar lesiones musculares.
                </p>
            """, unsafe_allow_html=True)
        else:
            st.info("No se detecta fatiga significativa en los jugadores de estos equipos.")
    # --- SECCIÓN: JUGADORES CLAVE (Mantenemos tu lógica original) ---
    st.markdown(f"<h4 style='text-align:center; margin-top:30px; margin-bottom:20px; color:{colors['cornsilk']}'> Jugadores con mayor impacto (G+A)</h4>", unsafe_allow_html=True)

    def get_key_players(df_total, team_name):
        team_df = df_total[df_total['Squad'] == team_name].copy()
        min_limit = team_df['Min'].max() * 0.5
        team_df['G+A'] = team_df['Gls'] + team_df['Ast']
        return team_df[team_df['Min'] >= min_limit].sort_values(['G+A', 'Min'], ascending=False).head(2)

    stars_a = get_key_players(df, teamA)
    stars_b = get_key_players(df, teamB)
    all_stars = pd.concat([stars_a, stars_b])

    s_cols = st.columns(len(all_stars))
    for i, (_, p) in enumerate(all_stars.iterrows()):
        side_color = colors['copperwood'] if p['Squad'] == teamA else colors['sunlit-clay']
        with s_cols[i]:
            st.markdown(f"""
            <div style="background:{colors['olive-leaf']}; padding:15px; border-radius:15px; border:2px solid {side_color}; text-align:center; min-height:220px;">
                <img src="{p['PlayerImg']}" style="width:70px; height:70px; border-radius:50%; object-fit:cover; border:2px solid white;">
                <div style="font-weight:bold; font-size:15px; margin-top:5px; color:white;">{p['Player']}</div>
                <div style="font-size:11px; color:{colors['cornsilk']}; opacity:0.8;">{p['Pos']} | {int(p['Min'])} min</div>
                <div style="display:flex; justify-content:space-around; margin-top:15px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
                    <div><small style="display:block; color:white;">Goles</small><b style="color:white;">{int(p['Gls'])}</b></div>
                    <div><small style="display:block; color:white;">Asist</small><b style="color:white;">{int(p['Ast'])}</b></div>
                    <div><small style="display:block; color:white;">Total</small><b style="color:{colors['sunlit-clay']}">{int(p['G+A'])}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
# -----------------
# TAB 3: JUGADOR VS JUGADOR (ELITE ANALYTICS) - VERSIÓN VISUAL COMPLETA
# -----------------
with tab3:
    st.markdown(f"""
        <div style="text-align:center; padding:20px; margin-bottom:20px; border-radius:15px; 
        background: linear-gradient(90deg, {colors['black-forest']}, {colors['pitch-dark']}); 
        border: 1px solid {colors['sunlit-clay']};">
            <h1 style='margin:0; font-size:2.5em; letter-spacing: 2px;'>COMPARATIVA DE ÉLITE</h1>
            <p style='color:{colors['sunlit-clay']};'>Análisis Estadístico Avanzado e Índice de Rendimiento</p>
        </div>
    """, unsafe_allow_html=True)

    # Selección de jugadores
    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        p1_name = st.selectbox("Seleccionar Jugador A", sorted(df["Player"].unique()), key="vs_p1_final")
    with c_sel2:
        p2_name = st.selectbox("Seleccionar Jugador B", sorted(df["Player"].unique()), index=1, key="vs_p2_final")

    p1 = df[df["Player"] == p1_name].iloc[0]
    p2 = df[df["Player"] == p2_name].iloc[0]

    # --- TARJETAS DE JUGADOR VISUALES ---
    t_col1, t_vs, t_col2 = st.columns([2, 0.5, 2])
    
    with t_col1:
        st.markdown(f"""
            <div style="background:{colors['olive-leaf']}; border: 2px solid {colors['sunlit-clay']}; border-radius: 20px; padding: 25px; text-align: center;">
                <img src="{p1['PlayerImg']}" style="width: 140px; height: 140px; border-radius: 50%; border: 4px solid white; object-fit: cover; margin-bottom: 10px;">
                <h2 style="margin:0; color:white;">{p1['Player']}</h2>
                <p style="color:{colors['cornsilk']}; opacity:0.9;">{p1['Squad']} | {p1['Pos']} | {int(p1['Age'])} años</p>
            </div>
        """, unsafe_allow_html=True)

    with t_vs:
        st.markdown("<h1 style='text-align:center; margin-top:80px; color:white; opacity:0.6;'>VS</h1>", unsafe_allow_html=True)

    with t_col2:
        st.markdown(f"""
            <div style="background:{colors['olive-leaf']}; border: 2px solid {colors['sunlit-clay']}; border-radius: 20px; padding: 25px; text-align: center;">
                <img src="{p2['PlayerImg']}" style="width: 140px; height: 140px; border-radius: 50%; border: 4px solid white; object-fit: cover; margin-bottom: 10px;">
                <h2 style="margin:0; color:white;">{p2['Player']}</h2>
                <p style="color:{colors['cornsilk']}; opacity:0.9;">{p2['Squad']} | {p2['Pos']} | {int(p2['Age'])} años</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INDICADORES CLAVE DE RENDIMIENTO (KPI) ---
    st.markdown("### Indicadores Clave de Rendimiento (KPI)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    metrics_kpi = [
        ("Impacto G+A / 90", "G+A", ""),
        ("Creatividad SCA / 90", "SCA90", "🪄"),
        ("Verticalidad Prog. Dist", "PrgDist", ""),
        ("Solidez Duelos Ganados", "Won", "")
    ]

    for col, (label, key, emoji) in zip([kpi1, kpi2, kpi3, kpi4], metrics_kpi):
        val1, val2 = p1[key], p2[key]
        diff = ((val1 - val2) / (val2 if val2 != 0 else 1)) * 100
        color_diff = "#2ecc71" if diff > 0 else "#e74c3c"
        
        col.markdown(f"""
            <div style="background:{colors['pitch-dark']}; padding:15px; border-radius:10px; border-bottom: 3px solid {colors['copperwood']}; text-align:center;">
                <div style="font-size:0.85em; color:{colors['sunlit-clay']}">{emoji} {label}</div>
                <div style="font-size:1.3em; font-weight:bold; color:white;">{val1:.2f} <span style="font-size:0.7em; color:grey;">vs</span> {val2:.2f}</div>
                <div style="font-size:0.75em; color:{color_diff}">{diff:+.1f}% diferencia</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()


    # --- RADAR Y DISTRIBUCIÓN DE POSESIÓN ---
    col_viz, col_data = st.columns([1.5, 1])

    with col_viz:
        st.markdown("<p style='text-align:center; font-weight:bold; color:white;'>Huella Estadística (Percentiles Liga)</p>", unsafe_allow_html=True)
        categories = ['SCA90', 'GCA90', 'PrgP', 'PrgC', 'Touches', 'Tkl+Int', 'Blocks', 'Won']
        
        fig_radar = go.Figure()
        for player, name, color in [(p1, p1_name, colors['copperwood']), (p2, p2_name, colors['sunlit-clay'])]:
            r_values = [(df[cat] < player[cat]).mean() * 100 for cat in categories]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values + [r_values[0]], theta=categories + [categories[0]], fill='toself',
                name=name, line=dict(color=color, width=3),
                fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.3)"
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#444", tickfont=dict(color="grey", size=8)),
                angularaxis=dict(gridcolor="#444", tickfont=dict(color="white", size=10)),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
            height=450, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_data:
        st.markdown(f"<p style='color:{colors['sunlit-clay']}; font-weight:bold;'>Distribución de Posesión</p>", unsafe_allow_html=True)
        tercios = ['Def 3rd_stats_possession', 'Mid 3rd_stats_possession', 'Att 3rd_stats_possession']
        labels_tercios = ['Defensivo', 'Central', 'Atacante']
        
        fig_tercios = go.Figure()
        fig_tercios.add_trace(go.Bar(y=labels_tercios, x=[p1[t] for t in tercios], name=p1_name, orientation='h', marker_color=colors['copperwood']))
        fig_tercios.add_trace(go.Bar(y=labels_tercios, x=[p2[t] for t in tercios], name=p2_name, orientation='h', marker_color=colors['sunlit-clay']))
        
        fig_tercios.update_layout(
            barmode='group', height=280, 
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), margin=dict(t=10, b=10),
            xaxis=dict(title="Toques por zona", gridcolor="#333"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_tercios, use_container_width=True)

        # --- INSIGHTS DE ESTILO (CONCLUSIÓN TÁCTICA) ---
        max_idx_p1 = np.argmax([p1[t] for t in tercios])
        max_idx_p2 = np.argmax([p2[t] for t in tercios])
        
        st.markdown(f"""
            <div style="background:rgba(255,254,224,0.05); padding:15px; border-radius:10px; border-left:4px solid {colors['sunlit-clay']}; margin-top:10px;">
                <p style="margin:0; font-size:0.9em; line-height:1.5; color:white;">
                    <b>Insights de Estilo:</b><br>
                    <b>{p1_name}:</b> {p1['PlayerStyle']} con gran presencia en zona <b>{labels_tercios[max_idx_p1]}</b>.<br>
                    <b>{p2_name}:</b> Perfil de {p2['PlayerStyle']} destacando en zona <b>{labels_tercios[max_idx_p2]}</b>.
                </p>
            </div>
        """, unsafe_allow_html=True)

# -----------------
# TAB 4: MARKET DISCOVERY (BUSCADOR)
# -----------------
with tab4:
    st.markdown(f"<h2 style='color:{colors['sunlit-clay']}'>Buscador de Talento Avanzado</h2>", unsafe_allow_html=True)
    
    with st.expander("Configurar Filtros de Scouting", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            min_minutos = st.slider("Mínimo Minutos", 0, int(df['Min'].max()), 500)
            pos_sel = st.multiselect("Posiciones", df['Pos'].unique(), default=df['Pos'].unique())
        with f_col2:
            min_sca = st.slider("Mín. Creación (SCA90)", 0.0, float(df['SCA90'].max()), 2.0)
            min_ga = st.slider("Mín. Goles + Asistencias", 0.0, float(df['G+A'].max()), 0.0)
        with f_col3:
            age_range = st.slider("Rango de Edad", int(df['Age'].min()), int(df['Age'].max()), (18, 30))

    # Lógica de filtrado
    scout_df = df[
        (df['Min'] >= min_minutos) & 
        (df['Pos'].isin(pos_sel)) & 
        (df['SCA90'] >= min_sca) & 
        (df['G+A'] >= min_ga) &
        (df['Age'].between(age_range[0], age_range[1]))
    ].copy()

    st.write(f"Jugadores que coinciden: **{len(scout_df)}**")
    st.dataframe(
        scout_df[['Player', 'Squad', 'Age', 'Pos', 'G+A', 'SCA90', 'PrgP', 'Won']].sort_values(by='SCA90', ascending=False),
        use_container_width=True
    )

# -----------------
# TAB 5: PERFORMANCE LABS (ANÁLISIS)
# -----------------
with tab5:
    st.markdown(f"<h2 style='color:{colors['sunlit-clay']}; text-align:center;'>Laboratorio de Rendimiento</h2>", unsafe_allow_html=True)
    
    # --- NUEVA SECCIÓN: LÍDERES DE ÉLITE (TOPS) ---
    st.markdown("<h3 style='text-align:center;'>Líderes de Élite del Momento</h3>", unsafe_allow_html=True)
    
    # Cálculos de Tops
    top_goleador = df.loc[df['Gls'].idxmax()]
    top_asistidor = df.loc[df['Ast'].idxmax()]
    df['DefScore'] = df['Tkl+Int'] + df['Clr']
    top_defensa = df.loc[df['DefScore'].idxmax()]
    df['MidScore'] = df['KP'] + df['1/3'] + df['PrgP']
    top_medio = df.loc[df['MidScore'].idxmax()]

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    tops = [
        ("Máximo Goleador", top_goleador, f"{top_goleador['Gls']} Goles", ""),
        ("Líder Asistencias", top_asistidor, f"{top_asistidor['Ast']} Asist.", ""),
        ("Muro Defensivo", top_defensa, f"{top_defensa['Tkl+Int']} Tkl+Int", ""),
        ("Gran Arquitecto", top_medio, f"{top_medio['PrgP']} Pases Prg", "🪄")
    ]

    for col, (label, player, stat, emoji) in zip([t_col1, t_col2, t_col3, t_col4], tops):
        col.markdown(f"""
            <div style="background:{colors['pitch-dark']}; padding:15px; border-radius:15px; 
            border: 1px solid {colors['sunlit-clay']}; text-align:center; height: 280px;">
                <p style="color:{colors['sunlit-clay']}; font-size:0.8em; font-weight:bold; margin-bottom:10px;">{emoji} {label}</p>
                <img src="{player['PlayerImg']}" style="width: 80px; height: 80px; border-radius: 50%; border: 2px solid white; object-fit: cover; margin-bottom:10px;">
                <h4 style="margin:0; font-size:1em;">{player['Player']}</h4>
                <p style="color:grey; font-size:0.7em; margin-bottom:5px;">{player['Squad']}</p>
                <div style="background:{colors['copperwood']}; border-radius:20px; padding:2px 10px; display:inline-block;">
                    <span style="font-weight:bold; font-size:0.9em;">{stat}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<h3 style='text-align:center;'>Rankings de Élite (Top 5)</h3>", unsafe_allow_html=True)

    # --- TODO ESTE BLOQUE AHORA ESTÁ DENTRO DE TAB 5 ---
    col_g, col_a, col_d, col_m = st.columns(4)

    with col_g:
        st.markdown("<h4 style='text-align:center;'>Goleadores</h4>", unsafe_allow_html=True)
        top_5_gls = df.nlargest(5, 'Gls')[['Player', 'Gls']]
        st.dataframe(top_5_gls, column_config={"Player": "Jugador", "Gls": st.column_config.ProgressColumn("Goles", min_value=0, max_value=int(df['Gls'].max()))}, hide_index=True, use_container_width=True)

    with col_a:
        st.markdown("<h4 style='text-align:center;'>Asistentes</h4>", unsafe_allow_html=True)
        top_5_ast = df.nlargest(5, 'Ast')[['Player', 'Ast']]
        st.dataframe(top_5_ast, column_config={"Player": "Jugador", "Ast": st.column_config.ProgressColumn("Asist.", min_value=0, max_value=int(df['Ast'].max()))}, hide_index=True, use_container_width=True)

    with col_d:
        st.markdown("<h4 style='text-align:center;'>Defensivo</h4>", unsafe_allow_html=True)
        top_5_def = df.nlargest(5, 'Tkl+Int')[['Player', 'Tkl+Int']]
        st.dataframe(top_5_def, column_config={"Player": "Jugador", "Tkl+Int": st.column_config.ProgressColumn("Tkl+Int", min_value=0, max_value=int(df['Tkl+Int'].max()))}, hide_index=True, use_container_width=True)

    with col_m:
        st.markdown("<h4 style='text-align:center;'>🪄 Arquitectos</h4>", unsafe_allow_html=True)
        top_5_mid = df.nlargest(5, 'PrgP')[['Player', 'PrgP']]
        st.dataframe(top_5_mid, column_config={"Player": "Jugador", "PrgP": st.column_config.ProgressColumn("Pases Prg", min_value=0, max_value=int(df['PrgP'].max()))}, hide_index=True, use_container_width=True)
        