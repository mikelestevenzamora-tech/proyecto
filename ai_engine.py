import flet as ft
import pandas as pd
import numpy as np
import joblib
import os
import requests


df = pd.read_csv("final.csv")
df['Potencial'] = (df['SCA'] + df['Gls'] + df['Ast']) / df['Age']
df['Eficiencia'] = df['Gls'] / (df['SoT'] + 1)
df['DefImpact'] = df['Tkl'] + df['Int']

df['Score_Ataque'] = df['xG']*2 + df['Gls'] + df['Ast'] + df['KP'] + df['PPA'] + df['CrsPA']
df['Score_Defensa'] = df['Tkl'] + df['Int'] + df['Blocks'] + df['Clr'] + df['Def 3rd_stats_possession']
df['Score_Posesion'] = df[['Live_stats_possession','Touches','PrgDist']].mean(axis=1)


rf_valor_jugadores = joblib.load("rf_valor_jugadores.pkl")
scaler_valor_jugadores = joblib.load("scaler_valor_jugadores.pkl")

rf_valor_porteros = joblib.load("rf_valor_porteros.pkl")
scaler_valor_porteros = joblib.load("scaler_valor_porteros.pkl")

rf_goles = joblib.load("rf_goles.pkl")
rf_asistencias = joblib.load("rf_asistencias.pkl")
rf_paradas = joblib.load("rf_paradas.pkl")
scaler_paradas = joblib.load("scaler_paradas.pkl")

# Features
features_valor = [
    "Age","MP","Starts","Min","Touches","Carries","PrgDist",
    "PrgC","PrgP","PrgR","Succ%","Gls","Ast","G+A",
    "xG","xAG","KP","SCA","GCA","Tkl+Int","Blocks","FatigueIndex",
    "Sh","SoT","SoT%","Sh/90","SoT/90","G/Sh","G/SoT"
]

features_gk_paradas = ["MP","Min","Rec","GCA","Tkl+Int","FatigueIndex"]


def predecir_jugador(nombre):
    row = df[df['Player'].str.lower() == nombre.lower()]
    if row.empty:
        return "No se encontró ese jugador."
    
    row = row.iloc[0]
    
    if row['Pos'] == 'GK':
        # Portero
        X_gk = row[features_valor].values.reshape(1,-1)
        X_gk_scaled = scaler_valor_porteros.transform(X_gk)
        valor = rf_valor_porteros.predict(X_gk_scaled)[0]
        
        Xp = row[features_gk_paradas].values.reshape(1,-1)
        Xp_scaled = scaler_paradas.transform(Xp)
        paradas = rf_paradas.predict(Xp_scaled)[0]
        
        return {
            'Tipo': 'Portero',
            'Valor_M': round(valor,1),
            'Paradas': int(paradas),
            'FatigueIndex': round(row['FatigueIndex'],2)
        }
    else:
        # Campo
        X_field = row[features_valor].values.reshape(1,-1)
        X_field_scaled = scaler_valor_jugadores.transform(X_field)
        valor = rf_valor_jugadores.predict(X_field_scaled)[0]
        goles = rf_goles.predict(X_field_scaled)[0]
        asistencias = rf_asistencias.predict(X_field_scaled)[0]
        
        return {
            'Tipo': 'Campo',
            'Valor_M': round(valor,1),
            'Goles': round(goles,1),
            'Asistencias': round(asistencias,1),
            'FatigueIndex': round(row['FatigueIndex'],2)
        }


def predecir_partido(equipoA, equipoB):
    dfA = df[df['Squad'].str.lower() == equipoA.lower()]
    dfB = df[df['Squad'].str.lower() == equipoB.lower()]
    
    if dfA.empty or dfB.empty:
        return "No se encontraron datos para uno de los equipos."
    
    ataqueA = dfA['Score_Ataque'].sum()
    defensaA = dfA['Score_Defensa'].sum()
    posesionA = dfA['Score_Posesion'].mean()
    
    ataqueB = dfB['Score_Ataque'].sum()
    defensaB = dfB['Score_Defensa'].sum()
    posesionB = dfB['Score_Posesion'].mean()
    
    scoreA = (ataqueA / (defensaB + 1)) * 0.6 + posesionA*0.4
    scoreB = (ataqueB / (defensaA + 1)) * 0.6 + posesionB*0.4
    
    total = scoreA + scoreB
    probA = scoreA / total
    probB = scoreB / total
    
    analisis = f"Análisis: {equipoA} tiene ataque {round(ataqueA,1)} y posesión {round(posesionA,1)}; {equipoB} tiene defensa {round(defensaB,1)} y posesión {round(posesionB,1)}"
    
    return {
        'EquipoA': equipoA,
        'EquipoB': equipoB,
        'ProbabilidadA': round(probA*100,1),
        'ProbabilidadB': round(probB*100,1),
        'Analisis': analisis
    }


def main(page: ft.Page):
    # 🎨 Colores de la web
    colors = {
        "olive_leaf": "#606c38",
        "black_forest": "#283618",
        "cornsilk": "#fefae0",
        "sunlit_clay": "#dda15e",
        "copperwood": "#bc6c25",
        "pitch_dark": "#1a1a1a",
        "text_white": "#ffffff"
    }
    
    page.title = "Soccer Analytics IA"
    page.window_width = 950
    page.window_height = 800
    page.bgcolor = colors["black_forest"]
    
    # Area de chat
    chat_area = ft.Column(expand=True, spacing=10, scroll="auto")
    chat_box = ft.Container(
        content=chat_area, expand=True, 
        bgcolor=colors["olive_leaf"], padding=20, border_radius=12,
        border=ft.border.all(2, colors["sunlit_clay"])
    )
    
    input_box = ft.TextField(expand=True, label="Escribe tu consulta", text_style=ft.TextStyle(color=colors["text_white"], size=16), bgcolor=colors["pitch_dark"], border_radius=8)
    
    # Burbujas de chat
    def bubble(role, text):
        is_user = role == "user"
        bubble_color = colors["copperwood"] if is_user else colors["pitch_dark"]
        chat_area.controls.append(
            ft.Row(
                [ft.Container(
                    content=ft.Text(text, color=colors["text_white"], size=14),
                    bgcolor=bubble_color,
                    padding=14,
                    border_radius=14,
                    width=600
                )],
                alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
            )
        )
        page.update()
    
    # Enviar mensaje
    def send(_):
        if not input_box.value:
            return
        text = input_box.value
        bubble("user", text)
        
        text_lower = text.lower()
        if " vs " in text_lower or " contra " in text_lower:
            partes = text_lower.replace(" contra "," vs ").split(" vs ")
            if len(partes)==2:
                equipoA, equipoB = partes
                resp = predecir_partido(equipoA.strip(), equipoB.strip())
                if isinstance(resp,str):
                    bubble("assistant", resp)
                else:
                    bubble("assistant", f"{resp['EquipoA']} {resp['ProbabilidadA']}% vs {resp['EquipoB']} {resp['ProbabilidadB']}%\n{resp['Analisis']}")
        else:
            resp = predecir_jugador(text)
            if isinstance(resp,str):
                bubble("assistant", resp)
            else:
                salida = ""
                for k,v in resp.items():
                    if k != 'Tipo':
                        salida += f"{k}: {v}\n"
                bubble("assistant", salida)
        
        input_box.value=""
        page.update()
    
    # Título premium
    page.add(
        ft.Text("SOCCER ANALYTICS IA", size=32, weight="bold", color=colors["sunlit_clay"], text_align=ft.TextAlign.CENTER),
        chat_box,
        ft.Row([input_box, ft.ElevatedButton("ANALIZAR", on_click=send, bgcolor=colors["copperwood"], color=colors["text_white"])])
    )
    
    bubble("assistant", "Sistema listo. Puedes preguntar por jugadores o partidos (Ej: 'Courtois' o 'Real Madrid vs Barcelona').")

# ===============================
ft.app(target=main)
