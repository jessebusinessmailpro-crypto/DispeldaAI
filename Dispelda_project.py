import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scikit-learn import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. CONFIGURATION DU DASHBOARD (BRANDING DISPELDA) ---
st.set_page_config(
    page_title="Dispelda | AI Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour un look "Dispelda Corporate"
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    div[data-testid="stMetricValue"] {font-size: 24px; color: #0f172a;}
    h1 {color: #0369a1;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTEUR DE DONNÉES (DIGITAL TWIN SIMULATION) ---
@st.cache_data
def generate_telemetry_data(days=30):
    """Génère des données de télémétrie réalistes pour un Data Center."""
    hours = days * 24
    dates = pd.date_range(start="2025-01-01", periods=hours, freq="H")
    
    # Simulation: Charge Serveur (Cycle jour/nuit + Bruit)
    t = np.linspace(0, days * 2 * np.pi, hours)
    cpu_load = 60 + 20 * np.sin(t) + np.random.normal(0, 3, hours) # % de charge
    cpu_load = np.clip(cpu_load, 10, 100)
    
    # Simulation: Météo (Température & Humidité)
    temp_ext = 15 + 10 * np.sin(t - 2) + np.random.normal(0, 2, hours) # °C
    humidity = 60 + 20 * np.cos(t) + np.random.normal(0, 5, hours) # %
    humidity = np.clip(humidity, 20, 100)
    
    df = pd.DataFrame({
        'Timestamp': dates,
        'Temp_Ext': temp_ext,
        'Humidity': humidity,
        'IT_Load_MW': cpu_load / 10 # Charge en MégaWatts fictifs
    })
    
    # PHYSIQUE DU DATA CENTER (BASELINE)
    # Formule thermodynamique simplifiée pour la simulation
    df['Water_Usage_Baseline_L'] = (
        (df['IT_Load_MW'] * 1500) +       # 1500L/h par MW (Base)
        (df['Temp_Ext'] * 50) +           # Impact Température
        (df['Humidity'] * 10)             # Impact Humidité
    ) * np.random.uniform(0.95, 1.05, hours) 
    
    return df

# --- 3. MOTEUR IA DISPELDA (PREDICTIVE OPTIMIZER) ---
def train_and_predict(df):
    """Cœur de l'IA Dispelda : Entraînement et Optimisation."""
    
    # A. Feature Engineering
    X = df[['Temp_Ext', 'Humidity', 'IT_Load_MW']]
    y = df['Water_Usage_Baseline_L']
    
    # B. Entraînement (Scikit-Learn Random Forest)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # C. Prédiction de la consommation "Normale"
    predicted_baseline = model.predict(X)
    
    # D. APPLICATION DE LA STRATÉGIE DISPELDA
    efficiency_factor = []
    for i in range(len(df)):
        h = df.iloc[i]['Humidity']
        t = df.iloc[i]['Temp_Ext']
        
        # L'algorithme Dispelda détecte les opportunités d'évaporation optimale
        if h < 50 and t < 20:
            eff = 0.75  # -25% (Mode Eco Max)
        elif h < 70:
            eff = 0.85  # -15% (Mode Eco Standard)
        else:
            eff = 0.95  # -5% (Optimisation Mineure)
            
        efficiency_factor.append(eff)
        
    df['Water_Usage_Dispelda_L'] = predicted_baseline * np.array(efficiency_factor)
    return df

# --- 4. INTERFACE UTILISATEUR (FRONTEND) ---

# Sidebar de contrôle
with st.sidebar:
    st.header("⚙️ Paramètres Dispelda")
    days_sim = st.slider("Période d'analyse (Jours)", 7, 90, 30)
    st.info(f"Simulation sur {days_sim * 24} heures.")
    
    st.markdown("---")
    st.markdown("**État du Système:**")
    st.success("🟢 API Connectée")
    st.success("🟢 IA Dispelda v1.0: Active")
    
    st.markdown("---")
    st.caption("Dispelda Systems © 2025")

# Header Principal
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("Dispelda | Water Intelligence")
    st.markdown("### Optimisation prédictive des ressources hydriques pour Data Centers")

# Exécution de la simulation
with st.spinner('Le moteur Dispelda analyse vos données...'):
    raw_data = generate_telemetry_data(days=days_sim)
    processed_data = train_and_predict(raw_data)

# KPI SECTION (Le "Money Shot")
total_base = processed_data['Water_Usage_Baseline_L'].sum()
total_ai = processed_data['Water_Usage_Dispelda_L'].sum()
water_saved = total_base - total_ai
percent_saved = (water_saved / total_base) * 100
money_saved = (water_saved / 1000) * 3.5  # Prix moyen 3.5€/m3

st.markdown("### 📊 Rapport de Performance Dispelda")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Consommation Standard", f"{int(total_base/1000):,} m³", delta_color="off")
with kpi2:
    st.metric("Consommation Dispelda", f"{int(total_ai/1000):,} m³", delta=f"-{int(percent_saved)}%", delta_color="inverse")
with kpi3:
    st.metric("Eau Économisée", f"{int(water_saved/1000):,} m³", delta="Impact Positif")
with kpi4:
    st.metric("Gain Financier Est.", f"{int(money_saved):,} €", delta="Profit Net")

st.markdown("---")

# GRAPHIQUES AVANCÉS
tab1, tab2 = st.tabs(["📈 Analyse Temps Réel", "🧠 Facteurs IA"])

with tab1:
    st.subheader("Comparatif de Consommation (7 derniers jours)")
    subset = processed_data.tail(168)
    
    fig = go.Figure()
    
    # Zone grise (Ancien monde)
    fig.add_trace(go.Scatter(
        x=subset['Timestamp'], y=subset['Water_Usage_Baseline_L'],
        fill=None, mode='lines', line=dict(color='gray', width=1, dash='dot'),
        name='Baseline (Sans IA)'
    ))
    
    # Zone Bleue Dispelda
    fig.add_trace(go.Scatter(
        x=subset['Timestamp'], y=subset['Water_Usage_Dispelda_L'],
        fill='tonexty', 
        mode='lines', line=dict(color='#0ea5e9', width=3),
        name='Optimisation Dispelda'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Consommation Eau (Litres/Heure)",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("La Logique Dispelda")
    st.markdown("Notre algorithme croise la **Charge IT** et la **Météo** pour piloter le refroidissement.")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.caption("Corrélation Température Extérieure vs Charge IT")
        st.line_chart(subset.set_index('Timestamp')[['Temp_Ext', 'IT_Load_MW']])
        
    with chart_col2:
        st.caption("Taux d'Humidité (Facteur critique d'évaporation)")

        st.area_chart(subset.set_index('Timestamp')['Humidity'], color="#84cc16")
