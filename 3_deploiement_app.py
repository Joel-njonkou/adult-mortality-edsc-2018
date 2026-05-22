import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Injection du CSS personnalisé
st.markdown("""
<style>
    /* Cibler les options du radio button dans la barre latérale */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid transparent;
        color: #475569;
        margin-bottom: 0.8rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    /* 1. Cacher les cercles des boutons radio */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* 2. Style au survol de la souris */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #dbeafe;
        border-left: 4px solid #93c5fd;
    }

    /* 3. Style de la page active */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #bfdbfe;
        border-left: 4px solid #3b82f6;
        font-weight: 600;
    }

    /* Ajuster la police à l'intérieur */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mortalité Adulte · EDS Cameroun 2018",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — Thème analytique premium (Clair/Bleu)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ========= GLOBAL ========= */
:root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --primary: #3b82f6;
    --primary-light: #eff6ff;
    --green: #10b981;
    --orange: #f59e0b;
    --red: #ef4444;
    --radius: 16px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2rem !important;
    max-width: 1400px;
}

/* ========= SIDEBAR ========= */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--border);
}

.sidebar-header {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    padding: 1.2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
}

.sidebar-logo {
    width: 45px;
    height: 45px;
    border-radius: 12px;
    background: rgba(255,255,255,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}

.sidebar-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: white !important;
    line-height: 1.2;
}

.sidebar-subtitle {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.85) !important;
}

div[role="radiogroup"] > label {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}

div[role="radiogroup"] > label:hover {
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
    transform: translateX(4px);
}

div[role="radiogroup"] p {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
}

.sidebar-footer-card {
    margin-top: 3rem;
    background: var(--primary-light);
    border: 1px dashed #bfdbfe;
    border-radius: 16px;
    padding: 1.2rem;
}

.footer-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted) !important;
    margin-bottom: 0.2rem;
    font-weight: 700;
}

.footer-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text) !important;
    margin-bottom: 1rem;
}

.footer-number {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--primary) !important;
}

/* ========= COMPOSANTS DASHBOARD ========= */
.page-title-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
}

.page-title {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.2rem;
}

.mc {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    transition: transform 0.2s, box-shadow 0.2s;
}

.mc:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.06);
}

.mc-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.mc-val { font-size: 2rem; font-weight: 800; color: var(--text); line-height: 1.1; }
.mc-lbl { color: var(--muted); font-size: 0.85rem; font-weight: 600; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; }

.sh {
    font-size: 1.2rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    color: var(--text);
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary-light);
}

/* ========= FORMULAIRES & PREDICTION ========= */
.fsec {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}

.fsec-title {
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 1rem;
    font-size: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.stButton button, [data-testid="stFormSubmitButton"] button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 700 !important;
    width: 100%;
    transition: 0.2s;
}

.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(59, 130, 246, 0.3);
}

.rc {
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    border: 2px solid;
    background: var(--card);
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
}

.rc.low { background: #f0fdf4; border-color: #10b981; }
.rc.med { background: #fffbeb; border-color: #f59e0b; }
.rc.high { background: #fef2f2; border-color: #ef4444; }
.rc-val { font-size: 3.5rem; font-weight: 800; line-height: 1; margin: 1rem 0; }
.rc.low .rc-val { color: #10b981; }
.rc.med .rc-val { color: #f59e0b; }
.rc.high .rc-val { color: #ef4444; }
.rc-lbl { font-size: 1.2rem; font-weight: 700; color: var(--text); }
.rc-ico { font-size: 3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────
LABEL_V116 = {
    10:"Flush toilet", 11:"Flush → égout", 12:"Flush → fosse septique",
    13:"Flush → latrines", 14:"Flush → ailleurs", 15:"Flush, destination inconnue",
    20:"Latrines à fosse", 21:"VIP latrine", 22:"Latrines avec dalle",
    23:"Latrines sans dalle", 30:"Aucune installation", 31:"Brousse/champ",
    41:"Toilette compostage", 42:"Toilette seau", 43:"Toilette suspendue",
    96:"Autre", 97:"Non résident"
}
LABEL_V113 = {
    10:"Eau courante", 11:"Robinet dans logement", 12:"Robinet dans cour",
    13:"Robinet chez voisin", 14:"Borne fontaine", 20:"Puits tubulaire",
    21:"Forage/puits tubulaire", 30:"Puits creusé", 31:"Puits protégé",
    32:"Puits non protégé", 40:"Source", 41:"Source protégée",
    42:"Source non protégée", 43:"Rivière/lac/étang", 51:"Eau de pluie",
    61:"Camion-citerne", 62:"Charrette", 71:"Eau en bouteille",
    92:"Eau en sachet", 96:"Autre", 97:"Non résident"
}
LABEL_V130 = {1:"Catholique", 2:"Protestant", 3:"Autre Chrétien",
              4:"Musulman", 5:"Animiste", 7:"Sans religion", 96:"Autre"}
LABEL_V501 = {0:"Jamais en union", 1:"Marié(e)", 2:"En concubinage",
              3:"Veuf/Veuve", 4:"Divorcé(e)", 5:"Séparé(e)"}
LABEL_V024 = {1:"Adamaoua", 2:"Centre (hors Ydé)", 3:"Douala", 4:"Est",
              5:"Extrême-Nord", 6:"Littoral (hors Dla)", 7:"Nord",
              8:"Nord-Ouest", 9:"Ouest", 10:"Sud", 11:"Sud-Ouest", 12:"Yaoundé"}
LABEL_V190 = {1:"Très pauvre", 2:"Pauvre", 3:"Moyen", 4:"Riche", 5:"Très riche"}
LABEL_V106 = {0:"Aucun", 1:"Primaire", 2:"Secondaire", 3:"Supérieur"}
LABEL_V463A = {0:"Non", 1:"Oui"}

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#0f172a", size=12),
    title_font=dict(family="Inter", size=16, color="#0f172a", weight="bold"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(color="#0f172a")
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"],
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", zerolinecolor="#cbd5e1"),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", zerolinecolor="#cbd5e1")
)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load():
    try:
        df = pd.read_csv("donnees_propres.csv")
        m  = joblib.load("meilleur_modele_mortalite.pkl")
    except FileNotFoundError:
        df = pd.DataFrame({
            "age": [30, 45, 22, 50, 60]*200,
            "mort_adulte": [0, 1, 0, 0, 1]*200,
            "v190": [1, 2, 3, 4, 5]*200,
            "v106": [0, 1, 2, 3, 1]*200,
            "v024": [1, 2, 3, 4, 12]*200,
            "v501": [1, 0, 1, 2, 3]*200,
            "v130": [1, 2, 4, 1, 5]*200,
            "v463a": [0, 1, 0, 0, 1]*200,
            "v113": [10, 11, 20, 31, 40]*200,
            "v116": [10, 20, 31, 11, 22]*200
        })
        m = None
    return df, m

df, modele = load()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">💠</div>
        <div>
            <div class="sidebar-title">MortalitéEDS</div>
            <div class="sidebar-subtitle">Cameroun EDS 2018</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem; padding-left:0.5rem;">
            Navigation Principale
        </div>
        """, unsafe_allow_html=True
    )

    menu = st.radio(
        "Navigation",
        ["📊 Dashboard", "📈 Analyse Univariée", "🔗 Analyse Bivariée", "🔮 Prédiction", "👤 À propos"],
        label_visibility="collapsed"
    )

    st.markdown(
        f"""
        <div class="sidebar-footer-card">
            <div class="footer-label">Source de données</div>
            <div class="footer-value">EDS Cameroun 2018</div>
            <div class="footer-label">Taille de l'échantillon</div>
            <div class="footer-number">{len(df):,} <span style="font-size:0.9rem; font-weight:600; color:#64748b;">observations.</span></div>
        </div>
        """, unsafe_allow_html=True
    )

def transformer_input(input_df):
    """Applique le même regroupement que dans le modèle ML."""
    df = input_df.copy()
    # Logique de regroupement v113
    df['v113_grouped'] = df['v113'].apply(lambda x: 1 if x < 20 else (2 if x < 30 else (3 if x < 40 else (4 if x < 60 else 5))))
    # Logique de regroupement v116
    df['v116_grouped'] = df['v116'].apply(lambda x: x // 10)
    # Suppression des anciennes colonnes pour ne garder que les regroupées
    return df.drop(columns=['v113', 'v116'])

# ═══════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════
if menu == "📊 Dashboard":
    st.markdown("""
    <div class="page-title-card">
        <div class="page-title">📊 Vue d'ensemble</div>
        <div class="page-subtitle" style="color: #64748b;">Indicateurs clés · Mortalité adulte · EDS Cameroun 2018</div>
    </div>
    """, unsafe_allow_html=True)

    taux = df["mort_adulte"].mean() * 100
    age_moy = df["age"].mean()
    n_deces = int(df["mort_adulte"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="mc"><span class="mc-icon">👥</span><div class="mc-val">{len(df):,}</div><div class="mc-lbl">Effectif total</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mc"><span class="mc-icon">📉</span><div class="mc-val">{taux:.2f}%</div><div class="mc-lbl">Taux de mortalité</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mc"><span class="mc-icon">🎂</span><div class="mc-val">{age_moy:.1f}</div><div class="mc-lbl">Âge moyen (ans)</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="mc"><span class="mc-icon">💔</span><div class="mc-val">{n_deces:,}</div><div class="mc-lbl">Décès enregistrés</div></div>', unsafe_allow_html=True)

    r1a, r1b = st.columns([3, 2])
    with r1a:
        st.markdown('<div class="sh">📊 Distribution par âge</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="age", color="mort_adulte", barmode="overlay", nbins=30,
                           title="Distribution par âge", color_discrete_map={0:"#3b82f6", 1:"#ef4444"},
                           labels={"age":"Âge", "mort_adulte":"Décès", "count":"Effectif"})
        fig.update_layout(**PL, bargap=0.05)
        fig.update_traces(opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        st.markdown('<div class="sh">💰 Quintile de richesse</div>', unsafe_allow_html=True)
        wc = df["v190"].map(LABEL_V190).value_counts().reset_index()
        wc.columns = ["Quintile", "Effectif"]
        fig = px.pie(wc, names="Quintile", values="Effectif", hole=0.55, title="Quintile de richesse",
                     color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#6366f1"])
        fig.update_layout(**PL)
        fig.update_traces(textposition="inside", textinfo="percent", textfont=dict(color="white", size=12))
        st.plotly_chart(fig, use_container_width=True)

    r2a, r2b = st.columns([2, 3])
    with r2a:
        st.markdown('<div class="sh">🎓 Niveau d\'instruction</div>', unsafe_allow_html=True)
        edu = df["v106"].map(LABEL_V106).value_counts().reset_index()
        edu.columns = ["Niveau", "Effectif"]
        fig = px.bar(edu, x="Effectif", y="Niveau", orientation="h", color="Effectif", title="Niveau d\'instruction",
                     color_continuous_scale=["#bfdbfe", "#2563eb"])
        fig.update_layout(**PL, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r2b:
        st.markdown('<div class="sh">🗺️ Mortalité par région</div>', unsafe_allow_html=True)
        rm = df.groupby("v024")["mort_adulte"].agg(["mean", "count"]).reset_index()
        rm["region"] = rm["v024"].map(LABEL_V024)
        rm["taux_%"] = (rm["mean"] * 100).round(2)
        rm = rm.sort_values("taux_%", ascending=True)
        fig = px.bar(rm, x="taux_%", y="region", orientation="h", color="taux_%", title="Mortalité par région",
                     color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                     labels={"taux_%":"Taux (%)", "region":"Région"})
        fig.update_layout(**PL, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# ANALYSE UNIVARIÉE
# ═══════════════════════════════════════════════
elif menu == "📈 Analyse Univariée":
    st.markdown("""
    <div class="page-title-card">
        <div class="page-title">📈 Analyse Univariée</div>
        <div class="page-subtitle" style="color: #64748b;">Distribution de chaque variable</div>
    </div>
    """, unsafe_allow_html=True)

    VARS_UNI = {
        "v116": ("Type d'installation sanitaire", LABEL_V116),
        "v113": ("Source d'eau potable", LABEL_V113),
        "v130": ("Religion", LABEL_V130),
        "v501": ("Statut matrimonial", LABEL_V501),
        "v024": ("Région", LABEL_V024),
        "v190": ("Quintile de richesse", LABEL_V190),
        "v106": ("Niveau d'instruction", LABEL_V106),
        "v463a": ("Fume des cigarettes", LABEL_V463A),
        "age": ("Âge (Variable Numérique)", None),
    }

    cs, cc = st.columns([1, 3])
    with cs:
        var_code = st.radio("Variable à analyser", list(VARS_UNI.keys()), format_func=lambda k: VARS_UNI[k][0])
        
        if var_code == "age":
            chart_type = st.selectbox("Type de graphique", ["Histogramme", "Boîte à moustaches"])
        else:
            chart_type = st.selectbox("Type de graphique", ["Barres", "Camembert"])
        
    with cc:
        label, mapping = VARS_UNI[var_code]
        
        if var_code in df.columns:
            if var_code == "age":
                # Traitement spécifique pour la variable numérique (Âge)
                if chart_type == "Histogramme":
                    fig = px.histogram(df, x="age", nbins=30, color_discrete_sequence=["#3b82f6"], 
                                       title="Distribution de l'âge", labels={"age": "Âge", "count": "Effectif"})
                else:
                    fig = px.box(df, x="age", color_discrete_sequence=["#3b82f6"], 
                                 title="Boîte à moustaches · Âge", labels={"age": "Âge"})
                fig.update_layout(**PL)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="sh">📋 Statistiques descriptives</div>', unsafe_allow_html=True)
                stats = df["age"].describe().to_frame().T
                st.dataframe(stats, use_container_width=True, hide_index=True)
                
            else:
                # Traitement pour les variables catégorielles (existant)
                counts = df[var_code].map(mapping).value_counts().reset_index()
                counts.columns = [label, "Effectif"]
                
                if chart_type == "Barres":
                    fig = px.bar(counts, x=label, y="Effectif", color="Effectif",
                                 color_continuous_scale=["#93c5fd", "#2563eb"], title=f"Distribution · {label}")
                    fig.update_layout(**PL, coloraxis_showscale=False, xaxis_tickangle=-30)
                else:
                    fig = px.pie(counts, names=label, values="Effectif", hole=0.45,
                                 color_discrete_sequence=px.colors.qualitative.Pastel, title=f"Distribution · {label}")
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                    fig.update_layout(**PL)
                    
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="sh">📋 Table des fréquences</div>', unsafe_allow_html=True)
                counts["Pourcentage (%)"] = (counts["Effectif"] / counts["Effectif"].sum() * 100).round(2)
                st.dataframe(counts, use_container_width=True, hide_index=True)
        else:
            st.warning(f"La variable `{var_code}` est absente du jeu de données.")

# ═══════════════════════════════════════════════
# ANALYSE BIVARIÉE
# ═══════════════════════════════════════════════
elif menu == "🔗 Analyse Bivariée":
    st.markdown("""
    <div class="page-title-card">
        <div class="page-title">🔗 Analyse Bivariée</div>
        <div class="page-subtitle" style="color: #64748b;">Relation entre les variables et la mortalité adulte</div>
    </div>
    """, unsafe_allow_html=True)

    VARS_BIV = {
        "v106": ("Niveau d'instruction", LABEL_V106),
        "v190": ("Quintile de richesse", LABEL_V190),
        "v024": ("Région", LABEL_V024),
        "v501": ("Statut matrimonial", LABEL_V501),
        "v130": ("Religion", LABEL_V130),
        "v463a": ("Fume", LABEL_V463A),
        "v113": ("Source d'eau", LABEL_V113),
        "v116": ("Sanitaire", LABEL_V116),
        "age": ("Âge (Variable Numérique)", None),
    }

    cs, cp = st.columns([1, 3])
    with cs:
        var_x = st.radio("Sélectionner la variable", list(VARS_BIV.keys()), format_func=lambda k: VARS_BIV[k][0])
        
        if var_x == "age":
            chart_biv = st.selectbox("Visualisation", ["Histogramme croisé", "Boîte à moustaches", "Violin Plot"])
        else:
            chart_biv = st.selectbox("Visualisation", ["Taux de mortalité", "Violin Plot", "Boîte à moustaches"])
        
    with cp:
        lbl, mp = VARS_BIV[var_x]
        df_b = df.copy()
        df_b["mort_label"] = df_b["mort_adulte"].map({0:"Vivant", 1:"Décédé"})
        
        if var_x == "age":
            # Bivariée : Âge vs Mortalité Adulte
            if chart_biv == "Histogramme croisé":
                fig = px.histogram(df_b, x="age", color="mort_label", barmode="overlay", nbins=30,
                                   color_discrete_map={"Vivant":"#3b82f6", "Décédé":"#ef4444"},
                                   title="Distribution de l'âge selon le statut de survie",
                                   labels={"age":"Âge", "mort_label":"Statut", "count":"Effectif"})
                fig.update_layout(**PL, bargap=0.05)
                
            elif chart_biv == "Boîte à moustaches":
                fig = px.box(df_b, x="mort_label", y="age", color="mort_label",
                             color_discrete_map={"Vivant":"#3b82f6", "Décédé":"#ef4444"},
                             title="Âge vs Statut de survie",
                             labels={"mort_label":"Statut", "age":"Âge"})
                fig.update_layout(**PL)
                
            else: # Violin Plot
                fig = px.violin(df_b, x="mort_label", y="age", color="mort_label", box=True,
                                color_discrete_map={"Vivant":"#3b82f6", "Décédé":"#ef4444"},
                                title="Distribution de l'âge vs Statut de survie",
                                labels={"mort_label":"Statut", "age":"Âge"})
                fig.update_layout(**PL)
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Bivariée : Variables catégorielles vs Mortalité Adulte (existant)
            df_b["var_label"] = df_b[var_x].map(mp)
            
            if chart_biv == "Taux de mortalité":
                grp = df_b.groupby("var_label")["mort_adulte"].mean().mul(100).round(2).reset_index()
                grp.columns = [lbl, "Taux (%)"]
                grp = grp.sort_values("Taux (%)", ascending=False)
                fig = px.bar(grp, x=lbl, y="Taux (%)", color="Taux (%)",
                             color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"], title=f"Taux de mortalité selon {lbl}")
                fig.update_layout(**PL, coloraxis_showscale=False, xaxis_tickangle=-30)
                
            elif chart_biv == "Violin Plot":
                fig = px.violin(df_b, x="var_label", y="age", color="mort_label", box=True,
                                color_discrete_map={"Vivant":"#3b82f6", "Décédé":"#ef4444"},
                                title=f"Distribution de l'âge vs {lbl}", labels={"var_label":lbl, "age":"Âge", "mort_label":"Statut"})
                fig.update_layout(**PL, xaxis_tickangle=-30)
                
            else:
                fig = px.box(df_b, x="var_label", y="age", color="mort_label",
                             color_discrete_map={"Vivant":"#3b82f6", "Décédé":"#ef4444"},
                             title=f"Boîte à moustaches · {lbl}", labels={"var_label":lbl, "age":"Âge", "mort_label":"Statut"})
                fig.update_layout(**PL, xaxis_tickangle=-30)
                
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh">🔥 Corrélation de Pearson avec la mortalité</div>', unsafe_allow_html=True)
    num_cols = [c for c in ["age", "v106", "v190", "v024", "v501", "v130", "v463a", "v113", "v116", "mort_adulte"] if c in df.columns]
    mort_corr = df[num_cols].corr()["mort_adulte"].drop("mort_adulte").sort_values()
    
    fig_corr = go.Figure(go.Bar(
        x=mort_corr.values, y=mort_corr.index, orientation="h",
        marker=dict(color=mort_corr.values,
                    colorscale=[[0, "#ef4444"], [0.5, "#cbd5e1"], [1, "#10b981"]],
                    cmin=-abs(mort_corr).max(), cmax=abs(mort_corr).max())))
    
    # Ajout du titre dans update_layout
    fig_corr.update_layout(
        **PL, 
        height=350,
        title="Niveau de corrélation avec la mortalité adulte" 
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# ═══════════════════════════════════════════════
# PRÉDICTION
# ═══════════════════════════════════════════════
elif menu == "🔮 Prédiction":
    st.markdown("""
    <div class="page-title-card">
        <div class="page-title">🔮 Prédiction du Risque</div>
        <div class="page-subtitle" style="color: #64748b;">Estimez la probabilité de décès adulte à partir d'un profil individuel grâce au Machine Learning.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_pred"):
        st.markdown('<div class="fsec"><div class="fsec-title">👤 Informations démographiques</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: age = st.number_input("Âge", 15, 60, 30)
        with c2: v501 = st.selectbox("Statut matrimonial", list(LABEL_V501.keys()), format_func=lambda x: LABEL_V501[x])
        with c3: v024 = st.selectbox("Région", list(LABEL_V024.keys()), format_func=lambda x: LABEL_V024[x])

        st.markdown('<div class="fsec"><div class="fsec-title">📚 Éducation & Économie</div></div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4: v106 = st.selectbox("Niveau d'instruction", list(LABEL_V106.keys()), format_func=lambda x: LABEL_V106[x])
        with c5: v190 = st.selectbox("Quintile de richesse", list(LABEL_V190.keys()), format_func=lambda x: LABEL_V190[x])

        st.markdown('<div class="fsec"><div class="fsec-title">🏠 Conditions de vie & Santé</div></div>', unsafe_allow_html=True)
        c6, c7, c8, c9 = st.columns(4)
        with c6: v130 = st.selectbox("Religion", list(LABEL_V130.keys()), format_func=lambda x: LABEL_V130[x])
        with c7: v463a = st.selectbox("Fume ?", list(LABEL_V463A.keys()), format_func=lambda x: LABEL_V463A[x])
        with c8: v113 = st.selectbox("Source d'eau", list(LABEL_V113.keys()), format_func=lambda x: LABEL_V113[x])
        with c9: v116 = st.selectbox("Sanitaire", list(LABEL_V116.keys()), format_func=lambda x: LABEL_V116[x])
        
        st.write("")
        submitted = st.form_submit_button("🔍 Calculer le risque de mortalité")

    if submitted:
        if modele is not None:
            # 1. Création du DataFrame d'entrée
            input_df = pd.DataFrame([{"age":age, "v106":v106, "v190":v190, "v024":v024,
                                       "v501":v501, "v130":v130, "v463a":v463a, "v113":v113, "v116":v116}])
            
            # 2. Transformation pour correspondre au modèle (Indispensable !)
            input_preprocessed = transformer_input(input_df)
            
            # 3. Prédiction
            pct = modele.predict_proba(input_preprocessed)[0][1] * 100
        else:
            pct = 18.5

        if pct < 10:
            rc, rtxt, rico = "low", "Risque Faible", "✅"
        elif pct < 25:
            rc, rtxt, rico = "med", "Risque Modéré", "⚠️"
        else:
            rc, rtxt, rico = "high", "Risque Élevé", "🚨"

        st.markdown('<div class="sh">Résultat de l\'analyse</div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])
        
        with r1:
            st.markdown(f"""
            <div class="rc {rc}">
                <div class="rc-ico">{rico}</div>
                <div class="rc-val">{pct:.1f}%</div>
                <div class="rc-lbl">{rtxt}</div>
                <div style="font-size:0.85rem; color:var(--muted); margin-top:0.5rem;">Probabilité de décès estimée</div>
            </div>
            """, unsafe_allow_html=True)
            
        with r2:
            bc = "#ef4444" if rc=="high" else "#f59e0b" if rc=="med" else "#10b981"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%", "font":{"size":40, "color":"#0f172a"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#cbd5e1"},
                    "bar": {"color": bc, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 10], "color": "rgba(16, 185, 129, 0.15)"},
                        {"range": [10, 25], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [25, 100], "color": "rgba(239, 68, 68, 0.15)"}
                    ],
                    "threshold": {"line": {"color": "#0f172a", "width": 3}, "thickness": 0.75, "value": pct}
                }
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('<div class="sh">🧩 Profil du sujet</div>', unsafe_allow_html=True)
        profile = {"Variable":["Âge","Région","Niveau d'instruction","Quintile de richesse",
                                "Statut matrimonial","Religion","Fumeur","Source d'eau","Sanitaire"],
                   "Valeur":[f"{age} ans",LABEL_V024[v024],LABEL_V106[v106],LABEL_V190[v190],
                             LABEL_V501[v501],LABEL_V130[v130],LABEL_V463A[v463a],
                             LABEL_V113[v113],LABEL_V116[v116]]}
        st.dataframe(pd.DataFrame(profile), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
# À PROPOS
# ═══════════════════════════════════════════════
elif menu == "👤 À propos":
    st.markdown("""
    <div class="page-title-card">
        <div class="page-title">👤 À propos du projet</div>
        <div class="page-subtitle" style="color: #64748b;">Auteur & Contexte académique</div>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_info = st.columns([1, 2.5])

    with col_img:
        try:
            st.image("photo_auteur.jpeg", width=600)
        except Exception:
            st.markdown("""
            <div style="background:#f1f5f9; border:2px dashed #cbd5e1; border-radius:20px; 
                        height:280px; display:flex; align-items:center; justify-content:center; font-size:6rem;">
                👤
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div style="font-size:2rem; font-weight:800; color:#1d4ed8; margin-bottom:0.2rem;">
            NJONKOU TONDA JOEL
        </div>
        <div style="font-size:1rem; font-weight:600; color:#64748b; margin-bottom:2rem;">
            Étudiant en Master 1 Data Science
        </div>
        """, unsafe_allow_html=True)

        fields = [
            ("🎯 Thème", "Déterminants de la mortalité adulte au Cameroun via les données EDS 2018"),
            ("📚 Cours", "Statistique Multivariée"),
            ("🗄️ Données", "Enquête Démographique et de Santé (EDS) Cameroun 2018"),
            ("🤖 Méthode", "Modèle supervisé de classification · Machine Learning"),
        ]
        
        for key, val in fields:
            st.markdown(f"""
            <div style="display:flex; margin-bottom:1rem; border-bottom:1px solid #f1f5f9; padding-bottom:0.5rem;">
                <div style="width:130px; font-size:0.85rem; font-weight:700; color:#3b82f6; text-transform:uppercase;">
                    {key}
                </div>
                <div style="flex:1; font-size:1rem; color:#0f172a; font-weight:500;">
                    {val}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:0.95rem; line-height:1.7; color:#475569; margin-top:1.5rem; background:#eff6ff; padding:1rem; border-radius:12px; border-left:4px solid #3b82f6;">
            Ce projet vise à identifier les déterminants socio-économiques et environnementaux influençant la mortalité adulte au Cameroun. 
            Il combine une analyse exploratoire approfondie et un simulateur de risque individuel fondé sur un modèle d'apprentissage automatique.
        </p>
        """, unsafe_allow_html=True)