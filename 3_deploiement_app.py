# ==============================================================================
#   APPLICATION STREAMLIT — MORTALITÉ ADULTE · EDS CAMEROUN 2018
#   Auteur : NJONKOU TONDA JOEL — Master 1 Data Science — Juin 2026
#   Version : 2.0 — Pipeline ML complet (modele_mortalite_complet.joblib)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE (DOIT ÊTRE EN PREMIER)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mortalité Adulte · EDS Cameroun 2018",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — THÈME PREMIUM
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:           #f8fafc;
    --card:         #ffffff;
    --border:       #e2e8f0;
    --text:         #0f172a;
    --muted:        #64748b;
    --primary:      #3b82f6;
    --primary-dk:   #1d4ed8;
    --primary-lt:   #eff6ff;
    --green:        #10b981;
    --orange:       #f59e0b;
    --red:          #ef4444;
    --purple:       #8b5cf6;
    --radius:       16px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2rem !important;
    max-width: 1500px;
}

/* ── SIDEBAR ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.sb-header {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    padding: 1.4rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(99,102,241,0.35);
    text-align: center;
}
.sb-logo  { font-size: 2.5rem; margin-bottom: 0.3rem; }
.sb-title { font-size: 1.1rem; font-weight: 800; color: white !important; }
.sb-sub   { font-size: 0.78rem; color: rgba(255,255,255,0.75) !important; margin-top: 2px; }

div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.4rem;
    transition: all 0.2s ease;
}
div[role="radiogroup"] > label:hover {
    background: rgba(59,130,246,0.2) !important;
    border-color: rgba(59,130,246,0.5) !important;
    transform: translateX(3px);
}
div[role="radiogroup"] p {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #e2e8f0 !important;
}

.sb-stats {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1.5rem;
}

/* ── COMPOSANTS CARTES ──────────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white !important;
    box-shadow: 0 10px 30px rgba(15,23,42,0.15);
}
.page-header h1 { font-size: 2rem; font-weight: 800; color: white !important; margin: 0; }
.page-header p  { color: rgba(255,255,255,0.7) !important; margin: 0.3rem 0 0; font-size: 1rem; }

.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,0,0,0.08); }
.kpi-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.kpi-val  { font-size: 2.2rem; font-weight: 800; color: var(--text); line-height: 1.1; }
.kpi-lbl  { font-size: 0.78rem; font-weight: 700; color: var(--muted); text-transform: uppercase;
            letter-spacing: 0.05em; margin-top: 0.4rem; }

.section-title {
    font-size: 1.15rem; font-weight: 700; color: var(--text);
    margin: 1.5rem 0 0.8rem;
    padding-left: 0.8rem;
    border-left: 3px solid var(--primary);
}

.chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    margin-bottom: 1rem;
}

/* ── FORMULAIRE PRÉDICTION ──────────────────────────────────────────────── */
.stButton button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dk)) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.9rem 2rem !important; font-weight: 700 !important; width: 100%;
    font-size: 1rem !important; transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.3) !important;
}
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59,130,246,0.4) !important; }

/* ── RÉSULTAT ───────────────────────────────────────────────────────────── */
.result-card {
    border-radius: 20px; padding: 2.5rem; text-align: center;
    border: 2px solid; background: var(--card); box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.result-card.low  { background: #f0fdf4; border-color: var(--green); }
.result-card.med  { background: #fffbeb; border-color: var(--orange); }
.result-card.high { background: #fef2f2; border-color: var(--red); }
.result-val { font-size: 4rem; font-weight: 800; line-height: 1; margin: 0.8rem 0; }
.result-card.low  .result-val { color: var(--green); }
.result-card.med  .result-val { color: var(--orange); }
.result-card.high .result-val { color: var(--red); }
.result-lbl { font-size: 1.2rem; font-weight: 700; color: var(--text); }

.model-badge {
    display: inline-block; background: var(--primary-lt); color: var(--primary);
    border: 1px solid #bfdbfe; border-radius: 999px;
    padding: 0.3rem 1rem; font-size: 0.85rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LABEL MAPS
# ──────────────────────────────────────────────────────────────────────────────
LABEL_V190 = {1: "Très pauvre", 2: "Pauvre", 3: "Moyen", 4: "Riche", 5: "Très riche"}
LABEL_V106 = {0: "Aucun", 1: "Primaire", 2: "Secondaire", 3: "Supérieur"}
LABEL_V024 = {1: "Adamaoua", 2: "Centre (hors Ydé)", 3: "Douala", 4: "Est",
              5: "Extrême-Nord", 6: "Littoral (hors Dla)", 7: "Nord",
              8: "Nord-Ouest", 9: "Ouest", 10: "Sud", 11: "Sud-Ouest", 12: "Yaoundé"}
LABEL_V025 = {1: "Urbain", 2: "Rural"}
LABEL_V501 = {0: "Jamais en union", 1: "Marié(e)", 2: "En concubinage",
              3: "Veuf/Veuve", 4: "Divorcé(e)", 5: "Séparé(e)"}

# Maps texte -> texte (pour le formulaire du nouveau modèle)
SEXE_OPTS     = ["Masculin", "Féminin"]
MILIEU_OPTS   = ["Urbain", "Rural"]
EDUC_OPTS     = ["Aucun", "Primaire", "Secondaire", "Supérieur"]
RICHESSE_OPTS = ["Plus pauvre", "Pauvre", "Moyen", "Riche", "Plus riche"]
MATRI_OPTS    = ["En union", "Célibataire", "Divorcé", "Veuf", "Jamais en union"]
REGION_OPTS   = ["Adamaoua", "Centre", "Est", "Extrême-Nord", "Littoral",
                 "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest", "Douala", "Yaoundé"]

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#0f172a", size=12),
    title_font=dict(family="Inter", size=15, color="#0f172a"),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"],
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
)

# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    for path in ["modele_mortalite_complet.joblib"]:
        if Path(path).exists():
            return joblib.load(path)
    return None

@st.cache_data
def charger_donnees():
    for path in ["data/processed/donnees_ml_long.csv", "donnees_ml_long.csv"]:
        if Path(path).exists():
            df = pd.read_csv(path)
            df = df.rename(columns={
                'lbl_region': 'region', 'v024': 'region',
                'lbl_milieu': 'milieu_residence', 'v025': 'milieu_residence',
                'lbl_instruction': 'niveau_education', 'v106': 'niveau_education',
                'lbl_richesse': 'indice_richesse', 'v190': 'indice_richesse',
                'lbl_matrimonial': 'etat_matrimonial', 'v501': 'etat_matrimonial',
                'lbl_sexe': 'sexe', 'sexe_fratrie': 'sexe',
                'age_ref': 'age'
            })
            if 'region' in df.columns and df['region'].dtype in ['int64', 'float64']:
                df['region'] = df['region'].map(LABEL_V024)
            if 'milieu_residence' in df.columns and df['milieu_residence'].dtype in ['int64', 'float64']:
                df['milieu_residence'] = df['milieu_residence'].map(LABEL_V025)
            if 'niveau_education' in df.columns and df['niveau_education'].dtype in ['int64', 'float64']:
                df['niveau_education'] = df['niveau_education'].map(LABEL_V106)
            if 'indice_richesse' in df.columns and df['indice_richesse'].dtype in ['int64', 'float64']:
                df['indice_richesse'] = df['indice_richesse'].map(LABEL_V190)
            return df
    # Données synthétiques de démonstration
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        "age": np.random.randint(15, 60, n),
        "sexe": np.random.choice(["Masculin", "Féminin"], n),
        "niveau_education": np.random.choice(list(LABEL_V106.values()), n),
        "indice_richesse": np.random.choice(list(LABEL_V190.values()), n),
        "region": np.random.choice(REGION_OPTS, n),
        "milieu_residence": np.random.choice(MILIEU_OPTS, n),
        "etat_matrimonial": np.random.choice(MATRI_OPTS, n),
        "mort_adulte": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })

artifacts = charger_modele()
df = charger_donnees()

pipeline         = artifacts['pipeline']         if artifacts else None
seuil_optimal    = artifacts['seuil_optimal']    if artifacts else 0.5
age_moyen_train  = artifacts['age_moyen_train']  if artifacts else 31.0

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <div class="sb-logo">💠</div>
        <div class="sb-title">MortalitéEDS</div>
        <div class="sb-sub">EDS Cameroun 2018 · Master 1 Data Science</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        [
         "🤖 Résultats ML",
         "🔮 Prédiction",
         "👤 À propos"],
        label_visibility="collapsed"
    )

# ──────────────────────────────────────────────────────────────────────────────
# PAGE : RÉSULTATS ML
# ──────────────────────────────────────────────────────────────────────────────
elif menu == "🤖 Résultats ML":
    st.markdown("""
    <div class="page-header">
        <h1>🤖 Résultats Machine Learning</h1>
        <p>Comparaison de 6 algorithmes · Cost-sensitive Learning · Validation croisée stratifiée (5-fold)</p>
    </div>
    """, unsafe_allow_html=True)

    if artifacts:
        st.markdown(f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                    padding:1rem 1.5rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:12px;">
            <span style="font-size:2rem;">🏆</span>
            <div>
                <div style="font-size:0.8rem;font-weight:700;color:#64748b;text-transform:uppercase;">
                    Modèle déployé
                </div>
                <div style="font-size:1.3rem;font-weight:800;color:#1d4ed8;">
                    Meilleur modèle sélectionné · Seuil optimal = {seuil_optimal:.3f}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠ Fichier `modele_mortalite_complet.joblib` introuvable. Placez-le dans le dossier de l'application.")

    # Affichage des résultats CSV si disponible
    for csv_path in ["outputs/tables/ML_05_comparaison_modeles.csv",
                     "resultats_modeles.csv"]:
        if Path(csv_path).exists():
            df_ml = pd.read_csv(csv_path)
            st.markdown('<div class="section-title">📊 Comparaison des modèles (5-Fold CV)</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_ml, use_container_width=True)
            break
    else:
        st.info("""
        ℹ Les résultats détaillés de la comparaison des modèles ne sont pas disponibles.
        Exécutez le notebook `2_Code_ML.ipynb` pour les générer dans `outputs/tables/`.
        """)

    # Résumé méthodologique
    st.markdown('<div class="section-title">📋 Résumé de la pipeline ML</div>', unsafe_allow_html=True)
    methodo = pd.DataFrame({
        "Étape": ["Feature Engineering", "Division des données", "Préprocesseur",
                  "Modèles testés", "Validation croisée", "Critère de sélection",
                  "Gestion déséquilibre", "Seuil de décision"],
        "Détail": [
            "Âge centré (age_centre), âge² (age_au_carre), interaction âge×sexe",
            "80% train / 20% test · Stratification sur mort_adulte",
            "StandardScaler (numérique) + OneHotEncoder drop='first' (catégoriel)",
            "Régression Logistique, Random Forest, Extra Trees, Hist GBM, XGBoost, LightGBM",
            "StratifiedKFold (5 folds, random_state=42)",
            "ROC-AUC (maximisation) | F1-Score (indicatif)",
            "class_weight='balanced' ou scale_pos_weight (ratio ~32:1)",
            f"Seuil F1-optimal sur courbe ROC = {seuil_optimal:.3f}"
        ]
    })
    st.dataframe(methodo, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE : PRÉDICTION
# ──────────────────────────────────────────────────────────────────────────────
elif menu == "🔮 Prédiction":
    st.markdown("""
    <div class="page-header">
        <h1>🔮 Prédiction du Risque de Mortalité Adulte</h1>
        <p>Estimez la probabilité de décès adulte à partir d'un profil socio-démographique individuel</p>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts:
        st.error("⚠️ Modèle non chargé. Veuillez placer `modele_mortalite_complet.joblib` dans le dossier de l'application.")
        st.stop()

    st.markdown(f"""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                padding:.8rem 1.2rem;margin-bottom:1rem;font-size:.9rem;color:#166534;">
        <strong>ℹ️ Note scientifique :</strong> Les probabilités sont issues d'un pipeline de 
        <span class="model-badge">Cost-Sensitive Learning</span> entraîné sur 
        l'EDS Cameroun 2018 (méthode CSS). Seuil de décision optimal : 
        <strong>{seuil_optimal:.3f}</strong> (maximisation du F1-Score).
    </div>
    """, unsafe_allow_html=True)

    # 1. OPTIONS ALIGNÉES STRICTEMENT SUR LES LABELS DE L'ENTRAÎNEMENT EDS 2018
    SEXE_OPTS_EDS     = ["Masculin", "Féminin"]
    MILIEU_OPTS_EDS   = ["Urbain", "Rural"]
    EDUC_OPTS_EDS     = ["Aucun", "Primaire", "Secondaire", "Supérieur"]
    RICHESSE_OPTS_EDS = ["Très pauvre", "Pauvre", "Moyen", "Riche", "Très riche"]
    MATRI_OPTS_EDS    = ["Jamais en union", "Marié(e)", "En concubinage", "Veuf/Veuve", "Divorcé(e)", "Séparé(e)"]
    REGION_OPTS_EDS   = ["Adamaoua", "Centre (hors Ydé)", "Douala", "Est", "Extrême-Nord", 
                         "Littoral (hors Dla)", "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest", "Yaoundé"]

    # Formulaire de saisie utilisateur
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Âge de l'individu (années)", min_value=15, max_value=59, value=30, step=1)
        sexe = st.selectbox("Sexe", SEXE_OPTS_EDS)
        milieu_residence = st.selectbox("Milieu de résidence", MILIEU_OPTS_EDS)

    with col2:
        niveau_education = st.selectbox("Niveau d'instruction", EDUC_OPTS_EDS)
        indice_richesse  = st.selectbox("Indice de richesse", RICHESSE_OPTS_EDS)
        etat_matrimonial = st.selectbox("État matrimonial", MATRI_OPTS_EDS)

    with col3:
        region = st.selectbox("Région de résidence", REGION_OPTS_EDS)

    st.markdown("---")
    if st.button("🚀 Calculer le risque de mortalité", use_container_width=True):

        # 2. FEATURE ENGINEERING NUMÉRIQUE
        age_centre          = age - age_moyen_train
        age_au_carre        = age_centre ** 2
        is_masculin         = 1 if sexe == "Masculin" else 0
        interaction_age_sexe = age_centre * is_masculin

        # ++++ CORRECTION : CRÉATION DES DICTIONNAIRES INVERSES ++++
        # Cela permet de retrouver le code (ex: "1") à partir du label (ex: "Très pauvre")
        REV_LABEL_V190 = {v: str(k) for k, v in LABEL_V190.items()}
        REV_LABEL_V106 = {v: str(k) for k, v in LABEL_V106.items()}
        REV_LABEL_V024 = {v: str(k) for k, v in LABEL_V024.items()}
        REV_LABEL_V025 = {v: str(k) for k, v in LABEL_V025.items()}
        REV_LABEL_V501 = {v: str(k) for k, v in LABEL_V501.items()}
        REV_SEXE       = {"Masculin": "1", "Féminin": "2"}
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 3. CONSTRUCTION DE L'ÉCOSYSTÈME DE COLONNES UNIVERSEL (Gère courts, longs et bruts)
        dictionnaire_universel = {
            # Formats de variables courts
            'age': age, 'age_ref': age,
            'sexe': REV_SEXE[sexe], 
            'milieu_residence': REV_LABEL_V025[milieu_residence],
            'niveau_education': REV_LABEL_V106[niveau_education], 
            'indice_richesse': REV_LABEL_V190[indice_richesse],
            'etat_matrimonial': REV_LABEL_V501[etat_matrimonial], 
            'region': REV_LABEL_V024[region],
            
            # Formats de variables longs (lbl_) issus du CSV original
            'lbl_sexe': REV_SEXE[sexe], 
            'lbl_milieu': REV_LABEL_V025[milieu_residence],
            'lbl_instruction': REV_LABEL_V106[niveau_education], 
            'lbl_richesse': REV_LABEL_V190[indice_richesse],
            'lbl_matrimonial': REV_LABEL_V501[etat_matrimonial], 
            'lbl_region': REV_LABEL_V024[region],
            
            # Indicateurs calculés
            'age_centre': age_centre,
            'age_au_carre': age_au_carre,
            'interaction_age_sexe': interaction_age_sexe
        }
        
        donnees_saisie = pd.DataFrame([dictionnaire_universel])

        # 4. ALIGNEMENT ET RÉORDONNANCEMENT DYNAMIQUE VIA LES MÉTADONNÉES DU MODÈLE
        colonnes_ordonnees = None
        
        # Vérification standard des variables d'entrée mémorisées par Scikit-Learn
        if hasattr(pipeline, 'feature_names_in_'):
            colonnes_ordonnees = list(pipeline.feature_names_in_)
        
        # Alternative : Extraction via la configuration interne du ColumnTransformer
        elif hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            prep = pipeline.named_steps['preprocessor']
            if hasattr(prep, 'transformers_'):
                list_cols = []
                for name, trans, cols in prep.transformers_:
                    if name != 'remainder':
                        list_cols.extend(cols)
                if list_cols:
                    colonnes_ordonnees = list_cols

        # Restructuration stricte du DataFrame si les métadonnées de l'entraînement existent
        if colonnes_ordonnees:
            for col in colonnes_ordonnees:
                if col not in donnees_saisie.columns:
                    donnees_saisie[col] = 0  # Sécurité structurelle
            donnees_saisie = donnees_saisie[colonnes_ordonnees]

        # 5. CONVERSION DES TYPES POUR LE PIPELINE
        for col in donnees_saisie.columns:
            if donnees_saisie[col].dtype == object or str(donnees_saisie[col].dtype) == 'category':
                donnees_saisie[col] = donnees_saisie[col].astype(str)

        # 6. ENCAPSULATION DE SÉCURITÉ ET EXÉCUTION
        try:
            probabilite_deces = pipeline.predict_proba(donnees_saisie)[0, 1]
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.stop()

        # 7. STRATIFICATION DES NIVEAUX DE RISQUE
        pct = probabilite_deces * 100
        seuil_bas = 0.450
        
        if probabilite_deces < seuil_bas:
            cls = "low"
            label_risque = "Risque Faible"
            ico = "✅"
        elif seuil_bas <= probabilite_deces < seuil_optimal:
            cls = "med"
            label_risque = "Risque Modéré"
            ico = "⚠️"
        else:
            cls = "high"
            label_risque = "Risque Élevé"
            ico = "🚨"

        # 8. RESTITUTION VISUELLE DES INDICES DE RISQUE
        st.write("### 📊 Résultat de l'analyse prédictive")
        r1, r2 = st.columns([1, 2])
        
        with r1:
            st.markdown(f"""
            <div class="result-card {cls}">
                <div style="font-size: 2.5rem;">{ico}</div>
                <div class="result-val">{probabilite_deces:.2%}</div>
                <div class="result-lbl">{label_risque}</div>
                <div style="font-size: 0.8rem; color: var(--muted); margin-top: 1rem;">
                    Seuil d'alerte critique : {seuil_optimal:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with r2:
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border: 1px solid var(--border); border-radius: var(--radius); height: 100%;">
                <h4 style="margin-top: 0; color: var(--text); font-weight:700;">📋 Évaluation Multivariée du Profil</h4>
                <p style="font-size: 0.95rem; color: #334155; margin-bottom: 0.8rem;">
                    L'analyse du profil pour cet individu de <strong>{age} ans</strong> ({sexe}) résidant en milieu 
                    <strong>{milieu_residence}</strong> (Région : <strong>{region}</strong>) indique :
                </p>
                <ul style="font-size: 0.9rem; color: #475569; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Composante Biologique :</strong> L'évolution du risque intègre l'âge centré et la dynamique d'interaction liée au genre masculin/féminin.</li>
                    <li><strong>Pondération Structurelle :</strong> Le niveau d'instruction (<em>{niveau_education}</em>) associé au niveau de richesse (<em>{indice_richesse}</em>) est maintenant correctement mappé et applique son poids factoriel complet.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # 9. MODULE D'INSPECTION (Débuggage visible en cas de doute)
        with st.expander("🔍 Inspecteur de Alignement ML (Debug App)") :
            st.write("**Colonnes actuellement transmises au modèle :**", list(donnees_saisie.columns))
            st.write("**Valeurs associées :**")
            st.dataframe(donnees_saisie)
            if colonnes_ordonnees:
                st.write("**Ordre strict détecté depuis le fichier d'entraînement :**", colonnes_ordonnees)
# ──────────────────────────────────────────────────────────────────────────────
# PAGE : À PROPOS
# ──────────────────────────────────────────────────────────────────────────────
elif menu == "👤 À propos":
    st.markdown("""
    <div class="page-header">
        <h1>👤 À propos du projet</h1>
        <p>Auteur · Contexte académique · Méthodologie complète</p>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_info = st.columns([1, 2.5])
    with col_img:
        if Path("photo_auteur.jpeg").exists():
            st.image("photo_auteur.jpeg", width=250)
        else:
            st.markdown("""
            <div style="background:#f1f5f9;border:2px dashed #cbd5e1;border-radius:20px;
                        height:250px;display:flex;align-items:center;justify-content:center;
                        font-size:5rem;">👤</div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div style="font-size:2rem;font-weight:800;color:#1d4ed8;margin-bottom:.2rem;">
            NJONKOU TONDA JOEL
        </div>
        <div style="font-size:1rem;font-weight:600;color:#64748b;margin-bottom:1.5rem;">
            Étudiant en Master 1 · Data Science & Statistiques
        </div>
        """, unsafe_allow_html=True)

        for key, val in [
            ("🎯 Thème",         "Déterminants de la mortalité adulte au Cameroun — EDS 2018"),
            ("📚 Cours",          "Statistique Multivariée & Machine Learning"),
            ("🗄️ Données",       "Enquête Démographique et de Santé (EDS) Cameroun 2018"),
            ("🔬 Méthode Stats", "Plan de sondage complexe · Régression logistique · OR · Test Rao-Scott"),
            ("🤖 Méthode ML",    "6 modèles · Cost-sensitive Learning · SHAP · Seuil optimal F1"),
            ("📅 Date",           "Juin 2026"),
        ]:
            st.markdown(f"""
            <div style="display:flex;margin-bottom:.8rem;border-bottom:1px solid #f1f5f9;padding-bottom:.5rem;">
                <div style="width:170px;font-size:.82rem;font-weight:700;color:#3b82f6;text-transform:uppercase;">{key}</div>
                <div style="font-size:.9rem;color:#0f172a;">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📖 Pipeline méthodologique complète</div>', unsafe_allow_html=True)
    st.markdown("""
    | Étape | Détail |
    |-------|--------|
    | **Données sources** | CMIR71FL.sav — EDS Cameroun 2018 (femmes 15-49 ans) |
    | **Variable cible** | Méthode CSS (Obermeyer et al.) : vivant=0 ∩ âge_décès∈[15,59] ∩ délai≤7 ans |
    | **Format long** | Restructuration wide→long par fratrie (pivot_longer sur mm1…mm20) |
    | **Calculs CMC** | Âge au décès = (deces_cmc − naiss_cmc) / 12 |
    | **Plan de sondage** | PSU (v021), Strates (v022), Poids (v005/1e6) |
    | **Tests bivariés** | Chi² de Rao-Scott (catégorielles), seuil p < 0.20 |
    | **Régression stats** | svyglm (quasibinomial) — Odds Ratios ajustés + Forest Plot |
    | **Feature Engineering** | Âge centré, âge², interaction âge×sexe |
    | **Préprocesseur** | StandardScaler + OneHotEncoder (drop='first') |
    | **Modèles ML** | Logistic Reg., Random Forest, Extra Trees, Hist GBM, XGBoost, LightGBM |
    | **Validation** | StratifiedKFold (5 folds) — critère AUC |
    | **Gestion déséquilibre** | class_weight='balanced' / scale_pos_weight (ratio ~32:1) |
    | **Seuil optimal** | Maximisation F1-Score sur courbe ROC du jeu de test |
    | **Explicabilité** | SHAP — LinearExplainer sur le meilleur modèle |
    """)
