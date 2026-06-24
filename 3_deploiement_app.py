
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# 1. Configuration de la page
st.set_page_config(
    page_title="Évaluation Risque Mortalité",
    page_icon="🏥",
    layout="wide"
)

# 2. Initialisation de la mémoire (Session State) pour l'historique
if 'historique_pred' not in st.session_state:
    st.session_state['historique_pred'] = []

# 3. Chargement du modèle mis en cache
@st.cache_resource
@st.cache_resource
def charger_modele():
    chemin_modele = "pipeline_mortalite_gb.joblib"
    if not os.path.exists(chemin_modele):
        st.error(f"❌ Le fichier du modèle est introuvable au chemin : {chemin_modele}.")
        return None
    try:
        return joblib.load(chemin_modele)
    except Exception as e:
        st.error("🚨 **Alerte Débug : Voici le module qui bloque l'application :**")
        st.exception(e)  # Force l'affichage du vrai message d'erreur non censuré
        return None

artifacts = charger_modele()

if artifacts is not None:
    pipeline = artifacts['pipeline']
    seuil_optimal = artifacts['seuil_optimal']
    
    # En-tête
    st.title("🏥 Système d'Évaluation du Risque de Mortalité Adulte")
    st.markdown("""
    Cette interface utilise le modèle **Gradient Boosting** optimisé sur les données de l'EDS.
    La jauge se remplit dynamiquement et change de couleur selon le niveau de risque calculé.
    """)
    st.markdown("---")

    # Formulaire de saisie sécurisé
    st.subheader("📋 Formulaire de Saisie Sécurisé")
    with st.form("formulaire_prediction"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔢 Variables Quantitatives")
            age_saisi = st.number_input(
                "Âge de l'individu (15 à 60 ans)", 
                min_value=15, max_value=60, value=30, step=1
            )
            taille_fratrie_saisie = st.number_input(
                "Taille de la fratrie (0 à 20 frères/sœurs)", 
                min_value=0, max_value=20, value=3, step=1
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌍 Origine Géographique")
            region_saisie = st.selectbox("Région de résidence", options=[
                "Adamaoua", "Centre", "Douala", "Est", "Extrême-Nord", 
                "Littoral", "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest", "Yaoundé"
            ])

        with col2:
            st.markdown("### 🎭 Variables Qualitatives")
            sexe_saisi = st.selectbox("Sexe de l'individu", options=["Masculin", "Féminin"])
            instruction_saisie = st.selectbox("Niveau d'instruction le plus élevé", options=[
                "Sans instruction", "Primaire", "Secondaire", "Supérieur"
            ])
            richesse_saisi = st.selectbox("Indice de richesse du ménage", options=[
                "Plus pauvre", "Pauvre", "Moyen", "Riche", "Plus riche"
            ])
            toilettes_saisie = st.selectbox("Type de commodités sanitaires (Toilettes)", options=[
                "Latrines améliorées", "Latrines traditionnelles", "Pas de toilettes/Nature", "Chasse d'eau"
            ])
            
        st.markdown("<br>", unsafe_allow_html=True)
        bouton_predire = st.form_submit_button("⚡ Valider et Analyser le profil")

    # Logique de traitement
    if bouton_predire:
        saisie_valide = True
        if not (15 <= age_saisi <= 60) or not (0 <= taille_fratrie_saisie <= 20):
            st.error("🚨 **Erreur de validation :** Veuillez vérifier les valeurs numériques.")
            saisie_valide = False

        if saisie_valide:
            donnees_saisie = pd.DataFrame({
                'age_ref': [int(age_saisi)],
                'taille_fratrie': [int(taille_fratrie_saisie)],
                'Sexe_Germain': [sexe_saisi],
                'Région': [region_saisie],
                'Instruction': [instruction_saisie],
                'Richesse_Cat': [richesse_saisi],
                'Toilettes': [toilettes_saisie]
            })

            try:
                # 1. Calcul de la probabilité brute
                probabilite_deces = pipeline.predict_proba(donnees_saisie)[0, 1]
                est_a_risque = probabilite_deces >= seuil_optimal
                
                # Convertis en pourcentages pour l'affichage Plotly
                val_pourcent = probabilite_deces * 100
                seuil_pourcent = seuil_optimal * 100
                
                # 2. Attribution DYNAMIQUE de la couleur du remplissage de la jauge
                if val_pourcent < (seuil_pourcent * 0.75):
                    couleur_barre = "#198754"  # Vert
                elif val_pourcent < seuil_pourcent:
                    couleur_barre = "#fd7e14"  # Orange
                else:
                    couleur_barre = "#dc3545"  # Rouge

                # 3. Enregistrement dans l'historique
                nouvelle_entree = {
                    "Heure": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Âge": int(age_saisi),
                    "Taille fratrie": int(taille_fratrie_saisie),
                    "Sexe": sexe_saisi,
                    "Région": region_saisie,
                    "Instruction": instruction_saisie,
                    "Richesse": richesse_saisi,
                    "Risque (%)": f"{probabilite_deces:.2%}",
                    "Statut": "🚨 ÉLEVÉ" if est_a_risque else "✅ FAIBLE"
                }
                st.session_state['historique_pred'].insert(0, nouvelle_entree)
                st.session_state['historique_pred'] = st.session_state['historique_pred'][:5]

                # 4. Affichage du Bloc Diagnostic
                st.markdown("---")
                st.subheader("📊 Diagnostic du Modèle Épidémiologique")
                
                c_jauge, c_statut = st.columns([1, 1])
                
                with c_jauge:
                    # Rendu de la jauge (Correction de la propriété 'weight')
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=val_pourcent,
                        number={'suffix': "%", 'valueformat': ".2f", 'font': {'color': couleur_barre, 'weight': 'bold'}},
                        title={'text': "Niveau de Risque Individuel", 'font': {'size': 18, 'weight': 'bold'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                            'bar': {'color': couleur_barre, 'thickness': 0.35},
                            'bgcolor': "#f0f2f6",
                            'borderwidth': 1,
                            'bordercolor': "silver",
                            'steps': [
                                {'range': [0, seuil_pourcent * 0.75], 'color': "#e8f5e9"},
                                {'range': [seuil_pourcent * 0.75, seuil_pourcent], 'color': "#fff3e0"},
                                {'range': [seuil_pourcent, 100], 'color': "#ffebee"}
                            ],
                            'threshold': {
                                'line': {'color': "#721c24", 'width': 4},
                                'thickness': 0.8,
                                'value': seuil_pourcent
                            }
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with c_statut:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if est_a_risque:
                        st.error("🚨 **Alerte : Profil identifié À RISQUE ÉLEVÉ**")
                        st.markdown(f"""
                        * **Seuil d'alerte franchi :** **{probabilite_deces:.2%}** (Le seuil critique calculé est de {seuil_optimal:.2%}).
                        * **Interprétation :** Ce profil requiert une attention sanitaire ciblée.
                        """)
                    else:
                        st.success("✅ **Statut : Profil à FAIBLE RISQUE**")
                        st.markdown(f"""
                        * **Situation stable :** **{probabilite_deces:.2%}** (En dessous du seuil critique de {seuil_optimal:.2%}).
                        * **Interprétation :** Les facteurs protecteurs compensent les variables de vulnérabilité.
                        """)
                        
            except Exception as e:
                st.error(f"💥 Erreur structurelle lors du traitement des données : {str(e)}")

    # 5. Affichage permanent de l'historique
    st.markdown("---")
    st.subheader("📜 Historique des 5 dernières analyses (Session en cours)")
    
    if st.session_state['historique_pred']:
        df_hist = pd.DataFrame(st.session_state['historique_pred'])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.info("💡 L'historique est vide. Remplissez le formulaire et valisez pour voir apparaître vos résultats.")
