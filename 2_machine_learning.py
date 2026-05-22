import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Import des modèles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def transformer_donnees(df):
    """Applique le regroupement des modalités pour stabiliser le modèle."""
    df = df.copy()
    
    # Regroupement v113 (Source d'eau)
    df['v113_grouped'] = df['v113'].apply(lambda x: 1 if x < 20 else (2 if x < 30 else (3 if x < 40 else (4 if x < 60 else 5))))
    
    # Regroupement v116 (Sanitaire)
    df['v116_grouped'] = df['v116'].apply(lambda x: x // 10)
    
    return df

def entrainer_modeles():
    df = pd.read_csv("donnees_propres.csv")
    
    # Application de la transformation
    df = transformer_donnees(df)
    
    # 1. Sélection des variables corrigées (avec les versions _grouped)
    vars_categorielles = ['v106', 'v190', 'v024', 'v501', 'v130', 'v463a', 'v113_grouped', 'v116_grouped']
    vars_numeriques = ['age']
    
    X = df[vars_numeriques + vars_categorielles]
    y = df['mort_adulte']
    poids = df['poids']
    
    # Séparation Train/Test
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, poids, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pipeline de prétraitement
    preprocesseur = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), vars_numeriques),
            ('cat', OneHotEncoder(handle_unknown='ignore'), vars_categorielles)
        ])
    
    # 2. Définition des modèles
    modeles = {
        "Régression Logistique": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, tree_method='hist', n_jobs=-1, eval_metric='logloss', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    meilleur_auc = 0
    meilleur_nom = ""
    meilleur_pipeline = None
    
    print("Début de l'entraînement des modèles avec variables regroupées...\n")
    
    for nom, modele in modeles.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocesseur), ('classifier', modele)])
        
        # Entraînement
        try:
            pipeline.fit(X_train, y_train, classifier__sample_weight=w_train)
        except TypeError:
            pipeline.fit(X_train, y_train)
            
        # Évaluation
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba, sample_weight=w_test)
        
        print(f"--- {nom} ---")
        print(f"ROC-AUC Score : {auc:.4f}")
        
        if auc > meilleur_auc:
            meilleur_auc = auc
            meilleur_nom = nom
            meilleur_pipeline = pipeline

    print(f"\n🏆 Le meilleur modèle est {meilleur_nom} avec un AUC de {meilleur_auc:.4f}")
    
    # Sauvegarde
    joblib.dump(meilleur_pipeline, 'meilleur_modele_mortalite.pkl')
    print("Modèle sauvegardé sous 'meilleur_modele_mortalite.pkl'")

if __name__ == "__main__":
    entrainer_modeles()