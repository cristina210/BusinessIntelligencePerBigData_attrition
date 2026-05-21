# -*- coding: utf-8 -*-
"""
Cross-Validation per Decision Tree e Random Forest
Versione corretta e completa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (StratifiedKFold,
                                     cross_val_score, cross_val_predict,
                                     GridSearchCV)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix)


random_seed = 42


# =============================================================================
# 1. CARICAMENTO DEI DATASET 
# =============================================================================

train_with_label = pd.read_excel('C:/Users/criba/OneDrive/Desktop/BusinessIntelligencePersonal/materiale_attrition/attrition_train.xlsx')
test_with_label  = pd.read_excel('C:/Users/criba/OneDrive/Desktop/BusinessIntelligencePersonal/materiale_attrition/attrition_test.xlsx')

# Separazione feature / target
# X_train: tutto tranne 'Attrition' (le variabili che il modello usa per imparare)
# y_train: solo 'Attrition' (quello che il modello deve predire)
X_train = train_with_label.drop(columns=['Attrition'])
y_train = train_with_label['Attrition']

X_test  = test_with_label.drop(columns=['Attrition'])
y_test  = test_with_label['Attrition']


# =============================================================================
# 2. PREPROCESSING
#    Le trasformazioni vanno fatte DOPO aver separato X e y,
#    ma il fit dello scaler va fatto SOLO su X_train (mai su X_test)
# =============================================================================

# Encoding delle variabili categoriche
#le = LabelEncoder()
#for col in X_train.select_dtypes(include='object').columns:
    #X_train[col] = le.fit_transform(X_train[col])   # fit + transform sul train
    #X_test[col]  = le.transform(X_test[col])         # solo transform sul test

# Scaling: fit SOLO su X_train, poi applica la stessa scala a X_test
#scaler  = StandardScaler()
#X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
#X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)


# =============================================================================
# 3. FUNZIONE DI VALUTAZIONE
# =============================================================================

def evaluate(y_true, y_pred, nome_modello):
    print(f"--- Report di Classificazione: {nome_modello} ---")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=['Reale: Rimasto', 'Reale: Uscito'],
                         columns=['Pred: Rimasto', 'Pred: Uscito'])
    print('\nConfusion Matrix:')
    print(cm_df)
    print("\n")


# =============================================================================
# 4. DECISION TREE
# =============================================================================

# Oggetto per la cross-validation stratificata
# 'stratified' mantiene la stessa proporzione di classi in ogni fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

# --- 4a. Valutazione base con Cross-Validation ---
print("--- DECISION TREE: VALUTAZIONE BASE CON CROSS-VALIDATION ---")

dt_base = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_impurity_decrease=0.001,
    class_weight='balanced',
    random_state=random_seed
)

# cross_val_score: restituisce un array con l'F1 di ogni fold
# Serve a misurare la stabilità del modello (quanto varia tra i fold)
dt_cv_scores = cross_val_score(dt_base, X_train, y_train, cv=skf, scoring='f1')
print(f"F1-Score per ogni fold: {np.round(dt_cv_scores, 3)}")
print(f"F1-Score Medio Base: {dt_cv_scores.mean():.3f} (+/- {dt_cv_scores.std():.3f})")

# cross_val_predict: restituisce le predizioni per ogni campione
# quando era nel fold di *validazione* (mai visto in addestramento)
# Serve per costruire confusion matrix e classification report sull'intero training set
dt_pred_cv = cross_val_predict(dt_base, X_train, y_train, cv=skf)
evaluate(y_train, dt_pred_cv, "Decision Tree (Base - CV)")


# --- 4b. Tuning manuale con Cross-Validation ---
print("--- TUNING DECISION TREE CON CROSS-VALIDATION ---")

depths   = [2, 3, 5, 8, 10, 15, 20, None]
criteria = ['entropy', 'gini']

best_f1_dt     = 0
best_params_dt = {}

for crit in criteria:
    for d in depths:
        clf = DecisionTreeClassifier(
            criterion=crit,
            max_depth=d,
            class_weight='balanced',
            random_state=random_seed
        )
        scores     = cross_val_score(clf, X_train, y_train, cv=skf, scoring='f1')
        mean_score = scores.mean()
        print(f"Criterion: {crit} | Max Depth: {d} -> F1-Score Medio CV: {mean_score:.3f}")

        if mean_score > best_f1_dt:
            best_f1_dt     = mean_score
            best_params_dt = {'criterion': crit, 'max_depth': d}

print(f"\nMigliori parametri DT: {best_params_dt} | F1 CV: {best_f1_dt:.3f}")


# --- 4c. Addestramento del modello finale sul TRAINING SET COMPLETO ---
# Ora che i parametri sono stati scelti tramite CV, addestriamo sul training set intero
dt_final = DecisionTreeClassifier(
    **best_params_dt,
    class_weight='balanced',
    random_state=random_seed
)
dt_final.fit(X_train, y_train)   # <-- addestramento sul training set completo


# =============================================================================
# 5. RANDOM FOREST
# =============================================================================

# --- 5a. Valutazione base con Cross-Validation ---
print("--- RANDOM FOREST: VALUTAZIONE BASE CON CROSS-VALIDATION ---")

rf_base = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=random_seed,
    n_jobs=-1
)

rf_cv_scores = cross_val_score(rf_base, X_train, y_train, cv=skf, scoring='f1')
print(f"F1-Score per ogni fold: {np.round(rf_cv_scores, 3)}")
print(f"F1-Score Medio Base: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std():.3f})")

rf_pred_cv = cross_val_predict(rf_base, X_train, y_train, cv=skf)
evaluate(y_train, rf_pred_cv, "Random Forest (Base - CV)")


# --- 5b. Tuning con GridSearchCV ---
# GridSearchCV usa internamente skf per valutare ogni combinazione di parametri
# NON tocca mai X_test
print("--- TUNING RANDOM FOREST CON GRIDSEARCHCV ---")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [5, 10, 20, None]
}

gs_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=random_seed, n_jobs=-1),
    param_grid_rf,
    cv=skf,
    scoring='f1',
    refit=True,   # default True: riaddestra automaticamente il best model sull'intero X_train
    n_jobs=-1
)
gs_rf.fit(X_train, y_train)   # <-- addestramento sul training set completo

print(f"Migliori parametri RF: {gs_rf.best_params_}")
print(f"Miglior F1 (CV):       {gs_rf.best_score_:.4f}")


# Heatmap dei risultati GridSearch
cv_res = pd.DataFrame(gs_rf.cv_results_)
cv_res['param_max_depth'] = cv_res['param_max_depth'].fillna('None (Senza limiti)')
pivot_rf = cv_res.pivot_table(values='mean_test_score',
                              index='param_max_depth',
                              columns='param_n_estimators')

plt.figure(figsize=(8, 4))
sns.heatmap(pivot_rf, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5)
plt.title('GridSearch RF — F1 Score per combinazione di parametri (CV)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.show()


# --- 5c. Recupero del modello finale (già fittato da GridSearchCV su X_train) ---
# Grazie a refit=True, best_estimator_ è già addestrato sull'intero training set
best_rf = gs_rf.best_estimator_


# --- 5d. Feature Importance ---
rf_importances = pd.DataFrame({
    'Feature':    X_train.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=rf_importances.head(15), x='Importance', y='Feature', palette='magma')
plt.title('Feature Importance — Random Forest (top 15)')
plt.xlabel('Importanza relativa')
plt.tight_layout()
plt.show()

print('\nTop 5 variabili più importanti:')
print(rf_importances.head(5).round(4))


# =============================================================================
# 6. VALUTAZIONE FINALE
# =============================================================================

# DecisionTree: Valutazione finale sul TEST SET (una sola volta)
print("--- DECISION TREE: VALUTAZIONE FINALE SUL TEST SET ---")
dt_pred_test = dt_final.predict(X_test)
evaluate(y_test, dt_pred_test, "Decision Tree (Finale - Test Set)")


# RandomForest: Valutazione finale sul TEST SET (una sola volta)
print("--- RANDOM FOREST: VALUTAZIONE FINALE SUL TEST SET ---")
rf_pred_test = best_rf.predict(X_test)
evaluate(y_test, rf_pred_test, "Random Forest (Finale - Test Set)")

