# Jane Street Financial Market — Ridge, LightGBM et LSTM

Ce projet compare **trois modèles de régression** pour prédire **`responder_6`** :

- **Ridge (scikit-learn)** : modèle linéaire de référence.
- **Gradient Boosting (LightGBM)** : ensemble d'arbres pour les relations non linéaires.
- **LSTM (PyTorch)** : réseau pour étudier l'intérêt de l'historique temporel.

La métrique principale est le **R² pondéré à moyenne nulle** de Jane Street.

## 1. Installation et lancement

À la racine du dossier `projetML-` :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

**Important :** Il faut d'abord disposer des données réelles exportées par ton script `prepare.py` (connexion réseau et DuckDB nécessaires) :

```bash
python prepare.py
python main.py
```

Pour ne lancer que certains modèles :

```bash
python main.py --models ridge gradient_boosting
```

Pour relire les métriques déjà calculées **sans relancer l'entraînement** :

```bash
python compare.py
```

Pour lancer l'analyse exploratoire d'origine :

```bash
python analysing-eda.py "chemin/vers/part-0.parquet"
```

Le fichier `.env` permet de modifier les chemins, la cible, les modèles exécutés et leurs hyperparamètres **sans modifier chaque script**. `true` signifie *activer* ; `false` signifie *désactiver*. Les plages de dates du `.env` sont utilisées pour **contrôler les CSV existants**, et **ne modifient pas** le découpage déjà écrit dans `prepare.py`.

## 2. Données et preprocessing commun

**Entrées brutes :** `feature_00` à `feature_78` (79 features).
**Entrées des modèles :** 70 features après exclusion des **9 colonnes fixées par l'EDA** :
`feature_00`, `feature_01`, `feature_02`, `feature_03`, `feature_04`, `feature_21`, `feature_26`, `feature_27`, `feature_31`.
Cette liste est définie une seule fois dans `EXCLUDED_FEATURES` du `.env`, et appliquée **à l'identique** à `train.csv`, `val.csv` et `test.csv` lors de leur lecture, **sans modifier les CSV sources ni le découpage `prepare.py`**. L'EDA portait sur `part-0.parquet`, pas seulement sur le train ; cette exclusion est donc une **décision explicite**, pas une détection automatique sur train.

**Cible :** `responder_6`.
**Pondération :** `weight`.
**Ordre temporel pour LSTM :** `symbol_id`, `date_id`, `time_id`.

Les autres colonnes `responder_*` ne sont jamais utilisées pour prédire la cible.

### Gestion des valeurs manquantes — choix retenu

1. **Exclure uniquement les neuf colonnes nommées ci-dessus**, pour les trois jeux. Aucune autre feature n'est retirée automatiquement, même si elle se révèle entièrement vide sur le train.
2. **Aucune ligne supprimée à cause d'un NaN dans les features.** Une cible ou un poids invalide provoque une erreur explicite.
3. **Pas de moyenne, médiane ni moyenne glissante pour remplir les NaN.**
4. **Ridge et LSTM :** standardiser à partir des statistiques des valeurs *observées* dans le train ; remplacer ensuite les `NaN` par `0` uniquement pour les calculs ; conserver les indicateurs binaires (`1` si absent, `0` si présent). Cela préserve notamment l'indication d'absence pour les features **partiellement** renseignées.
5. **LightGBM :** conserver directement les `NaN` des 70 features, sans standardisation.
6. **Les paramètres de standardisation sont calculés uniquement sur `train.csv`**, puis réutilisés pour `val.csv` et `test.csv`.

Pour garder les colonnes d'entrée identiques dans les trois jeux, le pipeline existant conserve un indicateur pour **chacune des 70 features retenues** : les indicateurs des features complètes restent à `0` et ceux des features partiellement renseignées signalent leurs absences. Ridge et LSTM reçoivent donc **140 entrées** (70 valeurs + 70 indicateurs), tandis que LightGBM reçoit **70 features**, avec leurs éventuels `NaN`.

Le preprocessing **ne découpe pas les données** : il lit `train.csv`, `val.csv` et `test.csv` déjà créés. Il contrôle les plages attendues de l'export d'origine : train **0–109**, validation **110–153**, test **154–219**.

## 3. Fonctionnement, avantages et limites des modèles

### Ridge — `ridge.py`

- **Principe :** prédit la cible par une combinaison linéaire des variables, avec une pénalisation L2.
- **Avantages :** rapide, simple et facile à interpréter ; bon point de comparaison.
- **Limites :** ne capture pas naturellement des interactions non linéaires complexes.
- **Traitement :** valeurs standardisées et indicateurs d'absence.
- **Choix du modèle :** tester les valeurs `RIDGE_ALPHAS` et retenir la meilleure selon le R² pondéré sur validation.

### Gradient Boosting / LightGBM — `gradient_boosting.py`

- **Principe :** construit une suite d'arbres qui corrigent progressivement les erreurs des précédents.
- **Avantages :** efficace pour les grands tableaux, apprend les interactions et gère directement les `NaN`.
- **Limites :** réglages plus nombreux, risque de surapprentissage ; n'exploite pas l'historique temporel en séquence.
- **Traitement :** features brutes, `NaN` conservés.
- **Choix du modèle :** *early stopping* sur l'erreur quadratique **pondérée** de validation.

### LSTM — `lstm.py`

- **Principe :** utilise les observations passées et courantes du **même instrument** pour construire une fenêtre, puis prédit `responder_6`.
- **Avantages :** peut apprendre des dépendances temporelles.
- **Limites :** plus lent et coûteux ; sensible à la longueur des séquences et aux réglages ; n'est pas automatiquement meilleur que Ridge ou LightGBM.
- **Traitement :** valeurs standardisées + indicateurs d'absence, comme Ridge.
- **Chronologie :** l'historique est trié par `symbol_id`, `date_id`, `time_id` et une fenêtre n'utilise **jamais de valeur future**. Les premières fenêtres sont complétées à gauche par des zéros. Un historique de features d'une période antérieure peut être utilisé en validation/test, sans utiliser de réponse cible passée comme entrée. On suppose que les features de l'instant courant sont disponibles à l'instant de prédiction.
- **Choix du modèle :** meilleure époque sur le R² pondéré de validation, avec arrêt anticipé.

## 4. Évaluation et comparaison

Pour chaque modèle, `metrics.py` calcule sur les **mêmes lignes** de validation et de test :

- **R² pondéré à moyenne nulle** : `1 − Σ(w × (y − ŷ)²) / Σ(w × y²)` ; **0 correspond à la prédiction constante zéro**.
- **RMSE pondéré** : racine de l'erreur quadratique moyenne pondérée.
- **MAE pondéré** : erreur absolue moyenne pondérée.

La sélection d'`alpha`, de l'itération LightGBM et de l'époque LSTM se fait **sur validation**. Le test n'est pas utilisé pour choisir les hyperparamètres ; il sert à la comparaison finale.

`main.py` écrit notamment :

```text
outputs/
├── model_metrics.csv
├── comparison_test.csv
├── gradient_boosting_importances.csv
├── models/
│   ├── preprocessing.joblib
│   ├── ridge.joblib
│   ├── gradient_boosting.txt
│   └── lstm.pt
├── figures/
│   ├── comparison_r2.png
│   └── *_predictions.png
└── predictions/
    └── *_test.csv   # Seulement si SAVE_TEST_PREDICTIONS=true
```

`SAVE_TEST_PREDICTIONS=false` évite de créer de gros CSV de prédictions ; le changer en `true` enregistre les prédictions individuelles de test.

## 5. Configuration `.env`

- `MODELS` : liste des modèles à lancer (séparés par des virgules).
- `RIDGE_ALPHAS` : forces de régularisation Ridge candidates.
- `GBM_*` : nombre d'arbres, taux d'apprentissage, taille des arbres et arrêt anticipé.
- `LSTM_*` : longueur des fenêtres, taille du réseau, batch, époques, taux d'apprentissage, CPU/GPU.
- `MAX_TRAIN_ROWS`, `MAX_VAL_ROWS`, `MAX_TEST_ROWS` : **0 = toutes les observations** ; une limite supérieure à zéro active un mode technique de débogage non représentatif.
- `LSTM_DEVICE=auto` choisit CUDA si disponible, sinon CPU ; `cpu` force le CPU ; `cuda` exige un GPU CUDA.

Chaque variable est commentée directement dans `.env`.

## 6. Tests

```bash
python -m unittest discover -s tests -v
```

Ils vérifient notamment l’exclusion fixe des neuf colonnes de l’EDA, les indicateurs de NaN sur les colonnes conservées, la standardisation sur train uniquement, l'absence d'observations futures dans les fenêtres LSTM et les métriques pondérées. Les tests utilisent des **données synthétiques uniquement pour vérifier le code** ; ils ne démontrent aucune performance réelle sur Jane Street.

