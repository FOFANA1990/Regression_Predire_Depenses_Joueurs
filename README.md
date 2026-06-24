# Etude de cas : Regression - Analyse de la depense des joueurs

## 1. Presentation du projet

Ce projet est une etude de cas de regression realisee dans le cadre du Mastere
Data & Intelligence Artificielle. L'objectif est de predire la depense
mensuelle des joueurs en jeux video (variable `monthly_game_spending_usd`)
a partir de leurs caracteristiques demographiques, comportementales et de
sante.

## 2. Contexte metier

Un cabinet de conseil en marketing souhaite identifier les facteurs qui
poussent les joueurs a investir financierement dans le jeu video, afin de
cibler les profils dits "Whales" (les gros depensiers). L'etude repond donc
a deux besoins :

- comprendre quels facteurs comportementaux et sociaux expliquent le niveau
  de depense ;
- construire un modele capable de predire cette depense pour un nouveau
  profil de joueur.

## 3. Jeu de donnees

Le notebook s'appuie sur le fichier `Gaming and Mental Health(in).csv`,
qui contient 1000 enregistrements de joueurs avec :

- des informations demographiques (age, genre) ;
- des informations comportementales (temps de jeu quotidien, plateforme,
  genre de jeu, anciennete dans le jeu video) ;
- des informations de sante et de mode de vie (sommeil, isolement social,
  vie sociale, activite physique, troubles associes) ;
- la variable cible : la depense mensuelle en jeux video
  (`monthly_game_spending_usd`).

Ce fichier CSV n'est pas inclus dans ce depot et doit etre place dans le
meme dossier que le notebook avant execution.

## 4. Plan du notebook

Le notebook `Etudes_de_cas_Regression.ipynb` est organise en quatre etapes
principales, precedees d'une cellule d'import des librairies utilisees pour
l'ensemble de l'etude.

### Etape 1 - Exploration du jeu de donnees (EDA)
- Description des colonnes du jeu de donnees.
- Analyse univariee de la variable cible.
- Detection des doublons et des valeurs manquantes.
- Detection de valeurs absurdes (erreurs de saisie potentielles).
- Detection des outliers (valeurs atypiques).
- Analyses bivariees et multivariees entre les variables explicatives et la
  cible.

### Etape 2 - Nettoyage des donnees
- Separation des variables explicatives (features) et de la variable cible.
- Suppression des doublons et des valeurs absurdes identifiees en EDA.
- Suppression des colonnes non pertinentes pour la modelisation.
- Traitement des valeurs manquantes. Les valeurs manquantes sur
  `grades_gpa` et `work_productivity_score` sont traitees comme
  structurelles (liees au statut etudiant ou actif du joueur, via la
  variable `is_student`) plutot que comme des valeurs aleatoires a imputer
  de facon generique.

### Etape 3 - Modelisation et optimisation
- Construction d'un modele baseline (regression lineaire sur donnees peu
  retraitees) servant d'ancre de comparaison.
- Split train / test sur les donnees nettoyees (80 % / 20 %, graine
  aleatoire fixee a 42 pour la reproductibilite).
- Construction d'une pipeline de pretraitement avec `ColumnTransformer` :
  imputation par la mediane puis standardisation pour les variables
  numeriques, imputation par le mode puis encodage one-hot pour les
  variables categorielles, et passage direct des variables booleennes
  (convertit en 0/1).
- Recherche d'hyperparametres par validation croisee (`GridSearchCV`,
  5 folds, metrique d'optimisation : MAE) pour deux familles de modeles :
  - Ridge (regularisation L2, recherche sur le parametre alpha) ;
  - Random Forest (recherche sur le nombre d'arbres, la profondeur maximale
    et la taille minimale des feuilles).
- Comparaison des trois modeles (baseline, Ridge optimise, Random Forest
  optimise) sur le jeu de test, selon les metriques MAE, RMSE et R2.

### Etape 4 - Validation et interpretation
- Analyse des residus du modele final retenu (Random Forest optimise), avec
  mise en evidence d'un biais de sous-estimation systematique pour les plus
  gros depensiers.
- Courbe d'apprentissage (learning curve), pour evaluer si davantage de
  donnees ameliorerait significativement la performance.
- Analyse de l'importance des variables (feature importance), qui confirme
  que le temps de jeu quotidien est de loin le facteur le plus determinant
  de la depense.
- Interpretation des metriques d'evaluation au regard du contexte metier
  (utilite du modele pour un classement relatif des joueurs plutot que pour
  une estimation absolue exacte).
- Prediction sur un cas personnel : application du modele final a un profil
  de joueur construit a titre d'exemple (le dictionnaire `mon_profil` dans
  le notebook peut etre librement modifie pour tester un autre profil).

## 5. Resultats principaux

Le modele final retenu est un Random Forest optimise par validation croisee,
qui surpasse nettement la baseline lineaire et le modele Ridge regularise :

- R2 d'environ 0,63 sur le jeu de test ;
- MAE d'environ 45 USD, a comparer a une depense mediane observee d'environ
  66 USD dans le jeu de donnees.

Ce niveau de performance indique que le modele est davantage adapte a un
classement relatif des joueurs (identifier les profils les plus susceptibles
de depenser) qu'a une estimation absolue et precise du montant exact qu'un
joueur va depenser. Le biais de sous-estimation observe sur les gros
depensiers doit etre garde en tete pour toute utilisation operationnelle de
ciblage marketing : les sorties du modele doivent etre interpretees comme un
minorant plutot que comme une estimation exacte du potentiel de depense.

## 6. Pistes d'amelioration identifiees

- Creation de variables d'interaction explicites (par exemple le croisement
  entre temps de jeu et isolement social).
- Test de modeles de gradient boosting (XGBoost, LightGBM).
- Modelisation dediee a la sous-population des gros depensiers, segment
  prioritaire pour le cabinet de marketing mais le moins bien predit par le
  modele actuel.

## 7. Structure attendue du projet

```
.
|-- Etudes_de_cas_Regression.ipynb     # Notebook principal de l'etude
|-- Gaming and Mental Health(in).csv   # Jeu de donnees (a fournir, non inclus)
|-- requirements.txt                   # Dependances Python necessaires
|-- README.md                          # Ce fichier
```

## 8. Installation et execution

1. Creer un environnement virtuel (recommande) :
   ```
   python -m venv venv
   source venv/bin/activate    # sur Windows : venv\Scripts\activate
   ```
2. Installer les dependances :
   ```
   pip install -r requirements.txt
   ```
3. Placer le fichier `Gaming and Mental Health(in).csv` dans le meme dossier
   que le notebook.
4. Lancer Jupyter et ouvrir le notebook :
   ```
   jupyter notebook Etudes_de_cas_Regression.ipynb
   ```
5. Executer les cellules dans l'ordre, de haut en bas. La graine aleatoire
   etant fixee (`RANDOM_STATE = 42`), les resultats obtenus doivent etre
   identiques a chaque execution.

## 9. Remarques

- Le notebook est entierement redige et commente en francais.
- Aucune cle d'API ni service externe n'est necessaire : l'ensemble du
  traitement (EDA, nettoyage, modelisation, evaluation) est realise en
  local avec les librairies listees dans `requirements.txt`.
