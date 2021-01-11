"""Objectif du TD : la réplication statistique est une méthode très utilisée en finance. On l'utilise par
exemple pour déterminer la pondération de chaque sous-jacent d'un indice, pour se couvrir avec
un nombre réduit de sous-jacents, pour réduire le périmètre de recherche et de trading
(composantes principales), etc. On s’intéresse ici à la réplication de l’indice Eurostoxx 50 (50 plus
grosses capitalisations Européennes) en utilisant différentes méthodes de sélection de variables
vues en cours."""

import numpy as np
import pandas as pd
import os
import time
import itertools
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
import statsmodels.api as sm

""" 1. Récupérer les données de l'Eurostoxx 50 dans Python à partir du fichier 'Data_Eurostoxx_10min_Feb19_Sep19.csv' """

eurostoxx_df = pd.read_csv(
    r".\Data_Eurostoxx_10min_Feb19_Sep19.csv", sep=';', decimal=",")
# print(eurostoxx_df.head())

"""
              Dates  ABI BB Equity  AD NA Equity  ADS GY Equity  AI FP Equity  ...  TEF SQ Equity  URW NA Equity  VIV FP Equity  VOW3 GY Equity  SX5E Index
0  27/02/2019 09:00          65.65        22.870          212.4        109.95  ...          7.573         143.46          24.23          150.30     3279.78
1  27/02/2019 09:10          65.69        22.490          212.2        109.90  ...          7.558         143.04          24.25          149.80     3275.70
2  27/02/2019 09:20          65.58        22.440          212.0        109.75  ...          7.552         143.00          24.14          150.30     3273.88
3  27/02/2019 09:30          65.74        22.545          211.9        109.90  ...          7.552         143.10          24.27          150.64     3277.38
4  27/02/2019 09:40          65.64        22.410          211.9        109.90  ...          7.546         142.82          24.25          150.40     3272.65

[5 rows x 50 columns]
"""

# On observe tout d'abord les dimensions de notre dataframe initial ainsi que les valeurs qu'il contient.

# print("Dimensions of original data:", eurostoxx_df.shape)
# Dimensions of original data: (7011, 50)

# print("Number of null values:", eurostoxx_df.isnull().sum().sum())
# Number of null values: 1100

# On supprime les lignes contenant des valeurs manquantes puis on affiche les dimensions de notre nouveau dataframe.

eurostoxx_df_clean = eurostoxx_df.dropna()
# print("Dimensions of original data:", eurostoxx_df_clean.shape)
# Dimensions of original data: (6660, 50)

# On vérifie que notre dataframe ne contient plus de valeurs manquantes.

# print("Number of null values:", eurostoxx_df_clean.isnull().sum().sum())
# Number of null values: 0

# On supprime la colonne "Dates" qui ne nous sera pas utile pour la suite.

eurostoxx_df_clean = eurostoxx_df_clean.drop(columns=['Dates'])

"""
2. Calculer les séries des rendements centrés et réduits de chaque composante et de
l'Eurostoxx ('SX5E Index'). Précisez quelle moyenne et std vous avez utilisées pour centrer
et réduire.
"""

# On calcule les séries de rendement

eurostoxx_df_rend = eurostoxx_df_clean.pct_change()

# On drop les NA du dataframe une nouvelle fois.

eurostoxx_df_rend = eurostoxx_df_rend.dropna()

# Considérons les données ci-dessus qui ont toutes des échelles  différentes.
# Nous pourrons commencer à comparer ces caractéristiques et les utiliser dans nos modèles une fois que nous les aurons normalisées.
# Plus tard, lorsque nous exécuterons les modèles (régression logistique, MVC, perceptrons, réseaux neuronaux, etc.), les poids estimés seront mis à jour de manière similaire plutôt qu'à des rythmes différents au cours du processus de construction.
# Nous obtiendrons ainsi des résultats plus précis lorsque les données auront été normalisées pour la première fois.

# Scale features
scaled_features = StandardScaler().fit_transform(eurostoxx_df_rend)

# Convert the scaled values into pandas DataFrame
eurostoxx_df_standardize = pd.DataFrame(
    scaled_features, columns=eurostoxx_df_rend.columns)

# Print the scaled data
# print(eurostoxx_df_standardize.head())

# We scale with the mean and the std of the whole dataset in order to have all values at the same scale
# and keep the proportion of values between columns

"""
3. Utiliser une méthode de subset selection pour répliquer au mieux l'indice à l'aide de 5, 10,
et 25 sous-jacents. Quels sont-ils ? Quelle est l'erreur de réplication dans chaque cas?
Proposer un nombre optimal de sous-jacents à retenir ?
"""

# Initialisation des variables
# Train/Test is a method to measure the accuracy of your model.
# It is called Train/Test because you split the the data set into two sets: a training set and a testing set.
# 80% for training, and 20% for testing.
# You train the model using the training set.
# You test the model using the testing set.
# Train the model means create the model.
# Test the model means test the accuracy of the model.

Y = eurostoxx_df_standardize['SX5E Index']

X_train, X_test, Y_train, Y_test = train_test_split(
    eurostoxx_df_standardize, Y, test_size=0.2, random_state=42)

X_train = X_train.drop(columns=['SX5E Index'])
X_test = X_test.drop(columns=['SX5E Index'])

# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)


def processSubset(feature_set):
    # Now we’ll fit the model on the training data:
    model = sm.OLS(Y_train, X_train[list(feature_set)])
    regr = model.fit()
    R2 = regr.rsquared
    R2_adj = regr.rsquared_adj
    # As you can see, we’re fitting the model on the training data and trying to predict the test data.
    predictions = regr.predict(X_test[list(feature_set)])
    MSE = mean_squared_error(Y_test, predictions)
    RSS = ((predictions - Y_test) ** 2).sum()
    return {"model": regr, "R2": R2, "R2_adj": R2_adj, "RSS": RSS, "MSE": MSE}


def getBestSubset(k):
    tic = time.time()
    results = []
    for combination in itertools.combinations(X_train.columns, k):
        results.append(processSubset(combination))

    # Wrap everything up in a dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest R2_adj
    best_model = models.loc[models['R2_adj'].argmax()]

    toc = time.time()
    print("Processed", models.shape[0], "models on",
          k, "predictors in", (toc-tic), "seconds.")

    return best_model

# Nous pouvons maintenant afficher les meilleurs modèles en fonction du nombre de features k
# Could take quite a while to complete...


def bestModelsComputation(nbr_features):
    models_best = pd.DataFrame(columns=["MSE", "RSS", "R2", "R2_adj", "model"])

    tic = time.time()
    for i in range(1, nbr_features):
        models_best.loc[i] = getBestSubset(i)

        toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")
    return models_best

# Exemple d'utilisation de la fonction pour 2 features, permet nottament d'afficher le nom des tickers inclus dans le modèle.

# plotBestModel(bestModelsComputation(2))

# Si l'on souhaites accéder à un modèle (i -> exemple : i = 2) en particuler, nous pouvons utiliser la fonction ci-après.

# print(models_best.loc[2, "model"].summary())


""" On réplique maintenant au mieux l'indice à l'aide de 3 sous jacents """
# 17296 combinations

# print(getBestSubset(3)["model"].summary())

"""
Processed 17296 models on 3 predictors in 452.66223883628845 seconds.
                                 OLS Regression Results
=======================================================================================
Dep. Variable:             SX5E Index   R-squared (uncentered):                   0.807
Model:                            OLS   Adj. R-squared (uncentered):              0.807
Method:                 Least Squares   F-statistic:                              7419.
Date:                Sun, 10 Jan 2021   Prob (F-statistic):                        0.00
Time:                        20:33:45   Log-Likelihood:                         -3183.9
No. Observations:                5327   AIC:                                      6374.
Df Residuals:                    5324   BIC:                                      6393.
Df Model:                           3
Covariance Type:            nonrobust
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
AI FP Equity      0.3831      0.007     54.621      0.000       0.369       0.397
MC FP Equity      0.3765      0.007     53.861      0.000       0.363       0.390
SAN SQ Equity     0.3878      0.007     58.295      0.000       0.375       0.401
==============================================================================
Omnibus:                     1779.043   Durbin-Watson:                   2.045
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            81151.005
Skew:                          -0.853   Prob(JB):                         0.00
Kurtosis:                      22.045   Cond. No.                         1.88
==============================================================================

Notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

""" On réplique maintenant au mieux l'indice à l'aide de 5 sous jacents """
# 1712304 combinations

# print(getBestSubset(5)["model"].summary())

""" On réplique maintenant au mieux l'indice à l'aide de 10 sous jacents """
# 6540715896 combinations

# print(getBestSubset(10)["model"].summary())

""" On réplique maintenant au mieux l'indice à l'aide de 25 sous jacents """
# 30957699535776 combinations

# print(getBestSubset(25)["model"].summary())

# We can also use a similar approach to perform forward stepwise, using a slight modification of the functions we defined above:


def forward(predictors):

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X_train.columns if p not in predictors]

    tic = time.time()

    results = []

    for p in remaining_predictors:
        results.append(processSubset(predictors+[p]))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest R2
    best_model = models.loc[models['R2_adj'].argmax()]

    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(
        predictors)+1, "predictors in", (toc-tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model


def forwardSubsetSelection():
    models_fwd = pd.DataFrame(columns=["MSE", "RSS", "R2", "R2_adj", "model"])

    tic = time.time()
    predictors = []

    for i in range(1, len(X_train.columns)+1):
        models_fwd.loc[i] = forward(predictors)
        predictors = models_fwd.loc[i]["model"].model.exog_names

    toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")

    return models_fwd


# models_fwd = forwardSubsetSelection()
# print(models_fwd)
# print(models_fwd.loc[1, "model"].summary())

# Si l'on cherche le meilleur modèle avec 1 feature, on obtient.
#       MSE         RSS        R2      R2_adj
# 1   0.412945  550.043306  0.538271  0.538184

"""
Total elapsed time: 29.788211822509766 seconds.
                                 OLS Regression Results
=======================================================================================
Dep. Variable:             SX5E Index   R-squared (uncentered):                   0.538
Model:                            OLS   Adj. R-squared (uncentered):              0.538
Method:                 Least Squares   F-statistic:                              6209.
Date:                Mon, 11 Jan 2021   Prob (F-statistic):                        0.00
Time:                        10:12:05   Log-Likelihood:                         -5506.8
No. Observations:                5327   AIC:                                  1.102e+04
Df Residuals:                    5326   BIC:                                  1.102e+04
Df Model:                           1
Covariance Type:            nonrobust
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
BAS GY Equity     0.7127      0.009     78.797      0.000       0.695       0.730
==============================================================================
Omnibus:                     3062.970   Durbin-Watson:                   1.996
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           397369.536
Skew:                           1.765   Prob(JB):                         0.00
Kurtosis:                      45.164   Cond. No.                         1.00
==============================================================================

Notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# On s'intéresse maintenant au meilleur modèle, on remarque que le R²out ajusté augmente dans ce cas, lui aussi, avec le nombre de features.
# On peut donc conclure qu'il y a un intérêt à conserser un maximum de régrésseurs ici.
# best_model = models_fwd.loc[models_fwd['R2_adj'].argmax()]
# print(best_model["model"].summary())

"""
                                 OLS Regression Results
=======================================================================================
Dep. Variable:             SX5E Index   R-squared (uncentered):                   0.994
Model:                            OLS   Adj. R-squared (uncentered):              0.993
Method:                 Least Squares   F-statistic:                          1.722e+04
Date:                Sun, 10 Jan 2021   Prob (F-statistic):                        0.00
Time:                        23:00:02   Log-Likelihood:                          5856.2
No. Observations:                5327   AIC:                                 -1.162e+04
Df Residuals:                    5280   BIC:                                 -1.131e+04
Df Model:                          47
Covariance Type:            nonrobust
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
BAS GY Equity       0.0535      0.002     28.834      0.000       0.050       0.057
AIR FP Equity       0.0466      0.002     27.478      0.000       0.043       0.050
MUV2 GY Equity      0.0350      0.002     23.016      0.000       0.032       0.038
MC FP Equity        0.0720      0.002     38.230      0.000       0.068       0.076
SAN SQ Equity       0.0538      0.002     22.290      0.000       0.049       0.059
BN FP Equity        0.0277      0.002     18.187      0.000       0.025       0.031
FP FP Equity        0.0756      0.002     40.060      0.000       0.072       0.079
SAP GY Equity       0.0862      0.002     55.819      0.000       0.083       0.089
SAN FP Equity       0.0532      0.001     38.785      0.000       0.051       0.056
ASML NA Equity      0.0639      0.002     38.352      0.000       0.061       0.067
LIN GY Equity       0.0624      0.001     42.902      0.000       0.060       0.065
BNP FP Equity       0.0499      0.002     24.842      0.000       0.046       0.054
ENGI FP Equity      0.0132      0.001      9.307      0.000       0.010       0.016
SIE GY Equity       0.0548      0.002     32.968      0.000       0.052       0.058
ABI BB Equity       0.0536      0.001     35.807      0.000       0.051       0.057
DAI GY Equity       0.0436      0.002     23.577      0.000       0.040       0.047
ENEL IM Equity      0.0299      0.002     19.443      0.000       0.027       0.033
BAYN GY Equity      0.0642      0.001     44.179      0.000       0.061       0.067
ADS GY Equity       0.0340      0.001     26.202      0.000       0.031       0.037
ORA FP Equity       0.0174      0.002     11.371      0.000       0.014       0.020
OR FP Equity        0.0542      0.002     35.703      0.000       0.051       0.057
PHIA NA Equity      0.0225      0.001     15.280      0.000       0.020       0.025
EL FP Equity        0.0290      0.001     24.072      0.000       0.027       0.031
SAF FP Equity       0.0344      0.002     22.640      0.000       0.031       0.037
CS FP Equity        0.0366      0.002     21.462      0.000       0.033       0.040
DG FP Equity        0.0304      0.002     19.722      0.000       0.027       0.033
ISP IM Equity       0.0228      0.002     13.667      0.000       0.020       0.026
ITX SQ Equity       0.0173      0.001     12.698      0.000       0.015       0.020
DTE GY Equity       0.0306      0.002     20.247      0.000       0.028       0.034
SU FP Equity        0.0245      0.002     13.582      0.000       0.021       0.028
FRE GY Equity       0.0188      0.001     13.228      0.000       0.016       0.022
INGA NA Equity      0.0350      0.002     17.849      0.000       0.031       0.039
AI FP Equity        0.0295      0.002     17.284      0.000       0.026       0.033
AMS SQ Equity       0.0238      0.002     15.820      0.000       0.021       0.027
KER FP Equity       0.0354      0.002     19.738      0.000       0.032       0.039
DPW GY Equity       0.0227      0.002     14.033      0.000       0.020       0.026
IBE SQ Equity       0.0287      0.002     19.023      0.000       0.026       0.032
NOKIA FH Equity     0.0223      0.001     17.488      0.000       0.020       0.025
VOW3 GY Equity      0.0310      0.002     16.165      0.000       0.027       0.035
CRH ID Equity       0.0202      0.001     13.575      0.000       0.017       0.023
GLE FP Equity       0.0206      0.002     11.628      0.000       0.017       0.024
AD NA Equity        0.0164      0.001     12.561      0.000       0.014       0.019
BBVA SQ Equity      0.0237      0.002     10.810      0.000       0.019       0.028
ENI IM Equity       0.0200      0.002     10.353      0.000       0.016       0.024
VIV FP Equity       0.0141      0.002      9.241      0.000       0.011       0.017
TEF SQ Equity       0.0132      0.002      8.057      0.000       0.010       0.016
URW NA Equity       0.0097      0.001      7.952      0.000       0.007       0.012
==============================================================================
Omnibus:                     1518.222   Durbin-Watson:                   1.962
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           103417.887
Skew:                          -0.454   Prob(JB):                         0.00
Kurtosis:                      24.566   Cond. No.                         10.4
==============================================================================

Notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Les 5 meilleurs sous jacents sont donc (dans le cadre d'une sélection forward):
# BAS GY Equity, AIR FP Equity, MUV2 GY Equity, MC FP Equity, SAN SQ Equity
#       MSE         RSS        R2      R2_adj
# 5   0.127115  169.316750  0.858443  0.858310

# Les 10 meilleurs sont :
# BAS GY Equity, AIR FP Equity, MUV2 GY Equity, MC FP Equity, SAN SQ Equity, BN FP Equity, FP FP Equity, SAP GY Equity, SAN FP Equity, ASML NA Equity
#       MSE         RSS        R2      R2_adj
# 10  0.056803   75.661625  0.942500  0.942392

# Les 25 meilleurs sous jacents sont : BAS GY Equity, AIR FP Equity, MUV2 GY Equity, MC FP Equity, SAN SQ Equity, BN FP Equity, FP FP Equity, SAP GY Equity, SAN FP Equity, ASML NA Equity, LIN GY Equity, BNP FP Equity, ENGI FP Equity, SIE GY Equity, ABI BB Equity, DAI GY Equity, ENEL IM Equity, BAYN GY Equity, ADS GY Equity, ORA FP Equity, OR FP Equity, PHIA NA Equity, EL FP Equity, SAF FP Equity, CS FP Equity
#       MSE         RSS        R2      R2_adj
# 25  0.016919   22.536773  0.985724  0.985657

# Ainsi le meilleur modèle parmis les 3 ci dessus serait celui comportant 25 sous jacents.

"""
On nous demande maintenant de calculer l'erreur de réplication dans chaque cas
Pour cela, nous devons calculer les returns des sous-jacents sélectionnés ainsi que le return de l'index
"""

"""
Le rendement d'un portefeuille est la moyenne pondérée des actifs individuels du portefeuille (ici weight = 1).
Nous allons calculer le rendement des sous jacents sélectionnés et le comparer au rendement de l'index.
Le rendement du portefeuille est simplement la somme des rendements pondérés des assets le composant.
"""


def trackingError(list_features):
    df = pd.DataFrame(
        {'Portfolio_Returns': eurostoxx_df_standardize[list_features].sum(axis=1).values, 'Bench_Returns': Y.values})
    tracking_error = np.sqrt(
        sum([val**2 for val in df['Portfolio_Returns'] - df['Bench_Returns']]))
    return tracking_error

# On vérifie bien que la tracking error du bench est bien nulle.
# print(trackingError(["SX5E Index"]))
# 0


list_5_features = ["BAS GY Equity", "AIR FP Equity",
                   "MUV2 GY Equity", "MC FP Equity", "SAN SQ Equity"]

# print(trackingError(list_5_features))
# 228.07002209304443
# Soit une trackingError moyenne de 45,6 par feature (228,07/5)

list_10_features = ["BAS GY Equity", "AIR FP Equity", "MUV2 GY Equity", "MC FP Equity",
                    "SAN SQ Equity", "BN FP Equity", "FP FP Equity", "SAP GY Equity", "SAN FP Equity", "ASML NA Equity"]

# print(trackingError(list_10_features))
# 454.2119340244009
# Soit une trackingError moyenne de 45,42 par feature (454,21/10)

list_25_features = ["BAS GY Equity", "AIR FP Equity", "MUV2 GY Equity", "MC FP Equity",
                    "SAN SQ Equity", "BN FP Equity", "FP FP Equity", "SAP GY Equity", "SAN FP Equity", "ASML NA Equity",
                    "LIN GY Equity", "BNP FP Equity", "ENGI FP Equity", "SIE GY Equity", "ABI BB Equity", "DAI GY Equity",
                    "ENEL IM Equity", "BAYN GY Equity", "ADS GY Equity", "ORA FP Equity", "OR FP Equity", "PHIA NA Equity",
                    "EL FP Equity", "SAF FP Equity", "CS FP Equity"]

# print(trackingError(list_25_features))
# 1125.305958283886
# Soit une trackingError moyenne de 45,014 par feature (1125.3095/25)

list_48_features = ["BAS GY Equity", "AIR FP Equity", "MUV2 GY Equity", "MC FP Equity", "SAN SQ Equity", "BN FP Equity", "FP FP Equity", "SAP GY Equity", "SAN FP Equity", "ASML NA Equity", "LIN GY Equity", "BNP FP Equity", "ENGI FP Equity", "SIE GY Equity", "ABI BB Equity", "DAI GY Equity", "ENEL IM Equity", "BAYN GY Equity", "ADS GY Equity", "ORA FP Equity", "OR FP Equity", "PHIA NA Equity", "EL FP Equity",
                    "SAF FP Equity", "CS FP Equity", "DG FP Equity", "ISP IM Equity", "ITX SQ Equity", "DTE GY Equity", "SU FP Equity", "FRE GY Equity", "INGA NA Equity", "AI FP Equity", "AMS SQ Equity", "KER FP Equity", "DPW GY Equity", "IBE SQ Equity", "NOKIA FH Equity", "VOW3 GY Equity", "CRH ID Equity", "GLE FP Equity", "AD NA Equity", "BBVA SQ Equity", "ENI IM Equity", "VIV FP Equity", "TEF SQ Equity", "URW NA Equity"]

# print(trackingError(list_48_features))
# 2125.1177943705666
# Soit une trackingError moyenne de 44,27 par feature (2125.11/48)

# Un nombre optimal de sous jacents à retenir serait probablement 5. En effet pour 5 sous jacents, les différents indicateurs,
# R²out, R² out ajusté, MSE, RSS et Tracking Error sont satisfaisants. Si nous sélectionnons 25 sous jacents nous aurons probablement
# un modèle plus précis mais peut-être moins performant lors de son traitement.

"""
4. Comparer les méthodes de réplication suivantes sur les séries des rendements (in et outsample) : multi-régression, multi-régression régularisée Ridge, Lasso et Elastic Net en
étudiant l'impact des paramètres de régularisation. Calculer l'erreur de réplication dans
chaque cas.
"""


def processSubsetRidge(feature_set):
    alphas = np.linspace(.00001, 2, 500)
    # Get the best version of alpha
    model = RidgeCV(
        alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
    regr = model.fit(X_train[list(feature_set)], Y_train)
    predictions = regr.predict(X_test[list(feature_set)])
    R2 = regr.score(X_train[list(feature_set)], Y_train)
    RSS = ((predictions - Y_test) ** 2).sum()
    MSE = mean_squared_error(Y_test, predictions)
    return {"model": regr, "R2": R2, "RSS": RSS, "MSE": MSE}


""" On effectue la méthode de réplication sur les sous jacents sélectionnées précédemment """

# print(processSubsetRidge(list_5_features))
# BAS GY Equity, AIR FP Equity, MUV2 GY Equity, MC FP Equity, SAN SQ Equity
# Linear Regressions results :
#       MSE        RSS        R2
#    0.127115  169.316750  0.858443

# Ridge results :
#       MSE                  RSS                  R2
#  0.1292996270896124  172.2271032833637  0.8581344824435092


def processSubsetLasso(feature_set):
    alphas = np.linspace(.00001, 2, 500)
    # Get the best version of alpha
    model = LassoCV(alphas=alphas, normalize=False,
                    fit_intercept=False, random_state=0, cv=5)
    regr = model.fit(X_train[list(feature_set)], Y_train)

    # valeurs des alphas qui ont été testés
    print(regr.alphas_)
    # valeurs des MSE en validation croisée
    print(regr.mse_path_)

    # La propriété mse_path_ est une matrice (500 x 5) : 500 parce que 500 versions de α ont été
    # testées ; 5 parce que nous avons demandé une 5-fold validation croisée.
    # Nous calculons la moyenne pour disposer d’une mesure de performance synthétique pour
    # chaque scénario. Puis nous affichons le tableau mettant en relation α et le MSE (moyen) en
    # validation croisée.

    # moyenne mse en validation croisée pour chaque alpha
    avg_mse = np.mean(regr.mse_path_, axis=1)

    # alphas vs. MSE en cross-validation
    print(pd.DataFrame({'alpha': regr.alphas_, 'MSE': avg_mse}))

    # 0.000010 est la valeur de alpha qui minimise la MSE

    predictions = regr.predict(X_test[list(feature_set)])
    R2 = regr.score(X_train[list(feature_set)], Y_train)
    RSS = ((predictions - Y_test) ** 2).sum()
    MSE = mean_squared_error(Y_test, predictions)
    return {"model": regr, "R2": R2, "RSS": RSS, "MSE": MSE}


# print(processSubsetLasso(list_5_features))
# BAS GY Equity, AIR FP Equity, MUV2 GY Equity, MC FP Equity, SAN SQ Equity
# Linear Regressions results :
#       MSE        RSS        R2
#    0.127115  169.316750  0.858443
# Lasso results :
#       MSE                 RSS                  R2
# 0.1277343683280292  170.1421786129349  0.8584138275189748


def processSubsetElasticNet(feature_set):
    alphas = np.linspace(.00001, 2, 500)
    # Get the best version of alpha
    model = ElasticNetCV(alphas=alphas, normalize=False,
                         fit_intercept=False, random_state=0, cv=5)
    regr = model.fit(X_train[list(feature_set)], Y_train)

    # valeurs des alphas qui ont été testés
    print(regr.alphas_)
    # valeurs des MSE en validation croisée
    print(regr.mse_path_)

    # La propriété mse_path_ est une matrice (500 x 5) : 500 parce que 500 versions de α ont été
    # testées ; 5 parce que nous avons demandé une 5-fold validation croisée.
    # Nous calculons la moyenne pour disposer d’une mesure de performance synthétique pour
    # chaque scénario. Puis nous affichons le tableau mettant en relation α et le MSE (moyen) en
    # validation croisée.

    # moyenne mse en validation croisée pour chaque alpha
    avg_mse = np.mean(regr.mse_path_, axis=1)

    # alphas vs. MSE en cross-validation
    print(pd.DataFrame({'alpha': regr.alphas_, 'MSE': avg_mse}))

    # 0.000010 est la valeur de alpha qui minimise la MSE

    predictions = regr.predict(X_test[list(feature_set)])
    R2 = regr.score(X_train[list(feature_set)], Y_train)
    RSS = ((predictions - Y_test) ** 2).sum()
    MSE = mean_squared_error(Y_test, predictions)
    return {"model": regr, "R2": R2, "RSS": RSS, "MSE": MSE}


# print(processSubsetElasticNet(list_5_features))
# BAS GY Equity, AIR FP Equity, MUV2 GY Equity, MC FP Equity, SAN SQ Equity
# Linear Regressions results :
#       MSE        RSS        R2
#    0.127115  169.316750  0.858443
# ElasticNet results :
#       MSE                 RSS                   R2
# 0.1278914496727536  170.35141096410777  0.8583981130006226

"""
Summary Results :
# BAS GY Equity, AIR FP Equity, MUV2 GY Equity, MC FP Equity, SAN SQ Equity
# Linear Regressions results :
#       MSE        RSS        R2
#    0.127115  169.316750  0.858443
-------------------------------------------------------------------
# Ridge results :
#       MSE                  RSS                  R2
#  0.1292996270896124  172.2271032833637  0.8581344824435092
-------------------------------------------------------------------
# Lasso results :
#       MSE                 RSS                  R2
# 0.1277343683280292  170.1421786129349  0.8584138275189748
-------------------------------------------------------------------
# ElasticNet results :
#       MSE                 RSS                   R2
# 0.1278914496727536  170.35141096410777  0.8583981130006226
"""

"""
5. Déterminer les composantes principales des 50 sous-jacents de l'indice et leur variance
expliquée et tracer leur évolution. Proposer un nombre de composantes à conserver et
essayer de les décrire simplement. En quoi la première composante diffère-t-elle de la
régression multiple ? Régresser l'indice sur les k premières composantes et calculer l'erreur
de réplication.
"""

features = ["BAS GY Equity", "AIR FP Equity", "MUV2 GY Equity", "MC FP Equity", "SAN SQ Equity", "BN FP Equity", "FP FP Equity", "SAP GY Equity", "SAN FP Equity", "ASML NA Equity", "LIN GY Equity", "BNP FP Equity", "ENGI FP Equity", "SIE GY Equity", "ABI BB Equity", "DAI GY Equity", "ENEL IM Equity", "BAYN GY Equity", "ADS GY Equity", "ORA FP Equity", "OR FP Equity", "PHIA NA Equity", "EL FP Equity",
            "SAF FP Equity", "CS FP Equity", "DG FP Equity", "ISP IM Equity", "ITX SQ Equity", "DTE GY Equity", "SU FP Equity", "FRE GY Equity", "INGA NA Equity", "AI FP Equity", "AMS SQ Equity", "KER FP Equity", "DPW GY Equity", "IBE SQ Equity", "NOKIA FH Equity", "VOW3 GY Equity", "CRH ID Equity", "GLE FP Equity", "AD NA Equity", "BBVA SQ Equity", "ENI IM Equity", "VIV FP Equity", "TEF SQ Equity", "URW NA Equity"]

# Separating out the features
X = eurostoxx_df_clean.loc[:, features].values

# Separating out the target
Y = eurostoxx_df_clean.loc[:, ['SX5E Index']].values

# Standardizing the features
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data=principalComponents, columns=[
                           'principal component 1', 'principal component 2'])

# Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.
finalDf = pd.concat([principalDf, eurostoxx_df_clean[['SX5E Index']]], axis=1)

# print(finalDf.head())
"""principal component 1  principal component 2  SX5E Index
0              -6.170550               3.898411     3279.78
1              -6.046791               3.987295     3275.70
2              -6.114200               4.028436     3273.88
3              -6.111830               3.937373     3277.38
4              -5.978858               4.080713     3272.65
"""


def show_PCA_2D():
    plt.figure(figsize=(8, 6))
    plt.scatter(principalComponents[:, 0],
                principalComponents[:, 1], c=eurostoxx_df_clean['SX5E Index'], cmap='rainbow')
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.show()


# show_PCA_2D()
# Clearly by using these two components we can separate these two classes.


def heatmap2D():
    data = pd.DataFrame(pca.components_, columns=features)
    plt.figure(figsize=(12, 7))
    sns.heatmap(data, cmap='YlGnBu', linewidths=.5)
    plt.show()

# heatmap2D()

# Explained Variance
# The explained variance tells you how much information (variance) can be attributed to each of the principal components.
# This is important as while you can convert 48 dimensional space to 2 dimensional space, you lose some of the variance (information) when you do this.
#  By using the attribute explained_variance_ratio_, you can see that the first principal component contains 48.57% of the variance and the second principal component contains 19.62% of the variance.
# Together, the two components contain 68.19% of the information.


print(pca.explained_variance_ratio_)
# [0.48565493 0.19621787] -> 2 dim
# [0.48565493 0.19621787 0.12978701 0.0555851  0.03639996] -> 5 dim


def show_explained_cum_var():
    pca = PCA()
    pca.fit(X)

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    x = range(1, exp_var_cumul.shape[0] + 1)
    y = exp_var_cumul

    plt.stackplot(x, y, labels=["# Components", "Explained Variance"])
    plt.legend(loc='upper left')
    plt.show()


# show_explained_cum_var()

# Au vu de la variance expliquée, nous pourrions retenir deux variables mais il serait préférable d'en sélectionner trois.
# Cela permettrait d'exprimer et représenter ~ 80% de la variance du modèle.


def show_PCA_3D():
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    principalDf = pd.DataFrame(data=components, columns=[
                               'principal component 1', 'principal component 2', 'principal component 3'])

    # Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.
    finalDf = pd.concat(
        [principalDf, eurostoxx_df_clean[['SX5E Index']]], axis=1)

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(principalDf['principal component 2'], principalDf['principal component 1'],
                    principalDf['principal component 3'], cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()

    # to Add a color bar which maps values to colors.
    surf = ax.plot_trisurf(principalDf['principal component 2'], principalDf['principal component 1'],
                           principalDf['principal component 3'], cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Rotate it
    ax.view_init(30, 45)
    plt.show()


# show_PCA_3D()
