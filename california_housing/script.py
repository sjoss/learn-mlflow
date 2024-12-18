#!/usr/bin/env python
# coding: utf-8

# In[3]:


import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


california = fetch_california_housing(as_frame=True)
df = california.frame
df


# In[ ]:


df.describe()


# In[ ]:


# %%writefile script.py

import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns




# Initialiser l'expérience MLflow
mlflow.set_experiment("california_housing_regression")
    with mlflow.start_run(run_name="data-pipeline"):
        mlflow.set_tag("mlflow.runName", "data-pipeline")

    with mlflow.start_run(run_name="data_exploration") as run:
        # Charger le jeu de données
        california = fetch_california_housing(as_frame=True)
        df = california.frame
        mlflow.log_param("dataset_shape", df.shape)

        # Analyse descriptive
        desc = df.describe()
        print(desc)
        mlflow.log_text(desc.to_string(), "description.txt")
        mlflow.log_artifact("script.py")

        # Visualisation des distributions (exemples)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['MedHouseVal'], bins=50)
        plt.title('Distribution des Prix des Maisons')
        plt.savefig("histogram_price.png")
        mlflow.log_artifact("histogram_price.png")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="MedInc", y="MedHouseVal", data=df)
        plt.title("Relation entre Revenu Médian et Prix")
        plt.savefig("scatter_income_price.png")
        mlflow.log_artifact("scatter_income_price.png")

        # Log des infos, observations, etc.
        mlflow.log_text("Observation: Le prix des maisons a une distribution non normale...", "data_insights.txt")
        print("Exploration terminée")


    # In[ ]:


    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    with mlflow.start_run(run_name="logistic_regression_baseline") as run:
        # Préparation des données pour la classification (exemple: prix > moyenne devient 1, sinon 0)
        df['target_class'] = (df['MedHouseVal'] > df['MedHouseVal'].mean()).astype(int)
        X = df.drop(['MedHouseVal','target_class'], axis=1)
        y = df['target_class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Mise à l'échelle des features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entraînement du modèle de régression logistique
        model = LogisticRegression(solver='liblinear', random_state=42)  #solver to prevent warnings
        model.fit(X_train_scaled, y_train)

        # Prédiction et évaluation
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log des paramètres, métriques et modèle
        mlflow.log_param("solver", "liblinear")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_dict(report, "classification_report.json")
        mlflow.sklearn.log_model(model, "logistic_model")
        print("Modèle Logistic Regression Entraîné")


    # In[ ]:


    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Prepare Data
    X = df.drop(['MedHouseVal','target_class'], axis=1) # using the target_class column for now
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"{name}: MSE = {mse:.2f}, R2 = {r2:.2f}")


    # In[ ]:


    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np

    with mlflow.start_run(run_name="random_forest_tuning") as run:
    rf_model = RandomForestRegressor(random_state=42)
    param_dist = {
            "n_estimators": np.arange(20, 60),
            "max_depth": [5, 10, 15, None],
            "min_samples_split": np.arange(2, 10),
            "min_samples_leaf": np.arange(1, 5)
    }
    rf_random = RandomizedSearchCV(rf_model, param_dist, n_iter=2, cv=3, random_state=42, scoring='neg_mean_squared_error')
    rf_random.fit(X_train, y_train)

    best_model = rf_random.best_estimator_
    best_params = rf_random.best_params_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_params(best_params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(best_model, "best_model")
    print(f"Best Random Forest Model (Tuned): MSE = {mse:.2f}, R2 = {r2:.2f}")


    # In[ ]:


    import mlflow.models
    from mlflow.models.signature import infer_signature

    with mlflow.start_run(run_name="linear_regression_signature") as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        signature = infer_signature(X_train, y_pred) # this also works on Pandas DataFrames
        mlflow.sklearn.log_model(model, "model", signature=signature)
        print("Model with signature logged.")


    # In[ ]:





    # In[ ]:


    from mlflow import MlflowClient
    with mlflow.start_run(run_name="register_best_model") as run:

        # Retrieve Best Model Run
        client = MlflowClient()
        best_run_id = run.info.run_id

        # register the model
        model_uri = f"runs:/758f873092154a7e90ce88cf6766e4ff/best_model"
        registered_model = mlflow.register_model(model_uri, "california_housing_model")
        print(f"Model registered in model registry with name:{registered_model.name}, version: {registered_model.version}")


    # In[30]:


    import mlflow
    from mlflow.tracking import MlflowClient


    """
    Liste les expériences, les runs, sélectionne un run et affiche ses métadonnées.
    """
    client = MlflowClient()

    print("------------------- Experiments --------------------")
    experiments = client.search_experiments()
    for exp in experiments:
        print(f"Experiment Name: {exp.name}, ID: {exp.experiment_id}")

    experiments



    # In[32]:


    experiment_id = 985020413726465322
    print("--------------------- Runs -----------------------")
    all_runs = client.search_runs(experiment_id)
    if not all_runs:
        print("No runs found for the current experiment. Please run an MLflow experiment first.")

    for run in all_runs:
        print(f"Run ID: {run.info.run_name},; Run ID: {run.info.run_id}, Experiment ID: {run.info.experiment_id}")

    all_runs


    # In[33]:


    selected_run_id = input("Enter the run ID of the run you want to inspect :")

    # Rechercher le run sélectionné
    try:
        selected_run = client.get_run(selected_run_id)
    except Exception as e:
        print(f"Error: Run with ID '{selected_run_id}' not found or an error occured. Details: {e}")

        
    print("----------------- Selected Run Metadata ------------------")
    print(f"Run ID: {selected_run.info.run_id}")
    print(f"Experiment ID: {selected_run.info.experiment_id}")
    print(f"Start Time: {selected_run.info.start_time}")
    print(f"Status: {selected_run.info.status}")
    print("--------------------- Parameters -----------------------")
    for key, value in selected_run.data.params.items():
        print(f"  {key}: {value}")

    print("--------------------- Metrics -----------------------")
    for key, value in selected_run.data.metrics.items():
        print(f"  {key}: {value}")

    print("--------------------- Tags -----------------------")
    for key, value in selected_run.data.tags.items():
        print(f"  {key}: {value}")

    print("--------------------- Artifacts -----------------------")
    artifacts_list = client.list_artifacts(selected_run_id)
    for artifact in artifacts_list:
        print(f"  {artifact.path}")


    # In[35]:


    client = MlflowClient()

    # register the model
    model_uri = f"runs:/{selected_run_id}/model"
    registered_model = mlflow.register_model(model_uri, "auto_registred_california_housing_model")
    print(f"Model registered in model registry with name:{registered_model.name}, version: {registered_model.version}")


# In[ ]:


# Tache parcours les runs de l'expriment "Diabetes-3" pour comparer les metrics (les afficher , choisir le metric)
# enregistrer le model avec les meilleurs  metrics

