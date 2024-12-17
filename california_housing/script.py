
import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == "__main__":
   
 
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(MLFLOW_TRACKING_URI)
 
    # Initialiser l'expérience MLflow
    mlflow.set_experiment("california_housing_regression")

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
