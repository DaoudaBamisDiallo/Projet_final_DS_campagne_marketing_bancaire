
# Liste des fonctions utilisées pour nettoyagesret et structurer les données brutes
# inportation des librairies
import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict
import pickle



# fonction de télechargement du dictionnaire de la base de données
def upload_save_meta():
    # fonction pour  télécharger le dictionnaire à partir du site
    data = fetch_ucirepo(id=222) 
    # Sauvegarder l'objet dotdict dans un fichier
    with open('input_raw_data/ucimlrepo.dotdict.dotdict.pkl', 'wb') as f:
        pickle.dump(data, f)

# fonction pour diviser les données en deux partier
def load_and_split_meta():
    # chargement du dictionnaire contenant la base de données
    with open('input_raw_data/ucimlrepo.dotdict.dotdict.pkl', 'rb') as f:
                    uploaded_file = pickle.load(f)
    
    # combine les Features et y la variables cible
    df = pd.concat([uploaded_file.data.features, uploaded_file.data.targets],axis=1)

    # Obtenir la longueur de la dataframe
    longueur = len(df)

    # Diviser la dataframe en deux moitiés égales
    df_premiere_moitie = df.iloc[:longueur // 2]
    df_deuxieme_moitie = df.iloc[longueur // 2:]

    # Affichage des dimensions
    # Creation du dossier des données brutes
    input_dir = "input_raw_data"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # Sauvegarder la première moitié dans un fichier CSV
    input_premiere_moitie_path = os.path.join(input_dir, 'data_bank_marketing.csv')

    df_premiere_moitie.to_csv(input_premiere_moitie_path, index=False)

    # Sauvegarder la deuxième moitié dans un fichier CSV
    input_deuxieme_moitie_path = os.path.join(input_dir, 'new_data_bank_marketing.csv')

    df_deuxieme_moitie.to_csv(input_deuxieme_moitie_path, index=False)

    # les deux dataframes sont disponible pour des utilisations ultérieures: pour facilité le chargement nous allons utiliser la moitié des données
    return df_deuxieme_moitie

# --------------Fonction de supressions de la colonnnes "poutcome" drop_poutcome_column

def drop_poutcome_column(df):
    """
    Supprime la colonne 'poutcome' de la DataFrame.
    """
    try:
        df_copy = df.copy()
        df_copy = df_copy.drop('poutcome', axis=1)
        return df_copy
    except KeyError as e:
        raise KeyError("La colonne 'poutcome' est absente de la DataFrame :", e)
    except Exception as e:
        raise Exception("Une erreur s'est produite lors de la suppression de la colonne 'poutcome' :", e)
    
# Description du role de la fonction
# '''
# Objectif : Supprimer la colonne 'poutcome' de la DataFrame.

# Retourne : Une copie de la DataFrame sans la colonne 'poutcome'
# '''
# --------------------Fonction imputer_valeurs_manquantes

def imputer_valeurs_manquantes(dataframe, variables_categorielles):
    """
    Impute les valeurs manquantes dans les variables catégorielles d'une DataFrame.
    """
    try:
        dataframe_copy = dataframe.copy()
        for variable in variables_categorielles:
            dataframe_copy[variable] = dataframe_copy[variable].fillna('unknown')
        return dataframe_copy
    except Exception as e:
        raise Exception("Une erreur s'est produite lors de l'imputation des valeurs manquantes :", e)
# Description du role de la fonction

# '''
# Objectif : Remplacer les valeurs manquantes dans les colonnes catégorielles par 'unknown'.

# Retourne : Une copie de la DataFrame avec les valeurs manquantes imputées.
# '''

# -------------------Fonction de notttoyge des donnéesclean_and_split_data

def clean_and_split_data(marketing_data):
    """
    Nettoie et divise les données marketing en trois DataFrames distinctes : clients, campagnes et economics.
    """
    try:
        marketing_data_clean1 = drop_poutcome_column(marketing_data)
        marketing_data_clean2 = imputer_valeurs_manquantes(
            marketing_data_clean1, 
            ['job', 'education', 'contact']
        )

        demographic_columns = ['age', 'job', 'marital', 'education']
        clients = marketing_data_clean2[demographic_columns]

        colonnes_campagnes = ['contact', 'day_of_week', 'month', 
                              'duration', 'campaign', 'pdays', 
                              'previous', 'y']
        campagnes = marketing_data_clean2[colonnes_campagnes]

        colonnes_economics = [
            c for c in marketing_data_clean2.columns
            if c not in clients.columns
            and c not in campagnes.columns
        ]
        economics = marketing_data_clean2[colonnes_economics]

        return clients, campagnes, economics

    except KeyError as e:
        raise KeyError("Certaines colonnes nécessaires sont absentes de la DataFrame :", e)
    except Exception as e:
        raise Exception("Une erreur s'est produite lors du nettoyage et de la division des données :", e)
# '''
# Objectif : Nettoyer les données et les diviser en trois DataFrames : clients, campagnes, et economics.
# '''

# -------------------Fonction de chargement de données

def load_to_csv(marketing_data):
    clients_df, campagnes_df, economics_df = clean_and_split_data(marketing_data)
    output_dir = "output_clean_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_clients_path = os.path.join(output_dir, "clients.csv")
    output_campagnes_path = os.path.join(output_dir, "campagnes.csv")
    output_economics_path = os.path.join(output_dir, "economics.csv")

    # On écrase les anciens fichiers et on les remplace par les nouvelles données
    clients_df.to_csv(output_clients_path, index=False)
    campagnes_df.to_csv(output_campagnes_path, index=False)
    economics_df.to_csv(output_economics_path, index=False)
# description du role de la fonction
# '''
# Objectif : Nettoyer et diviser les données, puis sauvegarder les DataFrames résultantes en fichiers CSV.

# '''


    # recuperation des features des labels