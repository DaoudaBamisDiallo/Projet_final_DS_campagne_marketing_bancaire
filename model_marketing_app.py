# def main():
#-------------------package------------------

# packages necessaires 1=V.1
import streamlit as st
import numpy as np
import pickle as pkl
import pandas as pd
import joblib 
from sklearn.preprocessing import MinMaxScaler


#-------------modelisation et deployement----------------------------
def intro():
    # description de l'application

    st.markdown(("Marketing Bank est une application conçue pour prédire la souscription ou non des clients à un prêt bancaire "))

#Fonction de lecture  du modele
def load_model():

    data = joblib.load("outpu_dataset_db/model_rf_end1.joblib")
    return data


# # fonction d'inference pour recuperer et predire le résultat
def interference(Balance, Age, Pdays, Campaign, Housing, Previous,  Marital_married, Month_Nov,Education_secondary, Month_may):
    
    # Préparation des données
    data = np.array([[Balance, Age, Pdays, Campaign, Housing, Previous,  Marital_married, Month_Nov,Education_secondary, Month_may]]).T
    
    # Normalisation des données
    scaler = MinMaxScaler()
    df = scaler.fit_transform(data).T
    # st.write(df,df.T)

    # Prédiction
    # chargement du modele
    model_marketing = load_model()
    
    pret_bank = model_marketing.predict(df)
    return pret_bank


# saisie des iinformations du patience
def prediction_client():  
    intro()   
    st.subheader("Information du client")
    # partage de la page en deux colonnes
    col1,col2 = st.columns(2)
    with col1 : 
        
        Balance = st.number_input(label="Solde moyen annuel",min_value=-4057,value=561)
        Age = st.number_input(label="L'age ",min_value=15,value=48)
        col_mari, coleduc = st.columns(2)
        with col_mari:
            Marital = st.radio(label="Situation Matrimonial?",options=["Marrier","Pas Marrier"], key="Marrier",horizontal=True)
            Marital_married = 1
            if Marital !="Marrier":
                Marital_married = 0
        with coleduc:
            Education_2_quiz = st.radio(label="Avez-vous  Atteint le niveau Secondaire?",options=["Yes","Non"], key="second",horizontal=True)
            Education_secondary=1
            if Education_2_quiz=="Non":
                Education_secondary=0

        Housing_qiuiz= st.radio(label="Avez-vous un prêt immobilier",options=["Yes","Non"], key="Housing",horizontal=True)
        Housing =1
        if Housing_qiuiz=="Non":
            Housing =0
    
        # st.write(Job_management,Job_blue_collar,Job_technician)
    # deuxieme colonne
    with col2 :
        # Les dernier mois de contact avec le client
        Months = st.multiselect(label="Les Derniers Mois de contact Avec le Client",options=["Mai", "Nov", "Aucun"])
        Month_Nov =0
        Month_may= 0

        
        if "Mai" in Months:
            Month_may=1
        
        if "Novembre" in Months:
            Month_Nov=1
        
        # La fequence de contact avec le client
        Pdays = st.number_input(label="Nombre jours écoulés depuis le dernier contact avec le client dans une campagne précédente",min_value=-1,value=-1)
        
        Previous = st.number_input(label="Nombre de contacts effectués avant cette campagne pour ce client",min_value=0,value=0)
        Campaign = st.number_input(label="Nombre de contacts effectués durant cette campagne pour ce client (inclut le dernier contact)",min_value=0, value=2)

        
    prevoir = st.button('Allez-vous souscrire  un prêt à terme?',type="primary")   
    if prevoir:
        
        resultat = interference(Balance, Age, Pdays, Campaign, Housing, Previous,  Marital_married, Month_Nov,Education_secondary, Month_may)
        if resultat[0] == 1:
            st.success("Oui j'ai besoin d'un prêt")
        elif resultat[0] == 0:
            st.warning("Désolé je suis pas en mesure  de prendre un prêt pour l'instant")
        else :
            st.error("Le résultat de l'examen inconnu merci de bien saisir les information ou de consulter le médecin pour plus de detail")
