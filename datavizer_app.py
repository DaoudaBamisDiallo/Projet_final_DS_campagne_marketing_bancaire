# La page qui contient toutes les fonctions et code necessaire pour analyer et modéliser les données (les données nettoyées ) 
#-------------------importation des librairies necessaires-----------------
import streamlit as st
import pandas as pd
import os
import io
from utils_dataviz import showing_data,cat_vs_y_plot,num_vs_y_plot,bar_plot,hist_box_plot,resume_statistique,reporting_pdf
from utils_model import missing_data,spliter_data,verify_deséquilibre,resolution_desequilibre,normalizer,vars_importences,load_data_clean   
from utils_model import statistiques_outliers,cleaner_cateogrial_data ,model_evaluation,models,finding_outliers,resolving_outliers
#---------------------chemiin du repertoire de stockages des données---------------------------
outpout_db="outpu_dataset_db/data_modele"

#-----------------------fonction pour récuperer et combiner les 3 dataframes (économics , démographiques et personnelles)------------------------------

# ----------------------chargement de la base de données finale ---------------------
def load_data():
    try :
        # Diviser la dataframe en deux moitiés égales et prenons une partie pour l'analyse
    
        data = pd.read_csv("output_clean_data/data_bank_full.csv",delimiter=",")
        longueur = len(data)
        df_premiere_moitie = data.iloc[:longueur // 2]
        return df_premiere_moitie
    except Exception as e:
        st.warning(f"Une erreure s'est produit lors de chargement de la base de données nettoyées,assurez vous de bien effectué la phase d'analye {e}")
    # Bouton pour exécuter le processus de nettoyage et de division
#-------------------La fonction princiapal contenant tout le code de l'analyse et de la modélisation------------------------

def analysing_marketing():
    # description des données et définision de l'objectif de cette phase:
    st.markdown("""Les données utilisées dans cette partie sont celles  traitées lors de la phase de nettoyage.Elles sont Repartient en 3 fichiers Economiques, Démocraphique et Personnelles.
                
                L'objectif de cette partie est d'analyser les données de la campagne marketing ,comprendre leurs dependances et Créer un modéle de classification capable prédire si un client souscrira ou non à un prêt.""",unsafe_allow_html=True)
    
    #chargement du jeu de données
    df = load_data()
    # convertissons la colonne day_of_week en string pour faciliter la visualisation des jour du mois
    df.day_of_week=df.day_of_week.astype("object")
    # Récuperation des colonnes selon le type
    categorical_columns=df.select_dtypes(exclude="number").columns.tolist()
    numerical_columns=df.select_dtypes(exclude="object").columns.tolist()


    #-----------analyse exploratoire----------------------------------------
  
    menu_ade= st.sidebar.radio("Menu Principal",options=["Exploration","Traitement","Modélisation"],horizontal=True)
   
    if menu_ade=="Exploration":
        st.header("1: Analyse exploratoire des données")
        AED= st.sidebar.radio(label="",options=["Acceuil","Chargement","Univariée","Bivariée","Résumé statistique"],horizontal=False)
        # bouton de création d'un rapport d'analyse automatique de tous les graphiques
        if st.sidebar.button('Générer un Rapport',type="primary"):
            st.subheader("Rapport d'analyse synthetique")
            st.image("Reportory_pdf/logo_pret.jpg")
            reporting_pdf(df,categorical_columns,numerical_columns)
    
        # 1:affichage des données
        if AED=="Chargement":
            st.subheader("A: Chargement et description de la base de données")
            showing_data(df)
            
        # 2:-------------# Analyse Univariée-------------------------
        if AED=="Univariée":

            univaries=st.radio(label="Analyse Univariée",options=["Categorielle","Numérique"],key="univaries",horizontal=True)
            # analyse Visuelle de la distribution des variables Categorielles
            if univaries=="Categorielle":
                st.subheader("B: Analyse univariée des variables catetorielles")

                col = st.selectbox("Choisir la variables à visualiser",categorical_columns,key="cat")
                bar_plot(col,df,"non save")
            # analyse Visuelle de la distribution des numériques
            if univaries=="Numérique":
                st.subheader("C: Analyse univariée des variables Numériques")
                col = st.selectbox("Choisir la variables à visualiser",numerical_columns,key="num")
                hist_box_plot(col,df,"non save")
        #3:-----------analyse bivariées- entre les variables et la variable cible (y)--------------------
        if AED=="Bivariée":
            st.subheader("D: Analyse bivariée des variables VS à la variable cible (Y)")
            bivaries=st.radio(label="Analyse bivariée",options=["Categorielle VS Y","Numérique VS Y"],key="bivaries",horizontal=True)
            # Analyse visuelle de la distribution des variables Categoriques en selon la variables cible (y)
            if bivaries=="Categorielle VS Y":
                colx, coly = st.columns([0.7,0.3])
                with colx:
                    st.markdown("##### Les variables catetorielles VS la variable cible (Y)")
                with coly:
                    choces=st.radio(label="Choix du graphique",options=["En Bare","En Box"],key="bivaries_choice",horizontal=True)
                if choces=="En Box":
                    col = st.selectbox("Choisir la variables à visualiser",categorical_columns,key="catbivar")
                    num_vs_y_plot(col,df,"non save")
                    
                if choces=="En Bare":
                    col = st.selectbox("Choisir la variables à visualiser",categorical_columns,key="catbivar")
                    cat_vs_y_plot(col,df,"non save")
            
            # Analyse visuelle de la distribution des variables numériques en selon la variables cible (y)
            if bivaries=="Numérique VS Y":
                st.markdown("##### Analyse Bivariée des variables Numériques VS la variable cible (Y)")
                col = st.selectbox("Choisir la variables à visualiser",numerical_columns,key="numbivar")
                num_vs_y_plot(col,df,"non save")
        # 4: Analyse statistiques des variables
        if AED=="Résumé statistique":
            st.subheader("D: Résumé statistiques")

            resume_statistique(df)
    
#-----------------Traitement des données encodage et gestion des valeurs manquantes et abérantes------------------------------------
    data_cleaned=pd.DataFrame()
    x_train=data_cleaned
    if menu_ade=="Traitement":
        # gestion des valeurs manquantes et aberantes
        st.header("2: Traitement des données pour l'analyse")
        traitemaint =st.sidebar.radio(label="Traitement des données",options=["Verification","Encodage"])
        # verification et affichage des valeurs manquantes et aberantes
        if traitemaint=="Verification":   
            st.subheader("A: Verification des valeurs manquantes et abérantes")
            colm,cola = st.columns(2)
            # affichage des valeurs manquantes
            with colm :
                st.markdown('**Statistique des valeurs manquantes**')
                missing_data(df)
            # afichage des valeurs abérantes
            with cola:
                st.markdown('**Statistique des valeurs Extrêmes (Abérantes)**')
                statistiques_outliers(df,numerical_columns)
        # Prétraitement de données pour la modélisation
        if traitemaint=="Encodage":   
            # reconvertion de la colonne jours en entier pour  éviter d'avoir une pour tous les 31 jours du mois
            df.day_of_week=df.day_of_week.astype("int64")
            # recuperation des colonnes selon le types
            categorical_columns=df.select_dtypes(exclude="number").columns.tolist()
            numerical_columns=df.select_dtypes(exclude="object").columns.tolist()
            # encodage de variables categorielles
            data_clean_categorial= cleaner_cateogrial_data(df,categorical_columns)
            # Correction des valeurs abérantes 
            data_clean_numerical= resolving_outliers(df,numerical_columns,'IQR')
            # sauvegarde du dataframes nettoyée
            data_cleaned= pd.concat([data_clean_categorial,data_clean_numerical],axis=1)
            data_cleaned.to_csv(outpout_db+"/data_cleaned.csv")
            # affichage des données prétes pour la modélisation
            st.subheader("B: Données pour la modélisation")
            st.dataframe(data_cleaned)
            # encodeurs=True
#----------------Moodélisation des données-----------------------------------
    if menu_ade=="Modélisation":
            
            st.header('3: Modélisation du modèle')
            # divison des  données en entrainement , test et validation
            st.subheader('A):Division des données en Train, test et validation')
          
            # les options de choix des données d'entrainement: données originale, données sur-échantionnées et sous-échantionnées
            choice_trains=["Originale","Upsampled","Downsampled"]
           
            # chargement de la base de données nettoyées
            data_cleaned = pd.read_csv(outpout_db+"/data_cleaned.csv",delimiter=',')
           
            # division
            x_train,y_train, x_test,y_test, x_val,y_val,x_sampled=spliter_data(data_cleaned)
            st.toast('Division des données réussi')
           
            # vérification de la porportion des classes prédictes (souscrit et non souscrit ) de la variables y
            st.markdown('Proportion des classes non équilibrées')
            verify_deséquilibre(y_train=y_train,y_test=y_test,y_val=y_val,title_train="Donnée d'entraînement",title_test="Données entrainement",title_val="Donnée de test")
            
            # correction  du désequilibre de classe
            st.markdown('Proportion des classes équilibrées')
            y_train_up,y_train_down=resolution_desequilibre(x_sampled)
            verify_deséquilibre(y_train,y_train_up,y_train_down,"Donnée originelles","Données Sur-échantillonées","Donnée sous-échantillonées")
           
            
            # les differntes étapes de la modélisation 
            # paramètrage de la modélisation
            # 1:test des algorithmes
            algos = st.sidebar.selectbox(label="Choix d'Alogrorithme",options=["Regression Logistique","RandomForestClassifier","SVM","KNN","Tree"])
            # 2:normalisation des données
            scaler =st.sidebar.selectbox(label="Normalisation",options=["Scaler",'MinMax','Z-Score'])
            
            # 3:choix des données d'entrainement
            trains_data=  st.sidebar.selectbox(label="Choix des données d'entrainement",options=choice_trains)
            # chargement des données sur et sous-échantillonées
            upsampleds ,downsampleds= load_data_clean()
           
            # entrainer avec les données sur(échantillonées)
            if trains_data ==  "Upsampled":
                train_features=upsampleds.drop('y',axis=1)
                train_labels = upsampleds['y']
            # entrainer avec les données ous(échantillonées)
            if trains_data == "Downsampled":
                train_features=downsampleds.drop('y',axis=1)
                train_labels = downsampleds['y']
            # entrainer avec les données originele
            else:
                train_features=x_train
                train_labels = y_train
            # 4: entrainer le modele
            train_model = st.sidebar.button("Entrenainnement du modèle",type="primary")
            if train_model:
                 # Normalisation des données par MinMaxScaler ou StandarScale ou non
                if scaler is not "Scaler":
                    train_features = normalizer(train_features,"scaler",x_sampled)
                    x_val=normalizer(x_val,scaler,x_train)
                    x_test=normalizer(x_test,scaler,x_train)
                    st.success('Normalisation réussit')
                    st.subheader("B): hoix de l'Algorithme")
                # a: entrainer l'algorithme de Regression logistique
                if algos=="Regression Logistique":
                    st.subheader("Performances de Regression de Logistique")
                     # évaluation du modéle sur les données d'entrainement
                    st.markdown("Performance sur les données d'entrainement")
                    model = models(algo="Regression Logistique",train_x=train_features,train_y=train_labels)
                     # évaluation du modéle sur les données de test
                    model_evaluation(model,train_features,train_labels,"train")
                    st.markdown("Performance sur les données de test")
                    model_evaluation(model,x_test,y_test,"valide")
                # b: entrainer l'algorithme de forêt aléatoire
                if algos=="RandomForestClassifier":
                    st.subheader("Performances de RandomForestClassifier")
                     # évaluation du modéle sur les données d'entrainement
                    st.markdown("Performance sur les données d'entrainement")
                    model = models(algo="RandomForestClassifier",train_x=train_features,train_y=train_labels)
                    model_evaluation(model,train_features,train_labels,'train')
                     # évaluation du modéle sur les données de test
                    st.markdown("Performance sur les données de test")
                    model_evaluation(model,x_test,y_test,"valide")
                # c: entrainer l'algorithme de supportt vecteur machine
                if algos=="SVM":
                    st.subheader("Performance de Support Vecteur Machine")
                     # évaluation du modéle sur les données d'entrainement
                    st.markdown("Performance sur les données d'entraineme")
                    model = models(algo="SVM",train_x=train_features,train_y=train_labels)
                    model_evaluation(model,train_features,train_labels,"train")

                     # évaluation du modéle sur les données de test
                    st.markdown("Performance sur les données de test")
                    model_evaluation(model,x_test,y_test,"valide")
                # d: entrainer l'algorithme de K plus proche voisines
                if algos=="KNN":
                    st.subheader("Performances de KNN")
                     # évaluation du modéle sur les données d'entrainement
                    st.markdown("Performance sur les données d'entraineme")
                    model = models(algo="KNN",train_x=train_features,train_y=train_labels)
                    model_evaluation(model,train_features,train_labels,"train")
                     # évaluation du modéle sur les données de test
                    st.markdown("Performance sur les données de test")
                    model_evaluation(model,x_test,y_test,"valide")
                # a: entrainer  l'algorithme d'arbre de décision
                if algos=="Tree":
                    st.subheader("Performances d'arbre de decision")
                    # évaluation du modéle sur les données d'entrainement
                    st.markdown("Performance sur les données d'entraineme")
                    model = models(algo="Tree",train_x=train_features,train_y=train_labels)
                    model_evaluation(model,train_features,train_labels,"train")

                     # évaluation du modéle sur les données de test
                    st.markdown("Performance sur les données de test")
                    model_evaluation(model,x_test,y_test,"valide")
                   
    
