
# La page qui contient toutes les fonction et le code neccessaire pour nettoyer et structurer les données
def cleaning_brute_data():
    # librairies necessaires
    import streamlit as st
    import pandas as pd
    import os
    import io
    # from ucimlrepo import fetch_ucirepo 
    from utils_cleaner import load_to_csv,load_and_split_meta
    import pickle
   
    # description de la base données et définition de l'objectif de cette phase:
    
    st.markdown("""La Base données utilisée est liées aux campagnes de marketing direct ( par appels téléphoniques) d'une institution bancaire portugaise, disponible dans le site **UCI Irvine Machine Mearning Repository** <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing" target="_blank">Source de données</a>.
                        
            L'objectif de cette partie est de nottoyer et structurer les données , les préparer pour la phase d'analyse.""",unsafe_allow_html=True)
        # Affichage d'un extrait de la DataFrame et de sa structure côte à côte Importation des données (Source : )
    
    # Téléchargement du fichier des données marketing
    if st.sidebar.button('Télechargé les données',type="primary"):
        try:
            marketing_data = load_and_split_meta()
            with open('input_raw_data/ucimlrepo.dotdict.dotdict.pkl', 'rb') as f:
                 uploaded_file = pickle.load(f)

            st.success('Chargement des données réussit')
        except e:
            st.warning(f"Une eurreur s'est produit lors du chargement des données{e}")
        # uploaded_file = fetch_ucirepo(id=222) 
        #telechargement des données

    # if uploaded_file is not None:
        # bank_marketing=uploaded_file
        # data_bank_marketing = pd.concat([bank_marketing.data.features, bank_marketing.data.targets],axis=1)
        # data_bank_marketing = df_moitie 
        
        
            # Lecture du fichier en DataFrame
       
        st.header("1: Aperçu et structure des données Brute:")

        col1, col2 = st.columns(2)
        with col1:
            st.write(marketing_data.head())
            st.write(f"Dimensions des données : {marketing_data.shape}")
            st.write(f"Notre jeu de données est composé de :   {marketing_data.shape[0]} Observations et {marketing_data.shape[1]} caracteristiques , dont {len(marketing_data.select_dtypes(exclude='number').columns.tolist())} de type categorielle(s) et {len(marketing_data.select_dtypes(exclude='O').columns.tolist())} de type numérique (s)")

        # st.markdoSwn("resume de data frame")
        with col2:
            buffer = io.StringIO()
            marketing_data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
    # Informations générales et régles de traitement des données brutes
    if st.sidebar.button("Documentation de la DB",type="primary"):
        # Chargement du  doction de données contenant toutes les information de la base de données
        with open('input_raw_data/ucimlrepo.dotdict.dotdict.pkl', 'rb') as f:
            uploaded_file = pickle.load(f)
        # fonction pour  télécharger le dictionnaire à partir du site
        # uploaded_file = fetch_ucirepo(id=222) 
    
        if uploaded_file is not None:
            st.write(f" ### 2: Documentation la la base données")
            st.markdown("""**Tableau description de toutes les variables: Nom, Type, Rôle, Caractère,Unité et valeurs manquante....**""")
            st.write(uploaded_file.variables)
            st.markdown("""
                                
                #### Voici la description de chaque variable basée sur le résultat du code précédent :

                ### 1. age

                - Role: Feature
                - Type: Integer
                - Demographic: Age
                - Description: Âge du client
                - Units:
                - Missing Values: no
                ### 2. job

                - Role: Feature
                - Type: Categorical
                - Demographic: Occupation
                - Description: Type d'emploi (catégories : 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
                - Units:
                - Missing Values: yes
                ### 3. marital

                - Role: Feature
                - Type: Categorical
                - Demographic: Marital Status
                - Description: État civil (catégories : 'divorced','married','single','unknown'; note: 'divorced' inclut les divorcés et les veufs)
                - Units:
                - Missing Values: no
                ### 4: education

                - Role: Feature
                - Type: Categorical
                - Demographic: Education Level
                - Description: Niveau d'éducation (catégories : 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
                - Units:
                - Missing Values: yes
                ### 5: default

                - Role: Feature
                T- ype: Binary
                - Demographic:
                - Description: Le client a-t-il un crédit en défaut ?
                - Units:
                - Missing Values: no
                ### 6: balance

                - Role: Feature
                - Type: Integer
                - Demographic:
                - Description: Solde moyen annuel
                - Units: euros
                - Missing Values: no
                ### 7: housing

                - Role: Feature
                - Type: Binary
                - Demographic:
                - Description: Le client a-t-il un prêt immobilier ?
                - Units:
                - Missing Values: no
                ### 8: loan

                -Role: Feature
                - Type: Binary
                - Demographic:
                - Description: Le client a-t-il un prêt personnel ?
                - Units:
                - Missing Values: no
                ### 9: contact

                - Role: Feature
                - Type: Categorical
                - Demographic:
                - Description: Type de communication de contact (catégories : 'cellular','telephone')
                - Units:
                - Missing Values: yes
                ### 10: day_of_week

                - Role: Feature
                - Type: Date
                - Demographic:
                - Description: Jour de la semaine du dernier contact
                - Units:
                - Missing Values: no
                ### 11: month

                - Role: Feature
                - Type: Date
                - Demographic:
                - Description: Mois du dernier contact (catégories : 'jan', 'feb', 'mar', ..., 'nov', 'dec')
                - Units:
                - Missing Values: no
                ### 12: duration

                - Role: Feature
                - Type: Integer
                - Demographic:
                - Description: Durée du dernier contact en secondes. Note importante : cette variable affecte fortement la cible. Par exemple, si duration=0 alors y='no'. Cependant, la durée n'est pas connue avant l'appel. Après l'appel, y est évidemment connu. Donc, cette variable doit être incluse uniquement à des fins de benchmark et doit être exclue pour un modèle prédictif réaliste.
                - Units: seconds
                - Missing Values: no
                ### 13: campaign

                - Role: Feature
                - Type: Integer
                - Demographic:
                - Description: Nombre de contacts effectués durant cette campagne pour ce client (inclut le dernier contact)
                - Units:
                - Missing Values: no
                ### 14: pdays

                - Role: Feature
                - Type: Integer
                - Demographic:
                - Description: Nombre de jours écoulés depuis le dernier contact avec le client dans une campagne précédente (-1 signifie que le client n'a pas été contacté auparavant)
                - Units:
                - Missing Values: yes
                ### 15: previous

                - Role: Feature
                - Type: Integer
                - Demographic:
                - Description: Nombre de contacts effectués avant cette campagne pour ce client
                - Units:
                - Missing Values: no
                ### 16: poutcome

                - Role: Feature
                - Type: Categorical
                - Demographic:
                - Description: Résultat de la campagne marketing précédente (catégories : 'failure','nonexistent','success')
                - Units:
                - Missing Values: yes
                ### 17: y

                - Role: Target
                - Type: Binary
                - Demographic:
                - Description: Le client a-t-il souscrit un dépôt à terme ?
                - Units:
                - Missing Values: no

                """)
        # Bouton pour exécuter le processus de nettoyage et de division
   
    # Nettoyage et division des données en : données économiques,démographiques et personnelles(client)
    if st.sidebar.button("Nettoyer et Structurer",type="primary"):
        try:
                # lecture du dictionnaires de la base de données:
                #Méthode 1ere:
                # uploaded_file = fetch_ucirepo(id=222)
                # Methode 2: le dictionaires est deja savegarder
                with open('input_raw_data/ucimlrepo.dotdict.dotdict.pkl', 'rb') as f:
                    uploaded_file = pickle.load(f)
                # recuperation des features et la targets (y)
                marketing_data = pd.concat([uploaded_file.data.features, uploaded_file.data.targets],axis=1)

                load_to_csv(marketing_data)

                output_dir = "output_clean_data"
                output_clients_path = os.path.join(output_dir, "clients.csv")
                output_campagnes_path = os.path.join(output_dir, "campagnes.csv")
                output_economics_path = os.path.join(output_dir, "economics.csv")
                output_df_full_path = os.path.join(output_dir, "data_bank_full.csv")

                # Fonction pour afficher DataFrame et info côte à côte
                def display_dataframe_info(df, title):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(title)
                        st.write(df.head())
                        st.write(f"Dimensions : {df.shape}")
                        st.write(f"Les données du  {title} est composé de :   {df.shape[0]} Observations et {df.shape[1]} caracteristiques , dont {len(df.select_dtypes(exclude='number').columns.tolist())} de type categorielle(s) et {len(df.select_dtypes(exclude='O').columns.tolist())} de type numérique (s)")

                    with col2:
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        s = buffer.getvalue()
                        st.text(s)

                # Affichage des trois DataFrames résultantes
                # Affichage des dimensions de la DataFrame
                st.write(f"### 3: Affichage et Division des données: économiques, demographique et personnelle")

                st.header(" A: DataFrame Clients et sa structure :")
                display_dataframe_info(pd.read_csv(output_clients_path), "DataFrame Clients :")

                st.header(" B: DataFrame Campagnes et sa structure :")
                display_dataframe_info(pd.read_csv(output_campagnes_path), "DataFrame Campagnes :")

                st.header("C: DataFrame Economics et sa structure :")
                display_dataframe_info(pd.read_csv(output_economics_path), "DataFrame Economics :")

                df_full = pd.concat([pd.read_csv(output_clients_path),pd.read_csv(output_economics_path),pd.read_csv(output_campagnes_path)],axis=1)
                df_full.to_csv(output_df_full_path,index=False)
                st.write(df_full.shape)
                st.success("Les fichiers CSV ont été sauvegardés avec succès!")
        except Exception as e:
                st.error(f"Une erreur s'est produite veillez vous s'assurez que les données ont étè bien chargées: {e}")