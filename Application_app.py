
#---------Page principale contenant les pages de nettoyage, d'analyse et de deploiement----------------
# importation des librairie nécessaires
import streamlit as st
from streamlit_option_menu import option_menu
from Cleaner_marketing_app import cleaning_brute_data

from model_marketing_app import prediction_client
from datavizer_app import analysing_marketing



# Configuration de la page
#----------header de la page--------------------------------------
st.set_page_config(page_title="Campagne marketing bancaire", page_icon="📊")
title_page="Application de compagne marketing de prêt bancaire"
st.title(title_page)
st.markdown("###### Réalisée par : Diallo Daouda DG de AI-Data_Consulting Group")

#les images d'acueille
colclna,colana,colpred= st.columns(3)
with colclna:
    st.image("images_bank/my_cleaning.jpg",width=300)
with colana:
    st.image("images_bank/logo_pret1.jpg",width=300)
with colpred:
    st.image("images_bank/marketing_banke1.jpg",width=300)

#------------------ Configiration du menu de la page--------------

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Nettoyage", "Analyse", "Prédiction"],
        icons=["house", "bar-chart", "gear","data"],
         default_index=0)
    
#--entete des 'autres pages
def intro_app(titre,logo):
    col_title, col_img = st.columns([0.7,0.3])
    with col_title:
        
        st.title(titre)
    with col_img:
        st.image(logo,width=200)
 #--------------------body de la  
# Contenu des page de nettoyage
if selected == "Nettoyage":
    try:
        # st.write(f"#### Nettoyage et Division des Données Marketing")
        intro_app("Nettoyage et Division des Données Marketing","images_bank/my_cleaning.jpg")
        cleaning_brute_data()

    except Exception as e:
        st.warning(f"Désolé une erreur s'est produit {e} réactualisez l'appli et réessayez")

 # Contenu des page d'analyse   
elif selected == "Analyse":
    # st.write(f"#### Analyse des Données Marketing")
    # try:
        intro_app("Analyse des Données Marketing","images_bank/logo_pret1.jpg")
        analysing_marketing()
    # except Exception as e:
    #     st.warning("Désolé une erreur s'est produit réactualisez l'appli et réessayez")

# contenue de la page de prédiction
elif selected == "Prédiction":
    # st.write(f"#### _Bienvenue à Marketing Bank")
    intro_app("Bienvenue à Marketing Bank","images_bank/marketing_banke1.jpg")
    prediction_client()
    
# configuration du pied de la pafge
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgb(240, 242, 246);
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #555;
    }
    </style>
    <div class="footer">
        <p>© 2024 IA_Data_Consulting group .Tous droits réservés. Contact : +225-05-76-17-82-40. Email: daoubamis66@gmail.com</p>
        <p><a href="https://www.example.com" target="_blank">Visitez notre site web</a></p>
    </div>
    """,
    unsafe_allow_html=True)