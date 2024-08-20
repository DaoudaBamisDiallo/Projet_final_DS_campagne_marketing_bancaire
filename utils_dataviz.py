
# packages necessaires
import streamlit as st
import numpy as np
# import  joblib as joblib
import pandas as pd
from scipy import stats
from PyPDF2 import PdfMerger
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
# from ipywidgets import interact
import os
import io
#--------evaluation de performance--------
from  sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler


# fonction de chagement de la base de données

# @st.cache_data(persist=True)

#--------------------------------------head et info------------------------------
#fonction d'affichage des données et leurs informations générales
def showing_data(data):  
    # division de la page en deux partie
    col,col2 = st.columns([0.6,0.4])
    # affichage des  100 premiere observation
    with col:
            
        df_sample = data.sample(100)
        st.markdown("**Jeu de données de compagne marketing  : Echantillons de 100 observations**")
        st.write(df_sample)
            # Affichage des dimensions de la DataFrame
        st.write(f"Notre jeu de données est composé de :   {data.shape[0]} Observations et {data.shape[1]} caracteristiques , dont {len(data.select_dtypes(exclude='number').columns.tolist())} de type categorielles.\n Apparement pas de valeurs manquantes ")

    # affichage des information générales de la base de données
    with col2 :
        st.markdown("**Information générales de la base de données**")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)


# ------------------------------analyse exploratoire des données-----------------------


#-------fonction d'analyse univariée-----------
def bar_plot(a,data,saving_fig):

    # Créer le graphique avec Matplotlib et Seaborn
    plt.figure(figsize=(14, 6))
    var = data[a].value_counts()
    g = sns.barplot(x=var.index, y=var.values, palette='Set2', color='#abc9ea')

    plt.ylabel("Proportion")
    plt.title("Distribution de la variable " + str(a))
    plt.xticks(rotation=30)
    # Ajouter les labels
    for p in g.patches:
        height = p.get_height()
        label = f'{(height / len(data[a]) * 100):.2f} %'
        g.annotate(label, (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Afficher le graphique dans Streamlit
   
    
    if saving_fig=="save":
         st.pyplot(plt)
         plt.savefig(f"Reportory_pdf/rapporting_{str(a)}_marketing_bank.pdf",orientation="landscape")
    else:
        st.pyplot(plt)
    plt.close()
#-----------------------------------------------------------------

# Création d'une fonction d'analyse univariéé de variables 
def hist_box_plot(b,data,saving_fig):
  
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  fig.suptitle('Distribution de '+str(b) + " Coef ASy "+str(data[b].skew().round(3)), fontsize=12, y=0.95) 
  # histogramme des variables
  g1 = sns.histplot(x=data[b] ,palette='Set2', color='#abc9ea',ax=axes[0])
  g1.set_title('Histogram ')
  g1.set_xlabel("")
  # boxplot des variables
  g2 = sns.boxplot(y=data[b] ,palette='husl', color='#abc6ea',ax=axes[1], gap=.1)
  g2.set_title('Boxplot')
  fig.supxlabel(b)

  if saving_fig=="save":
         st.pyplot(fig)
         plt.savefig(f"Reportory_pdf/rapporting_{str(b)}_marketing_bank.pdf",orientation="landscape")
  else:
        st.pyplot(fig)
  plt.close()
# ------------------------------------------------------------------------
# creation d'une fonction d'analyse bivariée (les variables vs Y )
#---------les variables nuleriques et la variables cible y
def num_vs_y_plot(a,data,saving_fig):
       fig = plt.figure(figsize=(14, 6))
       ax = sns.boxplot(y=data[a],x=data['y'],palette='Set2', color='#abc9ea')
       ax.set_title(f"Distibution de prêt en fonction de {a} ")
       ax.set_xlabel("Prêt")
       ax.set_ylabel(f"Proportion de {a}")
       if saving_fig=="save":
         st.pyplot(fig)
         plt.savefig(f"Reportory_pdf/rapporting_{str(a)}_vs_y_marketing_bank.pdf",orientation="landscape")
       else:
        st.pyplot(fig)
       plt.close()
      

#---------les variables categorique et la variables cible
def cat_vs_y_plot(a,data,saving_fig):
    plt.figure(figsize=(14, 6))
    var = data[a]
    g = sns.countplot(x=var,hue=data["y"],palette='Set2', color='#abc9ea')

    plt.ylabel("Proportion")
    plt.title("Distribution de la variable " + str(a))
    plt.xticks(rotation=30)

    if saving_fig=="save":
         st.pyplot(plt)
         plt.savefig(f"Reportory_pdf/rapporting_{str(a)}_vs_y_marketing_bank.pdf",orientation="landscape")
    else:
        st.pyplot(plt)
    plt.close()
#---------------------------------------resumé statistique---------------------------------
# creation d'une fonction d'analyse miltivariée (les variables et variables VS Y )

def resume_statistique(data):
    colc,coln = st.columns(2)
    with coln:
        st.write("Résumé statiqtique des variables Numériques")
        st.dataframe(data.describe())
    with colc:
        st.write("Résumé statiqtique des variables Categoriques")
        st.write(data.describe(exclude="number").T)
    
#---------------------- 
def reporting_pdf(data,categorials,numericals):
    # creation du dossier_stockages des fichier pdf
    repertory_pdf = "Reportory_pdf"
    # lectures du nom des fichiers
    
    if not os.path.exists(repertory_pdf):
        os.makedirs(repertory_pdf)
    # raport des données categorielles
    files_pdf=[]
    # rapport données numeriques vs y
    for col in numericals:
        paths=f"Reportory_pdf/rapporting_{str(col)}_vs_y_marketing_bank.pdf"
        files_pdf.append(paths)
        num_vs_y_plot(col,data,"save")
    
     # raport des données categorielles vs à y()
    for col in categorials:

        paths=f"Reportory_pdf/rapporting_{str(col)}_vs_y_marketing_bank.pdf"
        files_pdf.append(paths)
       
        cat_vs_y_plot(col,data,"save")
    

    # raport des données numerique
    for col in numericals:
        paths=f"Reportory_pdf/rapporting_{str(col)}_marketing_bank.pdf"
        files_pdf.append(paths)
        hist_box_plot(col,data,"save")
    
   
    for col in categorials:
        paths=f"Reportory_pdf/rapporting_{str(col)}_marketing_bank.pdf"
        files_pdf.append(paths)
        bar_plot(col,data,"save")   
   
    merger = PdfMerger()
    for pdf in files_pdf:
        merger.append(pdf)
    return merger.write('marketing_bank_dataViz_repporting.pdf')
