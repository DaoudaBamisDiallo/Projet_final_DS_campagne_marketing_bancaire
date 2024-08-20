# librairie necessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix,RocCurveDisplay,roc_curve,auc,f1_score,accuracy_score,recall_score,precision_score
from sklearn.feature_selection import RFE
import streamlit as st

#chemin de stockage des models
outpout_db="outpu_dataset_db/data_modele"

#------------------------------valeurs manquantes---------------------------
# fonction pour afficher les valeurs manquantes
def missing_data(data):
    total= len(data)
    nobre = data.isna().sum()
    percen = round((data.isna().sum()/total)*100,2)
    tab_missings=pd.concat([nobre,percen],axis=1,keys=["Nombre","Pourcentage(%)"])
    st.write(tab_missings)

# fonctions de detection des valeurs aberantes par la methode de IQR

#----borne inferieur lowers
def lowers(data,var_name):
    return data[var_name].quantile(0.25) - 1.5*stats.iqr(data[var_name])

#----borne superieur uppers
def uppers(data,var_name):
    return data[var_name].quantile(0.75) + 1.5*stats.iqr(data[var_name])
# outliers
def finding_outliers(data,var_name):
    outliers=data[(data[var_name]< lowers(data,var_name)) | (data[var_name] > uppers(data,var_name)) ]
    return outliers

# statistique des valeurs aberantes

def statistiques_outliers(data,colums):
    #Verification des valeurs aberantes
    # creation du dataframe
    outliers_stat=pd.DataFrame()
    # creation des listes pour stocker chaque unité
    variables=[]
    total=[]
    nbr=[]
    percen=[]
    # parcours et calcul de chaque variables
    for col in colums:
        variables.append(col)
        total.append(len(data))
        nbr.append(len(finding_outliers(data,col)))
        percen.append(round((len(finding_outliers(data,col))/len(data))*100,2))

    # ajoutes des valeurs dans le dataframe 
    outliers_stat["Variables"]=variables
    outliers_stat["Total"]=total
    outliers_stat["Nombre"]=nbr
    outliers_stat["Pourcentage (%)"]=percen
    # affichages
    
    st.write(outliers_stat)


# fonction de correction de valeurs aberrantes
def resolving_outliers(data,vars,strategie):
    
    for var in vars:

            if strategie=="IQR":
                data.loc[(finding_outliers(data,var).index,var)]=data.loc[(finding_outliers(data,var).index,var)].apply(
                    lambda x : lowers(data,var) if x < lowers(data,var) else uppers(data,var))
                
            elif strategie=="mediane":
                data.loc[(finding_outliers(data,var).index,var)]=data[var].median()
            else:
                print("veillez selection une strategie d'imputation")

    return data[vars]

# detection des valeurs aberantes

# fonction de 


#---encodage des variables categorielles-------------------
def cleaner_cateogrial_data(data,c):
    # récuperation des variables binaire et non binaire
    # transformation des variables multi modalité en des fonctionnalité
   
    col_not_binary=data[c].nunique()[data[c].nunique()>2].index.tolist()
    col_binary = data[c].nunique()[data[c].nunique()<3].index.tolist()
    
    # encodage des variables non  binaires
    df_not_binary =pd.get_dummies(data=data[col_not_binary], columns=col_not_binary, drop_first=True).astype(int)

    df_binary=data[col_binary]
    # encodage des variables binaire
    for colum in col_binary:

        df_binary[colum] = data[colum].apply(lambda val : 0 if val=="no" else (1 if val=="yes" else c) )
    
    # concatenation des deux dataframe
    df_categorial_clean=pd.concat([df_not_binary,df_binary],axis=1)
    
    # retour la nouvelle dataframes
    return df_categorial_clean




# # division des donnnes e train test et validation
def spliter_data(data,test_size=0.4):
   
    # vision des données
    x=data.drop('duration',axis=1)
    y=data['y']
    seed=95
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=seed,stratify=y)

    x_val,x_test,y_val,y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=95,stratify=y_test)
    data_train = pd.concat([x_train,y_train],axis=1)
    data_test= pd.concat([x_test,y_test],axis=1)
    data_val= pd.concat([x_val,y_val],axis=1)
    
    # savegarde des données
    data_train.to_csv(outpout_db+"/train_data.csv",index=False)
    data_test.to_csv(outpout_db+"/test_data.csv",index=False)
    data_val.to_csv(outpout_db+"/val_data.csv",index=False)
    return x_train,y_train, x_test,y_test, x_val,y_val,x

 # fonction de verification des lequilibre des classes (1 et 0)
def verify_deséquilibre(y_train,y_test,y_val,title_train,title_test,title_val):
    # Création de la grille d'axes
    fig, axes = plt.subplots(1, 3, figsize=(10 ,6))  # 1 ligne, 3 colonnes
    # Données pour les pie charts
    train = y_train.value_counts(normalize=True).sort_values(ascending=True)
    test = y_test.value_counts(normalize=True).sort_values(ascending=True)
    val = y_val.value_counts(normalize=True).sort_values(ascending=True)
    X=["Souscrit","Non Souscrit"]


    # Création des pie charts
    colors = plt.get_cmap('Greens')(np.linspace(0.2, 0.7, len(train)))
    explode = [0,0.1]
    axes[0].pie(train,  autopct='%1.1f%%', labels=X ,textprops={'fontsize': 8},explode=explode,radius=1,shadow=1)
    axes[0].set_title(title_train)  # Ajout du titre au données de d'entrainement

    axes[1].pie(test, autopct='%1.1f%%', labels=X,textprops={'fontsize': 8},explode=explode,radius=1,shadow=1)
    axes[1].set_title(title_test)  # Ajout du titre au données de test

    axes[2].pie(val, autopct='%1.1f%%', labels=X,textprops={'fontsize': 8},explode=explode,radius=1,shadow=1)
    axes[2].set_title(title_val)  # Ajout du titre au données de validation

    st.pyplot(fig)

# resolution desequilibre des classes
# resolution desequilibre des classes
def resolution_desequilibre(features):
        # Résolution du probleme de désequilibre de classe : Methode de Sur-échantillonage
        x2 = features
        # st.write(x2)
        x2['y']= features.values
        minority=x2[x2['y']==1]
        majority=x2[x2['y']==0]
        # # mehode de sur-echantillonnage : resample de sklearn (tirage avec remise)
        minority_upsample=resample(minority,n_samples= len(majority), replace = True,random_state=95)
        upsampled = pd.concat([majority,minority_upsample])
        # # ajouje des données dentrainement surechantionnées
       
        # # Résolution du probleme de désequilibre de classe : Methode de Sous-échantillonage
        # # mehode de sur-echantillonnage : resample de sklearn (tirage avec remise replace=False)
        majority_downsample=resample(majority,n_samples= len(minority), replace = True,random_state=95)
        downsampled = pd.concat([minority,majority_downsample])
    
      
        # #savegarde des données echantillonées
        upsampled.to_csv(outpout_db+"/upsamled_train_data.csv",index=False)
        downsampled.to_csv(outpout_db+"/dowsampled_train_data.csv", index=False)
        y_train_up=upsampled['y']
        y_train_down = downsampled['y']

        #retourn les fetaures sur et sous echantillonées
        return y_train_up,y_train_down
     

# fonction de normalisation des donnes

def normalizer(data,scaler,x):
    #MinMaxscaler
    scaler = MinMaxScaler()
    
    # Normalisation Z-Score
    if scaler=="Z-Score":
        scaler = StandardScaler()
    # transformation en dataframe
    data=pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data
# selction des  meilleurs variables

def vars_importences(features,labels,seuil=0.018):
    rf = RandomForestClassifier()
    rf.fit(features,labels)
    # Importance des variables indépendantes
    vars_imp = pd.Series(rf.feature_importances_, index = features.columns).sort_values(ascending=False)

    # Variables sélectionnées pour les algorithmes
    vars_selected = vars_imp[vars_imp > seuil]
# 
    return vars_selected

def drop_Unamed(data):
    unnamde='Unnamed: 0'
    if unnamde in data.columns:
        data  = data.drop(columns=[unnamde],axis=1)
    return data

# fonction pour afficher l'importance de chaque variables dans la prédiction
def plot_importances_variables(vars_imp):
    st.subheader('C): Importance des variables lors de la  Prédiction')
    g,ax = plt.subplots(figsize=(16, 5))
    sns.barplot(x = vars_imp.index, y=vars_imp,color='orange', edgecolor='red')
    plt.xticks(rotation=90)
    plt.xlabel("Variables")
    plt.ylabel("Score d'importance de la variable")
    plt.title("Importance des variables prédictrices")
    st.pyplot(plt)
# Validé et enregistrer les modele et les variables selectionnées dans le dossier (outpu_dataset_db\data_modele)
def save_modele(vars_imp,modele,algo):
       
    import joblib
    try:
        joblib.dump(modele,f'outpu_dataset_db/data_modele/finally_model_{algo}.joblib')
        vars_imp.to_csv("outpu_dataset_db/data_modele/vars_importances_selected.csv")
       
    except Exception as e:
        st.warning(f"Une Erreure d'enregistrement  du modle: {e}")
    
# ------------modelisation-----------------------------
def models(algo,train_x,train_y):
    if algo=="Regression Logistique":
        #intanciation du modele de Logisticrgression
        lr=LogisticRegression(C=1, max_iter=500, random_state=95)
        # entrainemant du modele
        model_lr=lr.fit(train_x,train_y)
        return model_lr
    if algo == "RandomForestClassifier":
        # instanciation de Random forest 
        rf = RandomForestClassifier(random_state=95,criterion="gini",max_depth=10,max_features=3,n_estimators=50)
        # entrainemant du modele
        model_rf=rf.fit(train_x,train_y)
        return model_rf
    if algo == "SVM" :
        # instanciation de SVM
        svm = SVC(C=0.0001, kernel='poly', random_state=95)
        # entrainemant du modele
        model_svm=svm.fit(train_x,train_y)
        return model_svm
    if algo =="KNN":
        # instanciation de KNN
        knn = KNeighborsClassifier(n_neighbors=2)
        #entrainement du modele
        model_knn  = knn.fit(train_x,train_y)
        return model_knn
    if algo=="Tree":
        tree = DecisionTreeClassifier(max_depth=50, random_state=95)
        # entrainement du modele
        model_tree = tree.fit(train_x,train_y)
        return model_tree


# Fonction d'évaluation de la performance d'un modèle
def plot_roc_curve(labels, pred, title='Courbe ROC'):
    # Calculer les courbes ROC
    fpr, tpr, _ = roc_curve(labels, pred)
    roc_auc = auc(fpr, tpr)
     # Créer la figure et l'axe
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Non Souscrit')
    ax.set_ylabel('Souscrit')
    ax.set_title(title)
    ax.legend(loc='lower right')

    # Afficher avec Streamlit
    st.pyplot(fig)

def model_evaluation(model, features, labels,cmap):
  # prediction

  pred = model.predict(features)
  cola,colr,colp,colf=st.columns(4)
  with cola :
        accuracy = accuracy_score(labels, pred)
        st.write(f"Accuracy : {round(accuracy, 2)}")
        
  with colr :
        recall = recall_score(labels, pred)
        st.write(f"Recall : {round(recall,2)}")
        
  with colp :
        precision = precision_score(labels, pred)
        st.write(f"Precision : {round(precision,2) } ")
        
  with colf :
        f1score = f1_score(labels, pred)
        st.write(f"F1 Score : {round(f1score,2)}")
        
  col1,col2 =st.columns([0.6,0.4])
  with col1 :
        cm = confusion_matrix(labels,pred)
            # Création de l'objet ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Non Souscrit","Souscrit"])
        # Affichage graphique interactif de la matrice de confusion avec ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(10, 5))

        if cmap !="valide":
             disp.plot(cmap=plt.cm.Blues,ax=ax)
        if cmap=="valide":
            disp.plot(ax=ax)
        # Affichage de la figure dans Streamlit
        st.pyplot(fig)
  with col2:
        plot_roc_curve(labels, pred)

def load_data_clean():
    data_upsampled=pd.read_csv(outpout_db+"/upsamled_train_data.csv")
    data_downsampled = pd.read_csv(outpout_db+"/dowsampled_train_data.csv")

    unnamde='Unnamed: 0'

    if unnamde in data_downsampled.columns:
        data_downsampled  = data_downsampled.drop(columns=[unnamde],axis=1)

    if  unnamde in data_upsampled.columns:
        data_upsampled = data_upsampled.drop(columns=[unnamde],axis=1)

    return data_upsampled,data_downsampled
