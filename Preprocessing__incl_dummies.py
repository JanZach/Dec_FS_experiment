
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

import shap
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import random
import statistics
import pickle
import matplotlib.pyplot as plt

# Merge diesen Datensatz mit der 2012er wave um mehr Daten zu haben




# import bgp dataset (2016 wave)
bgp = pd.read_csv("D:/SOEP/bgp.csv")

# import ppath
ppath = pd.read_csv("D:/SOEP/ppath.csv")

# import ppathl
ppathl = pd.read_csv("D:/SOEP/ppathl.csv")

# import bgpgen
bgpgen = pd.read_csv("D:/SOEP/bgpgen.csv")

# import bghgen
bghgen = pd.read_csv("D:/SOEP/bghgen.csv")

# import pl
#pl = pd.read_csv("D:/SOEP/pl.csv")



# Check the value counts of target vars
#bgp[['bgp0602']].value_counts()
#bgp[['bgp0608']].value_counts()

# merge bgp, ppathl, bghgen, and bgpgen
bgp_ppathl = pd.merge(bgp, ppathl, on=['pid', 'hid', 'syear'])
bgp_ppathl_bghgen = pd.merge(bgp_ppathl, bghgen, on=['hid', 'syear'])
bgp_ppathl_bghgen_bgpgen = pd.merge(bgp_ppathl_bghgen, bgpgen, on=['pid', 'hid', 'syear'])
#bgp_ppathl_bghgen_bgpgen_pl = pd.merge(bgp_ppathl_bghgen_bgpgen, pl, on=['pid', 'syear'])

Data = bgp_ppathl_bghgen_bgpgen

# create age var
Data["age"] = 2016 - Data["gebjahr"]
Data = Data.loc[Data["age"] < 105]

# Select predefined variables
int_variables  = ["age",
                 "bgbilzeit",
                 "labgro16"
                 ]

binary_variables = ["sex", 
                   "germborn", 
                   'bgp112'
                   #"bgp08"
                   ]

likert10_variables = ["bgp0108",
                     "bgp0101",
                     "bgp0105", 
                     "bgp0106",
                     "bgp0103",
                     'bgp0112',        
                     ]

nom_variables = ["bgfamstd"]

likertOther_variables = ["bgp05",
                        "bgp0111",
                        "bgp0610",
                        "bgp0702",
                        "bgp115"]

target_vars = ["bgp0602",
               "bgp0608"
               ]

# reduce dataframe to selected vars
Data = Data.loc[:, int_variables +
                binary_variables +
                likert10_variables + 
                #nom_variables +
                likertOther_variables + target_vars]



# ---- Convert categoricals to numerics

# create replace dictonaries

bgp0602_dict = {"[-1] keine Angabe":-1,
                "[1] 1 Sehr wichtig":1,
                "[2] 2 Wichtig":2,
                "[3] 3 Weniger wichtig":3,
                "[4] 4 Ganz unwichtig":4,
                "[-5] In Fragebogenversion nicht enthalten":-5}

Likert10_dict = {"[8] 8 Zufrieden: Skala 0-Niedrig bis 10-Hoch":8,
                "[-5] In Fragebogenversion nicht enthalten":-5,
                "[7] 7 Zufrieden: Skala 0-Niedrig bis 10-Hoch":7,
                "[9] 9 Zufrieden: Skala 0-Niedrig bis 10-Hoch":9,
                "[10] 10 Zufrieden: Skala 0-Niedrig bis 10-Hoch":10,
                "[-5] In Fragebogenversion nicht enthalten":-5,
                 "[6] 6 Zufrieden: Skala 0-Niedrig bis 10-Hoch":6,
                 "[5] 5 Zufrieden: Skala 0-Niedrig bis 10-Hoch":5,
                 "[4] 4 Zufrieden: Skala 0-Niedrig bis 10-Hoch":4,
                 "[3] 3 Zufrieden: Skala 0-Niedrig bis 10-Hoch":3,
                 "[2] 2 Zufrieden: Skala 0-Niedrig bis 10-Hoch":2,
                 "[1] 1 Zufrieden: Skala 0-Niedrig bis 10-Hoch":1,
                 "[0] 0 Zufrieden: Skala 0-Niedrig bis 10-Hoch":0,
                 "[-1] keine Angabe":-1,
                 "[-2] trifft nicht zu":-2}

bgfamstd_dict = {'[1] Verheiratet, mit Ehepartner zusammenlebend':'1_verheiratet und zusammenlebend',
                '[2] Verheiratet, dauernd getrennt lebend':'2_verheiratet und getrennt lebend',
                '[3] Ledig':'3_ledig',
                '[4] Geschieden / eingetragene gleichgeschlechtliche Partnerschaft aufgehoben':'4_geschieden',
                '[5] Verwitwet / Lebenspartner/-in aus eingetragener gleichgeschlechtlicher Partner':'5_verwitwet',
                '[6] Ehepartner im Ausland':'6_verheiratet und Ehepartner im Ausland',
                '[7] Eingetragene gleichgeschlechtliche Partnerschaft zusammenlebend':'7_eingetragene gleichgeschlechtliche Partnerschaft zusammenlebend',
                '[8] Eingetragene gleichgeschlechtliche Partnerschaft getrennt lebend':'8_eingetragene gleichgeschlechtliche Partnerschaft getrennt lebend'}

sex_dict = {"[2] weiblich":0,
            "[1] maennlich":1}

bgp112_dict = {"[2] Nein":0,
            "[1] Ja":1,
            "[-5] In Fragebogenversion nicht enthalten":0,  # na imputation with modus
            "[-1] keine Angabe": 0}                         # na imputation with modus

bgp08_dict = {"[1] ja":1,
              "[2] nein":0,
              "[-5] In Fragebogenversion nicht enthalten":-5,
              "[-1] keine Angabe":-1}

germborn_dict = {"[1] in Deutschland geboren oder immigr.<1950":1,
                 "[2] nicht in Deutschland geboren":0}

bgp05_dict = {"[5] 5 Risikobereit Skala 0-Gar nicht, 10-Sehr":5,
                "[7] 7 Risikobereit Skala 0-Gar nicht, 10-Sehr":7,
                "[6] 6 Risikobereit Skala 0-Gar nicht, 10-Sehr":6,
                "[3] 3 Risikobereit Skala 0-Gar nicht, 10-Sehr":3,
                "[8] 8 Risikobereit Skala 0-Gar nicht, 10-Sehr":8,
                "[2] 2 Risikobereit Skala 0-Gar nicht, 10-Sehr":2,
                 "[4] 4 Risikobereit Skala 0-Gar nicht, 10-Sehr":4,
                 "[1] 1 Risikobereit Skala 0-Gar nicht, 10-Sehr":1,
                 "[0] 0 Risikobereit Skala 0-Gar nicht, 10-Sehr":0,
                 "[9] 9 Risikobereit Skala 0-Gar nicht, 10-Sehr":9,
                 "[10] 10 Risikobereit Skala 0-Gar nicht, 10-Sehr":10,
                 "[-1] keine Angabe":-1}

bgp115_dict = {
    '[5] Einmal im Monat oder seltener':5,
    '[4] An zwei bis vier Tagen im Monat':4,
    '[6] Nie':6,
    '[-5] In Fragebogenversion nicht enthalten':5, # Impute negative values with modus
    '[3] An zwei bis drei Tagen in der Woche':3,
    '[2] An vier bis sechs Tagen in der Woche':2,
    '[1] Taeglich':1,
    '[-1] keine Angabe':5                           # Impute negative values with modus
    }

bgp0111_dict = {"[8] 8 auf Skala 0-10":8,
                "[9] 9 auf Skala 0-10":9,
                "[7] 7 auf Skala 0-10":7,
                "[10] 10=Ganz und gar zufrieden":10,
                "[6] 6 auf Skala 0-10":6,
                "[5] 5 auf Skala 0-10":5,
                "[4] 4 auf Skala 0-10":4,
                "[3] 3 auf Skala 0-10":3,
                "[2] 2 auf Skala 0-10":2,
                "[1] 1 auf Skala 0-10":1,
                "[-1] keine Angabe":-1,
                "[-5] In Fragebogenversion nicht enthalten":-5,
                "[0] 0=Ganz und gar unzufrieden":0}

bgp0610_dict = {"[3] weniger wichtig":2,
                "[4] ganz unwichtig":1,
                "[2] wichtig":3,
                "[1] sehr wichtig":4,
                "[-1] keine Angabe":-1,
                "[-5] In Fragebogenversion nicht enthalten":-5}

bgp0702_dict = {"[2] Mindestens 1 Mal pro Woche":4,
                "[3] Mindestens 1 Mal pro Monat":3,
                "[4] Seltener":2,
                "[1] Taeglich":5,
                "[5] Nie":1,
                "[-5] In Fragebogenversion nicht enthalten":-5,
                "[-1] keine Angabe":-1}


# Convert target variables
Data["bgp0602"] = Data["bgp0602"].replace(bgp0602_dict)
Data['bgp0602'].value_counts()
Data["bgp0608"] = Data["bgp0608"].replace(bgp0602_dict)
Data['bgp0608'].value_counts()

# Correct typo in data
Data.loc[Data['bgp0112'] == '[10] 10 Zufrieden: Skala 0-Niedrig bis 10-Hoc', 'bgp0112'] = '[10] 10 Zufrieden: Skala 0-Niedrig bis 10-Hoch'

# replace likert10 vars
for i in likert10_variables:
    print(i)
    Data.replace({i: Likert10_dict},
                                        inplace=True)
# replace sex
Data.replace({'sex': sex_dict},
                                 inplace = True)
# replace germborn
Data.replace({"germborn": germborn_dict},
                                 inplace=True)
# replace germborn
Data.replace({"bgp112": bgp112_dict},
                                 inplace=True)
# replace bgfamstd
#bgp_ppathl_bghgen_bgpgen.replace({"bgfamstd": bgfamstd_dict},
#                                inplace=True)
# replace bgp08 (online banking)
Data.replace({"bgp08": bgp08_dict},
                                   inplace=True)
# replace bgp05 (take risk)
Data.replace({"bgp05": bgp05_dict},
                                 inplace=True)
# replace bgp115 (alcohol)
Data.replace({"bgp115": bgp115_dict},
                                 inplace=True)

# replace bgp0111 (satisfaction with social life)
Data.replace({"bgp0111": bgp0111_dict}
                                             , inplace=True)
# replace bgp0610
Data.replace({"bgp0610": bgp0610_dict},
                                   inplace=True)
# replace bgp0702
Data.replace({"bgp0702": bgp0702_dict},
                                   inplace=True)


# fill negative Current Gross Labor Income and bgbilzeit with median
Data.loc[Data["labgro16"] < 0, 'labgro16'] = Data.loc[Data["labgro16"] > 0, 'labgro16'].median()
Data.loc[Data["bgbilzeit"] < 0, 'bgbilzeit'] = Data.loc[Data["bgbilzeit"] > 0, 'bgbilzeit'].median()

# Delete negative values from likert vars
"""
Data = Data.loc[#(Data['bgp08'] > 0) &
                (Data['bgp0101'] > 0) &
                (Data['bgp0103'] > 0) &
                (Data['bgp0105'] > 0) &
                (Data['bgp0106'] > 0) &
                (Data['bgp0108'] > 0) &
                (Data['bgp0111'] > 0) &
                (Data['bgp0112'] > 0) &
                (Data['bgp05'] > 0) &
                (Data['bgp0610'] > 0) &
                (Data['bgp0702'] > 0) #&
                #(Data['bgfamstd'] != '[-3] nicht valide') &
                #(Data['bgfamstd'] != '[-1] keine Angabe')
                ]
"""

# Alternatively: replace negative values with median
for col in likert10_variables+likertOther_variables:
    
    Data.loc[Data[col] < 0, col] = Data.loc[Data[col] > 0, col].median()
    print(Data.loc[Data[col] < 0, col]) # checkk whether loop was successfull: should contain only empties
    

"""
# Investigate which variable contains high amount of negative values 
for i in ['bgp0101','bgp0103','bgp0105','bgp0106','bgp0108','bgp0111','bgp0112','bgp05','bgp0610', 'bgp0702']:
    print(i)
    print(len(Data.loc[Data[i] > 0]))
    print('- - - - - - - -')
# -> bgp08 and bgp0103 lead to significant loss of observations!
"""

# Check the value counts of all variables
#for col in bgp_ppathl_bghgen_bgpgen_prepr.columns:
#    print(col)
#    print(bgp_ppathl_bghgen_bgpgen_prepr[col].value_counts())
#    print("- - - - - -")

"""
# OneHot encoder
# nominal
oneHot = pd.get_dummies(bgp_ppathl_bghgen_bgpgen_prepr['bgfamstd'])
bgp_ppathl_bghgen_bgpgen_prepr.drop('bgfamstd', axis=1, inplace=True)
bgp_ppathl_bghgen_bgpgen_prepr = bgp_ppathl_bghgen_bgpgen_prepr.join(oneHot)
"""


# Aggregate target variable and drop original target variables
Data['TARGET'] = (Data['bgp0602'] + Data['bgp0608'])/2
Data['TARGET'] = np.where(Data['TARGET'] <= 2, 1, 0)
Data['TARGET'].value_counts()
Data.drop(['bgp0602', 'bgp0608'], axis=1, inplace=True)



# - - - - Model training without na's - - - - 

### Assign X and Y        
X = Data.drop(['bgp0105','bgp0103', 'bgp112', 'bgp0101','TARGET'], axis=1)
y = Data['TARGET']
sum(y)/len(y) # -> balanced!


"""
#Assign test- and training set

acc_list = []

for seed in [50,70,90]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = seed)
    
    
    # ---- XGBoost -----#
    
    # Parameter tuning
    param_grid = [
        {'learning_rate': [0.1, 0.2, 0.3, 0.4],
         'max_depth': [2, 3, 4, 5, 6, 7]},
      ]
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    
    grid_search = GridSearchCV(xgb, param_grid,
                               scoring='accuracy',
                               return_train_score=True,
                               verbose=3)
    grid_search.fit(X_train, y_train)
    
    # Evaluate the grid search
    grid_search.best_params_
     
    cvres = grid_search.cv_results_
    
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)
    
    # Select final model and predict y_test
    final_model = grid_search.best_estimator_
    pred = final_model.predict(X_test)
    
    accuracy = accuracy_score(pred, y_test)
    acc_list.append(accuracy)

# Filter our irrelevant variables
### apply shap values
explainer       = shap.TreeExplainer(final_model, X_train ) 
shap_values     = explainer.shap_values(X_test, check_additivity=False)    
    
summary_df = pd.DataFrame([X_test.columns, abs(shap_values).mean(axis=0)]).T.sort_values(1,ascending=False)  #### Schaue ob das korrekt ist, dass mean() rausgenommen wurde
        


# ------ Otree preprocessing ---------

# Create the median list for Otree
dict_medians = {}
for col in X_train.columns:
    median = X_train.loc[:,col].median()
    dict_medians[col] = median
   
print(dict_medians)


# Create the dummies via for-loop
for col in X_train.columns:
    X_train[col+"_dummy"] = 0

#randomly insert nan
for col in X_train.columns[0:14]:
    X_train.loc[X_train.sample(frac=0.05).index, col] = np.nan

#Process the nan's
for col in range(0,14):
    for i in range(0,len(X_train)):
        if np.isnan(X_train.iloc[i,col]):
            X_train.iloc[i, col+14] = 1

# na imputer  for training set (median)
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
"""


# -------- Model training with na's

acc_list        = []
roc_auc_list    = []

for seed in [70, 90, 1000]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = seed)
    
    # --- X_train ---- #
    # Create the dummies via for-loop
    for col in X_train.columns:
        X_train[col+"_dummy"] = 0

    #randomly insert nan
    for col in X_train.columns[0:13]:
        X_train.loc[X_train.sample(frac=0.05).index, col] = np.nan

    #Process the nan's
    for col in range(0,13):
        for i in range(0,len(X_train)):
            if np.isnan(X_train.iloc[i,col]):
                X_train.iloc[i, col+13] = 1

    # na imputer  for training set (median)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    
    # --- X_test ---- #
    # Create the dummies via for-loop
    for col in X_test.columns:
        X_test[col+"_dummy"] = 0

    #randomly insert nan
    for col in X_test.columns[0:13]:
        X_test.loc[X_test.sample(frac=0.05).index, col] = np.nan

    #Process the nan's
    for col in range(0,13):
        for i in range(0,len(X_test)):
            if np.isnan(X_test.iloc[i,col]):
                X_test.iloc[i, col+13] = 1
                
    X_test_imputed = imputer.transform(X_test)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    
    # ---- XGBoost -----#
    
    # Parameter tuning
    param_grid = [
        {'learning_rate': [0.1, 0.2, 0.3, 0.4],
         'max_depth': [2, 3, 4, 5, 6, 7]},
      ]
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    
    grid_search = GridSearchCV(xgb, param_grid,
                               scoring='accuracy',
                               return_train_score=True,
                               verbose=3)
    grid_search.fit(X_train, y_train)
    
    # Evaluate the grid search
    grid_search.best_params_
     
    cvres = grid_search.cv_results_
    
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)
    
    # Select final model and predict y_test
    final_model = grid_search.best_estimator_
    pred = final_model.predict(X_test_imputed)
    
    accuracy = accuracy_score(pred, y_test)
    auc      = roc_auc_score(pred, y_test)
    
    acc_list.append(accuracy)
    roc_auc_list.append(auc)
    
print("Accuracy: ", acc_list)
print("AUC: ", roc_auc_list) 
  
#Output
# Accuracy: [0.6761825955040685, 0.676458419528341, 0.6644600744724866, 0.6742518273341608]
#AUC:        [0.6691069893704171, 0.6709221488521353, 0.6579175120671175, 0.666863572638362]


# save model
#filename = 'C:/Users/janmo/OneDrive/Dokumente/Goethe Uni/Doktor/Projekte/Decentralized Feature Selection 1/SOEP/altruism_prediction_model.sav'
#pickle.dump(final_model, open(filename, 'wb'))



# ----------------------
