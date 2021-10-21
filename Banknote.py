#!/usr/bin/env python
# coding: utf-8

# # Bank Note Authentication
# 
# #### Data were extracted from images that were taken from genuine and forged bank note-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final image have 400x400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extracted features from images.

# # Loading the Required Libraries

# In[1]:


#!pip install missingno


# In[2]:


# Basics
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
import missingno as msno
from sklearn.preprocessing import StandardScaler, MinMaxScaler, binarize

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# Metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score

# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2

# Warnings
import warnings as ws
ws.filterwarnings('ignore')


# # Loading the Dataset on which we will be working

# In[3]:


df = pd.read_csv(r"C:\Users\Binita Mandal\Desktop\Projects for Data Science\BankNote_Authentication.csv")


# In[4]:


# Lets see first few rows
df.head()


# In[5]:


# Last few rows
df.tail()


# In[6]:


# Size
df.size


# In[7]:


# Shape
df.shape


# In[8]:


# Columns
df.columns


# In[9]:


# Information
df.info()


# In[10]:


# Summary
def summary(df):
    df = {
     'Count' : df.shape[0],
     'NA values' : df.isna().sum(),
     '% NA' : round((df.isna().sum()/df.shape[0]) * 100, 2),
     'Unique' : df.nunique(),
     'Dtype' : df.dtypes,
     'min' : round(df.min(),2),
     '25%' : round(df.quantile(.25),2),
     '50%' : round(df.quantile(.50),2),
     'mean' : round(df.mean(),2),
     '75%' : round(df.quantile(.75),2),   
     'max' : round(df.max(),2)
    } 
    return(pd.DataFrame(df))

print('Shape is :', df.shape)
summary(df)


# ## Lets see the pairplot for the dataset

# In[11]:


# Import seaborn
import seaborn as sns
#import matplotlib
import matplotlib.pyplot as plt
# Use pairplot and set the hue to be our class
sns.pairplot(df, hue='class') 

# Show the plot
plt.show()


# #### Lets see the Histogram

# In[12]:


df.hist(figsize = (10,10))
plt.show()


# In[13]:


# Class vs each rows
col_names = df.drop('class', axis = 1).columns.tolist()

plt.figure(figsize = (10,3))
i = 0
for col in col_names:
    plt.subplot(1,4,i+1)
    plt.grid(True, alpha =0.5)
    sns.kdeplot(df[col][df['class'] ==0], label = 'Fake note')
    sns.kdeplot(df[col][df['class'] ==1], label = 'Original note')
    plt.title('Class vs ' + col)
    plt.tight_layout()
    i+=1
plt.show()


# ## Splitting the dataset

# In[14]:


X = df.drop('class', axis=1)   # input feature vector
y = df['class']                # labelled target vector


# In[15]:


### Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=7)


# In[16]:


# Model
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('GB', GradientBoostingClassifier()))


# In[17]:


def model_selection(X_train, y_train):
   acc_result = []
   auc_result = []
   names = []

   col = ['Model', 'ROC AUC Mean','ROC AUC Std','ACC Mean', 'ACC Std']
   result = pd.DataFrame(columns = col)

   i=0
   for name, model in models:
       kfold = KFold(n_splits = 10, random_state = 7)
       cv_acc_result  = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'accuracy')
       cv_auc_result  = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'roc_auc')

       acc_result.append(cv_acc_result)
       auc_result.append(cv_auc_result)
       names.append(name)

       result.loc[i] = [name, 
                        cv_auc_result.mean(), 
                        cv_auc_result.std(),
                        cv_acc_result.mean(),
                        cv_acc_result.std()]

       result = result.sort_values('ROC AUC Mean', ascending = False)
       i+= 1

   plt.figure(figsize = (10,5))
   plt.subplot(1,2,1)
   sns.boxplot(x = names, y = auc_result)
   plt.title('ROC AUC Score')

   plt.subplot(1,2,2)
   sns.boxplot(x = names, y = acc_result)
   plt.title('Accuracy Score')
   plt.show()

   return(result)


# In[18]:


model_selection(X_train, y_train)


# # Lets perform KNN

# In[19]:


def model_validation(model,X_test,y_test,thr = 0.5) :
    
    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = binarize(y_pred_prob.reshape(1,-1), thr)[0]
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (10,3))
    plt.subplot(1,2,1)
    sns.heatmap(cnf_matrix, annot = True, fmt = 'g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')

    fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
    plt.subplot(1,2,2)
    sns.lineplot(fpr, tpr)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    
    print('Classification Report :')
    print('===' * 20)
    print(classification_report(y_test, y_pred))

    score = tpr - fpr
    opt_threshold = sorted(zip(score,threshold))[-1][1]
    print('='*20)
    print('Area Under Curve', roc_auc_score(y_test,y_pred))
    print('Accuracy', accuracy_score(y_test,y_pred))
    print('Optimal Threshold : ',opt_threshold)
    print('='*20)
    
KNeighborsClassifier()


# In[20]:


param_grid = {
    'leaf_size' : [2,5,7,9,11],
    'n_neighbors' : [2,5,7,9,11],
    'p' : [1,2]    
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid)
grid.fit(X_train, y_train)


# In[21]:


grid.best_params_


# In[22]:


final_model = grid.best_estimator_


# In[23]:


model_validation(final_model, X_test, y_test)


# # Gini Index & Entropy

# ## Gini

# In[24]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)
X_train.head()


# In[25]:


clf_g = DecisionTreeClassifier(criterion="gini",random_state = 10)

clf_g.fit(X_train, y_train)


# In[26]:


kfold = KFold(n_splits=4, random_state=100)


# In[27]:


scores = cross_val_score(clf_g, X_train, y_train, cv=kfold)
scores.mean()


# In[28]:


y_pred = clf_g.predict(X_test)
y_pred


# In[29]:


cfm = confusion_matrix(y_pred, y_test)
cfm


# In[30]:


# accuracy score
print('accuracy using gini index: ',accuracy_score(y_pred,y_test))


# ## Entropy

# In[31]:


clf_e = DecisionTreeClassifier(criterion="entropy",random_state = 10)
clf_e.fit(X_train,y_train)


# In[32]:


scores = cross_val_score(clf_e, X_train, y_train, cv=kfold)
scores.mean()


# In[33]:


y_pred_e = clf_g.predict(X_valid)
y_pred_e


# In[34]:


cfm_e = confusion_matrix(y_pred_e, y_valid)
cfm_e


# In[35]:


# accuracy score
print('accuracy using entropy: ',accuracy_score(y_pred_e,y_valid))


# In[36]:


# generate the tree

from sklearn import tree
tree.export_graphviz(clf_g, out_file="tree_gini.dot")


# In[ ]:




