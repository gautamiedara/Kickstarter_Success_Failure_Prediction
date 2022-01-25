# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:59:44 2021

@author: Gautami Edara
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.io as pio
import plotly.express as px
from pandas.plotting import *
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import timeit
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score



#import data
kickstarter = pd.read_excel(r"C:\Users\Gautami Edara\OneDrive - McGill University\Fall Semester\INSY 662 - Data Mining & Data Viz\Final Individual Project\Kickstarter.xlsx")
kickstarter_grade = pd.read_excel(r"C:\Users\Gautami Edara\OneDrive - McGill University\Fall Semester\INSY 662 - Data Mining & Data Viz\Final Individual Project\Kickstarter-Grading-Sample.xlsx")

#Removing project ids from non graded dataset that are already present in graded dataset
list_project_id=list(kickstarter_grade['project_id'])
kickstarter=kickstarter[~kickstarter['project_id'].apply(lambda x: x in list_project_id)]


def data_preprocessing(kickstarter):
    #check the initial data
    kickstarter.head()
    
    #look the null values
    kickstarter.isna().sum()
    kickstarter = kickstarter.drop(['launch_to_state_change_days'], axis =1)
    kickstarter.shape
    #18568
    
    
    kickstarter.isna().sum()
    
    #drop the null records
    kickstarter = kickstarter.dropna()
    kickstarter.isna().sum()
    kickstarter.shape
    #16879
    
    #pick only successful or failed records from state column
    kickstarter = kickstarter[(kickstarter['state']=='successful') | (kickstarter['state']=='failed')]
    kickstarter.shape
    #14214
    
    
    #dummify the variables state, staff_pick, disable_communication, category
    dummy_state=pd.get_dummies(kickstarter.state, prefix="state")
    dummy_communication=pd.get_dummies(kickstarter.disable_communication, prefix="disable_communication")
    dummy_category=pd.get_dummies(kickstarter.category, prefix="category")
    dummy_country=pd.get_dummies(kickstarter.country, prefix="country")
    dummy_currency=pd.get_dummies(kickstarter.currency, prefix="currency")
    dummy_deadline_weekday=pd.get_dummies(kickstarter.deadline_weekday, prefix="deadline_weekday")
    # dummy_state_changed_at_weekday=pd.get_dummies(kickstarter.state_changed_at_weekday, prefix="state_changed_at_weekday")
    dummy_created_at_weekday=pd.get_dummies(kickstarter.created_at_weekday, prefix="created_at_weekday")
    dummy_launched_at_weekday=pd.get_dummies(kickstarter.launched_at_weekday, prefix="launched_at_weekday")
    
    
    
    kickstarter = kickstarter.join(dummy_state)
    kickstarter = kickstarter.join(dummy_communication)
    kickstarter = kickstarter.join(dummy_category)
    kickstarter = kickstarter.join(dummy_country)
    kickstarter = kickstarter.join(dummy_currency)
    kickstarter = kickstarter.join(dummy_deadline_weekday)
    # kickstarter = kickstarter.join(dummy_state_changed_at_weekday)
    kickstarter = kickstarter.join(dummy_created_at_weekday)
    kickstarter = kickstarter.join(dummy_launched_at_weekday)
    list(kickstarter.columns)
    
# =============================================================================
# -----------removed predictors as they contain distinct values 
# 'project_id'
# 'name'
# 'deadline'
# 'state_changed_at'
# 'created_at'
# 'launched_at'
# =============================================================================
# removed the predictors which have been dummified 
# (remove main columns and keep dummified columns) 
# state
# disable_communication
# category
# country
# =============================================================================
# removed 'spotlight','staff_pick' column as it would be used only once the project is launched
# =============================================================================
#  removed these columns 
#  'state_changed_at_month',
#  'state_changed_at_day',
#  'state_changed_at_yr',
#  'state_changed_at_hr',
# =============================================================================
    #Checking correlation between continuous variables and state
    
    df = pd.DataFrame(kickstarter,columns=['goal','pledged','static_usd_rate','usd_pledged',
                                            'name_len','name_len_clean','blurb_len','blurb_len_clean','state_successful'])
    
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    
    
    # =============================================================================
    # Looking at the correlation matrix we can say that 
    # -> pledged and usd_pledged are highly correlated. 
    #    Here we take usd_pledged i.e all Amount pledged in USD and remove static_usd_rate as well.
    # -> name_len_clean and name_len are highly correlated.
    #    Here the 
    # -> blurb_len and blurb_len_clean are also moderatly correlated
    # we can keep one column and remove another to avoid any correlation issue.
    # =============================================================================
    
    #remove the following columns 
    removeColumns=[
     'usd_pledged', #after launch
     'project_id',
     'name',
     'backers_count', #after launch
     'pledged', 
     'state', 
     'disable_communication', 
     'country', 
     'currency',
     'deadline',  #timestamps
     'state_changed_at', #timestamps
     'created_at', #timestamps
     'launched_at', #timestamps
     'category', 
     'spotlight', #after launch
     'name_len',
     'blurb_len',
     'deadline_weekday',
     'state_changed_at_weekday',
     'created_at_weekday',
     'launched_at_weekday',
      'state_changed_at_month',
      'state_changed_at_day',
     'state_changed_at_yr',
      'state_changed_at_hr',
     'state_failed',
     'state_successful',
     'country',
     'currency']
    
    allColumns = list(kickstarter.columns)
    # taking the y Variable
    y = kickstarter["state_successful"]
    # taking only relevant predictors
    
    x=kickstarter[list(set(allColumns)-set(removeColumns))]
        
    
    # =============================================================================
    
    #Running feature selection technique Random Forest
    randomforest = RandomForestClassifier(random_state=0)
    model = randomforest.fit(x, y)
    model.feature_importances_
    randForest = pd.DataFrame(list(zip(x.columns, model.feature_importances_)),columns=['predictor','feature importance'])
      
    selectedFeaturesRFC =[]
    print("\n")
    for i in range(len(randForest)):
                if randForest['feature importance'][i]>=0.01:
                    selectedFeaturesRFC.append(randForest['predictor'][i])
                    print(randForest['predictor'][i]) 
                    
    # =================Random Forest OUTPUT========================================
    selectedFeaturesRFC.append('state_successful')
    return kickstarter[selectedFeaturesRFC]

kickstarter=data_preprocessing(kickstarter)
kickstarter_grade=data_preprocessing(kickstarter_grade)
# =============================================================================
#                               Gradient Boosting  
# =============================================================================


# Divided the train and test set such that training has all the columns except for state and testing data just has 
#state
X_train=kickstarter[[cols for cols in kickstarter.columns if 'state_successful' not in cols]]
X_test=kickstarter_grade[[cols for cols in kickstarter_grade.columns if 'state_successful' not in cols]]
y_train=kickstarter['state_successful']
y_test=kickstarter_grade['state_successful']

gbt = GradientBoostingClassifier(random_state = 0, learning_rate=0.1,max_depth=4,max_features='log2',n_estimators=100)
model2 = gbt.fit(X_train, y_train)
y_test_pred2 = model2.predict(X_test)
print("\n")
print('Accuracy:', metrics.accuracy_score(y_test, y_test_pred2))
print('Precision Score:', metrics.precision_score(y_test, y_test_pred2))
print('Recall Score:', metrics.recall_score(y_test, y_test_pred2))

# Calculate the F1 score
print('F1 Score:', metrics.f1_score(y_test, y_test_pred2))
print('Confusion Matrix :')
metrics.confusion_matrix(y_test , y_test_pred2)

# Beautifying the Confusion Matrix
df_confusion = pd.crosstab(y_test, y_test_pred2, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)

# =============================================================================
# Clustering
# Elbow method to show optimal value of k
# =============================================================================
df=X_train

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    kmeanVal = KMeans(n_clusters=k)
    kmeanVal = kmeanVal.fit(df)
    Sum_of_squared_distances.append(kmeanVal.inertia_)
labels =kmeanVal.labels_
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# =============================================================================
# 
#        silhouette score
#
# =============================================================================

silhouette = silhouette_samples(df, labels)

print('\nSilhouette Score:',silhouette_score(df, labels))
print("\n")

#Taken only numerical columns
cols=['goal', 'static_usd_rate', 'blurb_len_clean',
       'created_at_yr','create_to_launch_days', 'launch_to_deadline_days']
df=df[cols]
scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
pca = PCA(n_components=2)
df = pca.fit_transform(standardized_x)
reduced_df = pd.DataFrame(df, columns=['PC1','PC2'])
plt.scatter(reduced_df['PC1'], reduced_df['PC2'], alpha=.1, color='blue')
plt.xlabel('Principal Component Analysis 1')
plt.ylabel('Principal Component Analysis  2')
plt.show()

kmeans=KMeans(n_clusters=4)
model=kmeans.fit(reduced_df)
labels=model.predict(reduced_df)
reduced_df['cluster'] = labels
list_labels=labels.tolist()
count1=0
count2=0
count3=0
count4=0
for i in list_labels:
    if i==0:
        count1=count1+1
    elif i==1:
        count2=count2+1
    elif i==2:
        count3=count3+1
    elif i==3:
        count4=count4+1
u_labels=np.unique(labels)
print("\nTotal datapoints in cluster 1 (K Means):", count1)
print("Total datapoints in cluster 2 (K Means):", count2)
print("Total datapoints in cluster 3 (K Means):", count3)
print("Total datapoints in cluster 4 (K Means):", count4)
for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] )
plt.legend(u_labels)
plt.show()

# =============================================================================
# 
#  Create a data frame containing our centroids
#  Clustering
# 
# =============================================================================
df=X_train
cols=['goal', 'static_usd_rate', 'blurb_len_clean','created_at_yr','create_to_launch_days', 'launch_to_deadline_days']

df=df[cols]

scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
standardized_x=pd.DataFrame(standardized_x,columns=df.columns)
df=standardized_x
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(df)
labels=model.predict(df)
df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Total datapoints in cluster 1 (K-Means):", count1)
print("Total datapoints in cluster 2 (K-Means):", count2)
print("Total datapoints in cluster 3 (K-Means):", count3)
print("Total datapoints in cluster 4 (K-Means):", count4)

# =============================================================================
# To open in new browser and display the graph for clarity
# =============================================================================
pio.renderers.default = 'browser'

centroids = pd.DataFrame(kmeans.cluster_centers_)
fig = px.parallel_coordinates(centroids,labels=df.columns,color=u_labels)
fig.show()
