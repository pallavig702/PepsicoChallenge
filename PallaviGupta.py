
# coding: utf-8

# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os, sys
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[38]:


# Dataset location
DATASET = 'shelf-life-study-data-for-analytics-challenge_prediction.xlsx'
assert os.path.exists(DATASET)


# Load and shuffle
dataset = pd.read_excel(DATASET).sample(frac = 1).reset_index(drop=True)
#dataset.describe()

olddataset=dataset


# In[39]:


#dataset.head()


# In[40]:


#print(dataset.columns)


# In[41]:


#dataset.columns = [x.strip(" ") for x in dataset.columns] 
dataset.columns=dataset.columns.str.replace(r'\s+', '')
dataset.head()


# In[42]:



NotFresh=dataset[dataset.DifferenceFromFresh >=20]
print("Ratio of not fresh :",NotFresh.shape[0]/dataset.shape[0])


# In[43]:


dataset['BinaryDifferenceFromFresh'] = np.where(dataset['DifferenceFromFresh']>=20, '1', '0')
#dataset.shape[]
for i in range(dataset.shape[0]):
    #print(i)
    print(dataset.DifferenceFromFresh[i], dataset.BinaryDifferenceFromFresh[i])


# In[44]:


dataset=dataset.drop(columns="DifferenceFromFresh")
#df.columns = df.columns.str.replace(' ', '_')
dataset['StorageConditions'].replace(['Warm Climate'], 'WarmClimate',inplace=True)
dataset['StorageConditions'].replace(['High Temperature and Humidity'], 'HighTemperatureandHumidity',inplace=True)
dataset['StorageConditions'].replace(['Cold Climate'], 'ColdClimate',inplace=True)
dataset.head()


# In[45]:


cols= dataset.columns
for i in cols:
    print(dataset[i].unique())


    #print(dataset.DifferenceFromFresh[i], dataset.BinaryDifferenceFromFresh[i])


# In[46]:


dfObj = pd.DataFrame(dataset, columns = cols)
print("Nan in each columns" , dfObj.isnull().sum(), sep='\n')


# In[47]:


df=dataset.drop(["StudyNumber","SampleID","Prediction","TransparentWindowinPackage","PackagingStabilizerAdded","PreservativeAdded","Hexanal(ppm)","ResidualOxygen(%)","Moisture(%)"], axis=1)
Test_df=dataset.drop(["StudyNumber","Prediction","TransparentWindowinPackage","PackagingStabilizerAdded","PreservativeAdded","Hexanal(ppm)","ResidualOxygen(%)","Moisture(%)"], axis=1)
df.shape


# In[48]:


Test_df.head()
df.head()


# In[49]:


yes_no_columns_df = list(filter(lambda i: df[i].dtype!=np.float64, df.columns))
yes_no_columns_Test_df = list(filter(lambda i: Test_df[i].dtype!=np.float64, Test_df.columns))
def PrintUniqueValues(yes_no_column,datase):
    for i in yes_no_column:
        print("'"+i+": '", datase[i].unique())
PrintUniqueValues(yes_no_columns_df,df)
PrintUniqueValues(yes_no_columns_Test_df,Test_df)

for column_name in yes_no_columns_df:
    mode = df[column_name].apply(str).mode()[0]
    print('Filling missing values of {} with {}'.format(column_name, mode))
    df[column_name].fillna(mode, inplace=True)

for column_name in yes_no_columns_Test_df:
    mode = Test_df[column_name].apply(str).mode()[0]
    print('Filling missing values of {} with {}'.format(column_name, mode))
    Test_df[column_name].fillna(mode, inplace=True)


# In[50]:


dfObj2 = pd.DataFrame(df, columns = cols)
print("Nan in each columns" , dfObj2.isnull().sum(), sep='\n')
PrintUniqueValues(yes_no_columns_df,df)


# In[51]:


X=df.iloc[:,df.columns!='BinaryDifferenceFromFresh']
y=df.BinaryDifferenceFromFresh


# In[52]:



df2 = pd.get_dummies(X)
df2.shape


# In[53]:



#X=df2.iloc[:,df.columns != 'BinaryDifferenceFromFresh']
X_train, X_test, y_train, y_test = train_test_split(df2,y,test_size=0.33, random_state=42)


# In[54]:




class_weight=dict({1:1.9, 0:20})
classifier2= RandomForestClassifier(n_estimators=50, max_depth=16,random_state=0, n_jobs=2)#,class_weight=class_weight)


# In[55]:


classifier2.fit(X_train,y_train)
y_pred=classifier2.predict(X_test)
accuracy_score(y_test,y_pred)


# In[56]:



print("Confusion Matrix:===> ","\n", confusion_matrix(y_test, y_pred),"\n")
print("Classification Report:===> ","\n", classification_report(y_test, y_pred))
print("Best Accuracy Score:===> ", accuracy_score(y_test, y_pred))


# In[57]:


Sampleid=Test_df.SampleID
new=Test_df.drop('SampleID',axis=1)
X_Test_df=new.iloc[:,:-1]
X_onehot=pd.get_dummies(X_Test_df)
y_Test_df=new.BinaryDifferenceFromFresh
X_Test_df.head()


# In[58]:


pred=classifier2.predict(X_onehot)
accuracy_score(y_Test_df,pred)


# In[59]:


predictions={}
actual={}
for i in range(Test_df.shape[0]):
    predictions[Sampleid[i]]=pred[i]
    actual[Sampleid[i]]=y_Test_df[i]

olddataset["pred"]=0

olddataset.head()


# In[60]:


#for keys,values in predictions.items():
#    print(keys,values)


# In[61]:


for i in range(olddataset.shape[0]):
    sampleid=olddataset.iloc[i,1]
    olddataset.pred[i]=predictions[sampleid]
    
    #print(i)
olddataset["Prediction"]=olddataset["pred"]

olddataset.head()
    


# In[64]:


final=olddataset.drop(['pred','BinaryDifferenceFromFresh'],axis=1)
final.to_excel("outputshelf_life_file_with_predictions.xlsx")

