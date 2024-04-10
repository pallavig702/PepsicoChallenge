
# Import Libraries
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

#########################################################################################
################################### DATA SET IMPORT #####################################
#########################################################################################
# Dataset location
DATASET = 'shelf-life-study-data-for-analytics-challenge_prediction.xlsx'
assert os.path.exists(DATASET)

# Load Dataset and shuffle
dataset = pd.read_excel(DATASET).sample(frac = 1).reset_index(drop=True)

#olddataset=dataset
#########################################################################################
#################### Replace Space in Columns names with no space #######################
#########################################################################################
dataset.columns=dataset.columns.str.replace(r'\s+', '')
dataset.head()

#########################################################################################
############################# BINARISE TARGET ###########################################
#########################################################################################
NotFresh=dataset[dataset.DifferenceFromFresh >=20]
print("Ratio of not fresh :",NotFresh.shape[0]/dataset.shape[0])

dataset['BinaryDifferenceFromFresh'] = np.where(dataset['DifferenceFromFresh']>=20, '1', '0')
#dataset.shape[]
for i in range(dataset.shape[0]):
    #print(i)
    print(dataset.DifferenceFromFresh[i], dataset.BinaryDifferenceFromFresh[i])

dataset=dataset.drop(columns="DifferenceFromFresh")

#########################################################################################
############################# SOME MORE DATA CARPENTRY ##################################
#########################################################################################
dataset['StorageConditions'].replace(['Warm Climate'], 'WarmClimate',inplace=True)
dataset['StorageConditions'].replace(['High Temperature and Humidity'], 'HighTemperatureandHumidity',inplace=True)
dataset['StorageConditions'].replace(['Cold Climate'], 'ColdClimate',inplace=True)
dataset.head()


#########################################################################################
########################## PRINT UNIQUE VALUES IN EACH COLUMN ###########################
#########################################################################################
cols= dataset.columns
for i in cols:
    print(dataset[i].unique())

#########################################################################################
########################## PRINT NULL VALUES IN EACH COLUMN ###########################
#########################################################################################
dfObj = pd.DataFrame(dataset, columns = cols)
print("Nan in each columns" , dfObj.isnull().sum(), sep='\n')

#########################################################################################
########################## DROP COLUMNS WITH HUGE NULL VALUES ###########################
#########################################################################################
df=dataset.drop(["StudyNumber","SampleID","Prediction","TransparentWindowinPackage","PackagingStabilizerAdded","PreservativeAdded","Hexanal(ppm)","ResidualOxygen(%)","Moisture(%)"], axis=1)
Test_df=dataset.drop(["StudyNumber","Prediction","TransparentWindowinPackage","PackagingStabilizerAdded","PreservativeAdded","Hexanal(ppm)","ResidualOxygen(%)","Moisture(%)"], axis=1)
df.shape


#########################################################################################
############################### SOME EXPLORATIONS  ######################################
#########################################################################################
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

#dfObj2 = pd.DataFrame(df, columns = cols)
#print("Nan in each columns" , dfObj2.isnull().sum(), sep='\n')
#PrintUniqueValues(yes_no_columns_df,df)

#########################################################################################
########################### SPLIT DATA INTO TRAIN AND TEXT# ############################
#########################################################################################
X=df.iloc[:,df.columns!='BinaryDifferenceFromFresh']
y=df.BinaryDifferenceFromFresh

#df2 = pd.get_dummies(X)
#df2.shape
#X=df2.iloc[:,df.columns != 'BinaryDifferenceFromFresh']
X_train, X_test, y_train, y_test = train_test_split(df2,y,test_size=0.33, random_state=42)

#########################################################################################
##################################### DEFINE CLASSIFIER #################################
#########################################################################################
#class_weight=dict({1:1.9, 0:20}) #or "balanced"
classifier2= RandomForestClassifier(n_estimators=50, max_depth=16,random_state=0, n_jobs=2)#,class_weight=class_weight)

##########################################################################################
################################## MODEL FITTING AND TEST R###############################
##########################################################################################
classifier2.fit(X_train,y_train)
y_pred=classifier2.predict(X_test)
accuracy_score(y_test,y_pred)

##########################################################################################
################################## PRINT MODEL METRICS ###################################
##########################################################################################
print("Confusion Matrix:===> ","\n", confusion_matrix(y_test, y_pred),"\n")
print("Classification Report:===> ","\n", classification_report(y_test, y_pred))
print("Best Accuracy Score:===> ", accuracy_score(y_test, y_pred))


##############################################################################################
############################ MAKE PREDICTION FOR COMPETITION #################################
##############################################################################################
Sampleid=Test_df.SampleID
new=Test_df.drop('SampleID',axis=1)
X_Test_df=new.iloc[:,:-1]
X_onehot=pd.get_dummies(X_Test_df)
y_Test_df=new.BinaryDifferenceFromFresh
X_Test_df.head()

pred=classifier2.predict(X_onehot)
accuracy_score(y_Test_df,pred)

predictions={}
actual={}
for i in range(Test_df.shape[0]):
    predictions[Sampleid[i]]=pred[i]
    actual[Sampleid[i]]=y_Test_df[i]

olddataset["pred"]=0

for i in range(olddataset.shape[0]):
    sampleid=olddataset.iloc[i,1]
    olddataset.pred[i]=predictions[sampleid]
    
olddataset["Prediction"]=olddataset["pred"]
olddataset.head()
final=olddataset.drop(['pred','BinaryDifferenceFromFresh'],axis=1)
final.to_excel("outputshelf_life_file_with_predictions.xlsx")

