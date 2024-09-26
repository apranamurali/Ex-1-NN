<H3>ENTER YOUR NAME : Aparna M
<H3>ENTER YOUR REGISTER NO :  212223220008
<H3>EX. NO.1
<H3>DATE 
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
Developed by: ANUBHARATHI SS
RegisterNumber: 212223040017

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
#Checking for null values
df.isnull().sum()
#Checking for duplicate values
df.duplicated()
#Describing the dataset
df.describe()
#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
## DATASET:
![NN1](https://github.com/user-attachments/assets/0ca19c42-7c3d-40c7-bb02-9015fa9d6f6b)

## DROPPING THE UNWANTED DATASET:
![NN2](https://github.com/user-attachments/assets/0f2df8f7-4187-4148-b0e9-a650e514ee91)

## CHECKING NULL VALUES:
![NN3](https://github.com/user-attachments/assets/7de50293-3a7c-407f-a61e-64e342ee7521)

## CHECKING FOR DUPICATION:
![NN4](https://github.com/user-attachments/assets/421a03d4-2226-4212-a9c1-08c184685b75)

## DESCRIBING THE DATASET:
![NN5](https://github.com/user-attachments/assets/03c964ef-1ce9-4827-85de-f375568ab3e3)

## SCALING THE DATASET:
![NN6](https://github.com/user-attachments/assets/891c8cf7-af60-463b-bb81-a346d1ba0c9b)

## X FEATURES:
![NN7](https://github.com/user-attachments/assets/7f78f2e0-a1f6-477f-ac1b-55ac2bcb1440)

## Y FEATURES:
![NN8](https://github.com/user-attachments/assets/c5bad6cc-519a-4594-b760-b99cad39c6fd)

## SPLLITING THE TRAINING AND TESTING DATASET:
![NN9](https://github.com/user-attachments/assets/575b6eec-84ea-4fbc-bce0-bf95f897848e)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


