#!/usr/bin/env python
# coding: utf-8

# ## This Data is about Healthcare dataset 
# 

# # About this data : 

# ## Context
# 
#    The dataset can be utilized to develop predictive models or algorithms to identify
#     individuals at higher risk of experiencing a stroke.
#     
#        
#   - id : Computer generated patient identification number.
#  
#   - gender : gender type of customer.
# 
#   - age :  age of patient.
#  
#   - hypertension :  high blood pressure.
#   
#   - heart disese : refers to a range of conditions that affect the heart and blood vessels.
#   
#   - ever married : indicates whether an individual has ever been married or not.
#   
#   - work type : indicates the type of occupation or work .
#   
#   - Residence type :  indicates the type of residence or housing status of the individuals.
#   
#   - avg glucose level : represents the average glucose (blood sugar) level of the individuals.
#   
#   - bmi : measure that assesses body weight in relation to height.
#   
#   - smoking status : represents the smoking habits , provides information about the current or past smoking behavior of the individuals.
#   
#   - stroke : the blood supply to the brain is interrupted or reduced.

# ## Import Libraries 

# In[103]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split


# ## Read the data  

# In[104]:


df=pd.read_csv("healthcare-dataset-stroke-data.csv")


# ### show some information about data 
# - what is the shape of data
# - what are types of data ?
# - is there any null values ? 
# - if there any duplicated values?
# 

# In[105]:


df.shape


# In[106]:


df.head()


# In[107]:


df.info()


# In[108]:


msno.bar(df)
plt.show()


# In[109]:


df.isna().sum()


# ### If  The Null is Exist fill by mean 

# In[110]:


df.fillna(df.bmi.mean(),inplace=True )


# In[111]:


df.isna().sum()


# ### Drop the patient id columns
# 

# In[112]:


df=df.drop("id",axis=1)


# In[113]:


df.describe().style.background_gradient(cmap="rocket")


# ###  if there any duplicated values?
#  

# In[114]:


df.duplicated().sum()


# ### How many classes at Residence Type column ?  
# 

# In[115]:


df.Residence_type.value_counts()


# ### How many Patient at Work Type column ? 
# 

# In[116]:


df.work_type.value_counts()


# ### How many Patient at Smoking Status column ? 
#  

# In[117]:


df.smoking_status.value_counts()


# ### is femails more than males ? 
#  

# In[118]:


df.gender.value_counts()


# In[119]:


df.smoking_status.value_counts()


# ### Drop the value others  
# 

# In[120]:


df=df[df.gender!="Other"]


# ### is males heart disease more or femals?

# In[121]:


df.groupby("gender")["heart_disease"].value_counts().to_frame()


# ### is Rural heart disease more or Urban?

# In[122]:


df.groupby("Residence_type")["heart_disease"].value_counts().to_frame()


# ### Find out the stroke and Residence typee of each Gender
# 

# In[123]:


df.groupby("gender")["stroke","Residence_type"].value_counts().to_frame()


# ### How many patients have heart disease and hypertension ?

# In[124]:


df.groupby("heart_disease")["hypertension"].value_counts().to_frame()


# ### Give me all patient have both heart_disease values and hypertension, dave it into a variable calles heart_hyper 
# ### Give me all patient have both heart_disease values and stroke and save at heart_stroke
# ### Give me all patient have both heart_disease values , stroke and hypertension and save at heart_stroke_hyper

# In[125]:


df["heart_hyper"]=df["heart_disease"]+df["hypertension"]


# In[126]:


df["heart_stroke"]=df["heart_disease"]+df["stroke"]


# In[127]:


df["heart_stroke_hyper"]=df["heart_disease"]+df["stroke"]+df["hypertension"]


# In[128]:


df.info()


# In[129]:


df.head()


# ### How many Patients have Heart _ Hyper,Heart _ Stroke and heart_stroke_hyper  ?

# In[130]:


df["heart_hyper"].value_counts()


# In[131]:


df["heart_stroke"].value_counts()


# In[132]:


df["heart_stroke_hyper"].value_counts()


# In[133]:


df["Stroke"]=df["stroke"]
df=df.drop("stroke",axis=1)


# ### Visualize the work type Duration Smoking status 

# In[32]:


sns.histplot(data =df, x ="work_type",hue ="smoking_status"  )
plt.title("Visualize of the Work Type")
plt.show()


# ### Distribution of the Gender With Smoking Status

# In[33]:


sns.histplot(data =df, x ="gender",hue ="smoking_status"  )
plt.title("Distribution of the Gender With Smoking Status",color="b")
plt.show()


# ### Distribution the Ever Married With Hypertension

# In[34]:


sns.catplot(y='hypertension', x='ever_married', data=df, kind='bar',ci=None,palette= "rocket");
plt.title("Distribution the Ever Married With Hypertension ",color ="b")
plt.show()


# ### Visualize the Age With BMI During Gender

# In[35]:


sns.scatterplot(x='age', y='bmi', data=df,palette= "husl",hue="gender",size="work_type");
plt.title("Visualize the Age With BMI ",color ="b")
plt.show()


# ### Visualize the Correlation Map

# In[36]:


sns.heatmap(df.corr(),cmap='magma', linewidth=5,linecolor='black', square=True,annot=True,fmt='.2f')
plt.title("Visualize the Correlation Map ",color ="b")
plt.show()
# cmap=Blues


# ### Distribution of the Smoking Status

# In[37]:


sns.countplot(x="smoking_status" ,data = df,palette='tab10',ec='black', hatch='-')
plt.title("Distribution of the Smoking Status ",color ="b")
plt.show()


# ### Visualize of BMI and Gvg Glucose Level 

# In[38]:


sns.jointplot(x='bmi', y='avg_glucose_level' 
              , data = df,kind = 'scatter'
              ,height=10,space=1
              , marginal_kws={'color': 'xkcd:golden'});


# ### Creat barpolt for Age and heart_stroke_hyper by Gender

# In[39]:


sns.barplot(y='age',x="heart_stroke_hyper" , data = df,hue ="gender",palette="hls");


# ## Creat Model

# In[140]:


X=df.iloc[:,:-1]
X.info()


# In[142]:


y=df.iloc[:,-1]
y.info()


# In[144]:


le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])


# In[145]:


object_columns = df.select_dtypes(include='object').columns
for col in object_columns:
    df[col] = le.fit_transform(df[col])


# In[147]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[148]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[149]:


print("Trainig Data..")
print("The shape of training features: ", X_train.shape)
print("The shape of training labels: ", y_train.shape)


# In[150]:


print("Testing Data..")
print("The shape of testing features: ", X_test.shape)
print("The shape of testing labels: ", y_test.shape)


# In[151]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[152]:


model.fit(X_train, y_train)


# In[153]:


y_pred=model.predict(X_train)


# In[154]:


training_acc = (model.score(X_train, y_train) * 100)
print("The Training Accuracy = ", training_acc, "%")


# In[63]:


testing_acc = np.ceil(model.score(X_test, y_test) * 100)
print("The Testing_acc Accuracy = ", testing_acc, "%")


# In[64]:


print(model.coef_)
print(model.intercept_)


# In[65]:


print("y = %s + %s X1 + %s X2 + %s X3 + %s X4 + %s X5 + %s X6" % (model.intercept_, model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3], model.coef_[4], model.coef_[5]) )


# In[68]:


#@title Plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred)
plt.plot([min(y_train), max(y_train)], [min(y_pred), max(y_pred)], color='gray', linestyle=':', linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices with Regression Line")
plt.show()


# In[69]:


train_df_encoded


# In[155]:


# iiiiiiiiiiii333333333 de data classfication 


# In[ ]:




