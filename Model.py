#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


train_df=pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# In[6]:


train_df.describe(include='all')


# In[30]:


test_df.describe(include='all')


# In[31]:


train_df.isna().sum()


# In[32]:


test_df.isna().sum()


# In[33]:


train_df.hist(bins=50,figsize=(20,15))
plt.show()


# In[34]:


train_df.Age.value_counts(bins=10)


# In[35]:


train_df.Parch.value_counts()


# In[36]:


corr_matrix = train_df.corr()


# In[37]:


corr_matrix


# In[38]:


corr_matrix['Survived'].sort_values(ascending=False)


# In[39]:


sns.catplot(x="Pclass",y= "Age", hue= "Sex", kind= "box", data= train_df)


# In[40]:


class Titanic:
    
    def fill_missing_values(x):
        def fill_age_missing_value(Pclass, Sex):
            df= x[(x['Pclass']== Pclass) & (x['Sex']== Sex)]
            df['Age'].fillna(df['Age'].median(),inplace= True)
            x.loc[(x['Pclass']==Pclass) & (x['Sex']==Sex),['Age']]= df['Age']
        
        def fill_Embarked_missing_value():
            x['Embarked'].fillna(x['Embarked'].mode()[0],inplace=True)  
        
        fill_age_missing_value(3,'female')
        fill_age_missing_value(2,'female')
        fill_age_missing_value(1,'female')
        fill_age_missing_value(3,'male')
        fill_age_missing_value(2,'male')
        fill_age_missing_value(1,'male')
        
        fill_Embarked_missing_value()
        
    def drop_features(x):
        x.drop(['Cabin','Ticket','Fare'],axis= 1,inplace=True)
            
    def convert_categorical_to_numerical(x):
        import re
        x['Name']=x['Name'].str.extract(r'(Mr\.|Mrs\.|Miss\.|Master\.)')
        x['Name'].fillna('Mr.',inplace=True)
        
        x['Pclass']= x['Pclass'].astype('category')
        x['Sex']= x['Sex'].astype('category')
        x['Embarked']= x['Embarked'].astype('category')
        x['Name']= x['Name'].astype('category')

        columns=x.columns
        categorical_columns=[]
        for column in columns:
            if x[column].dtype.name=='category':
                categorical_columns.append(column)

        dummy_df=pd.get_dummies(x[categorical_columns],drop_first=True,prefix=categorical_columns)
        x.drop(categorical_columns,axis=1,inplace=True)
        x= pd.concat([x,dummy_df],axis=1)
        return x


# In[41]:


Titanic.fill_missing_values(train_df)
Titanic.drop_features(train_df)
train_df= Titanic.convert_categorical_to_numerical(train_df)

Titanic.fill_missing_values(test_df)
Titanic.drop_features(test_df)
test_df= Titanic.convert_categorical_to_numerical(test_df)


# In[42]:


train_df


# In[43]:


test_df


# In[44]:


x_train= train_df
y_train= train_df['Survived'].copy()


# In[45]:


x_train.drop(['Survived'],axis=1,inplace=True)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(x_train,y_train)
y_predict = DTC.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission1.csv',index=False)


# In[46]:


from sklearn.model_selection import GridSearchCV
param_dict = {'bootstrap' : [False], 'n_estimators' : [10,20], 'max_depth' : [2,3,4],'min_samples_split' : [2,3,4], 'min_samples_leaf' : [4,5,6], 'n_jobs' : [20,30]}


# In[47]:


from sklearn.ensemble import RandomForestClassifier
GSCV= GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_dict)
GSCV.fit(x_train,y_train)


# In[48]:


GSCV.best_estimator_


# In[54]:


rfc= RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',max_depth=4, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=4, min_samples_split=4,min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=20,oob_score=False, random_state=None, verbose=0,warm_start=False)
rfc.fit(x_train,y_train)
y_predict=rfc.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission2.csv',index=False)
df_submit.to_csv('Submission10.csv',index=False)                            


# In[569]:


from xgboost import XGBClassifier
GSCV_XGB= GridSearchCV(estimator=XGBClassifier(),param_grid=param_dict,cv=10)
GSCV_XGB.fit(x_train,y_train)
GSCV_XGB.best_estimator_


# In[571]:


xgbc= XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=1, min_samples_leaf=4, missing=None,
              n_estimators=10, n_jobs=20, nthread=None,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
              subsample=1, verbosity=1)
xgbc.fit(x_train,y_train)
y_predict = xgbc.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
#df_submit.to_csv('Submission3.csv',index=False)
df_submit.to_csv('Submission9.csv',index=False)                              #using gbtree as dart.


# In[ ]:


y_predict = xgbc.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission5.csv',index=False)


# In[ ]:


from sklearn import svm
svc= svm.SVC()
svc.fit(x_train,y_train)
y_predict = svc.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission4.csv',index=False)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(x_train,y_train)
y_predict = knn.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission6.csv',index=False)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb= GaussianNB()
gnb.fit(x_train,y_train)
y_predict = gnb.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission7.csv',index=False)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
mnb= MultinomialNB()
mnb.fit(x_train,y_train)
y_predict = mnb.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission8.csv',index=False)


# In[ ]:


gnb= GaussianNB()
gnb.fit(x_train,y_train)
y_predict = gnb.predict(test_df)
df_submit = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_predict})
df_submit.to_csv('Submission7.csv',index=False)


# In[ ]:




