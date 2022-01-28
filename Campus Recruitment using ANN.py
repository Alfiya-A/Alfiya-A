#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd


# In[2]:


df = pd.read_csv("/Users/alfia/Desktop/Placement_Data_Full_Class.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df["salary"]=df["salary"].fillna(0)


# In[7]:


df.isnull().sum()


# In[8]:


df.head()


# In[9]:


df[0:3].T


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


le_gender = LabelEncoder()
ssc_b = LabelEncoder()
hsc_b = LabelEncoder()
hsc_s = LabelEncoder()
degree_t = LabelEncoder()
workex = LabelEncoder()
spec = LabelEncoder()
status= LabelEncoder()


# In[12]:


df["Gender"] = le_gender.fit_transform(df.gender)
df["SSC_Board"] = ssc_b.fit_transform(df.ssc_b)
df["HSC_Board"] = hsc_b.fit_transform(df.hsc_b)
df["HSC_Stream"] = hsc_s.fit_transform(df.hsc_s)
df["Degree"] = degree_t.fit_transform(df.degree_t)
df["Experience"] = workex.fit_transform(df.workex)
df["Specialization"] = spec.fit_transform(df.specialisation)
df["Status"] = status.fit_transform(df.status)


# In[13]:


df.head()


# In[14]:


df.columns


# In[15]:


df.drop(['sl_no', 'gender', 'ssc_b','hsc_b', 'hsc_s',
         'degree_t', 'workex', 'specialisation',
         'status'],axis=1,inplace=True)


# In[16]:


df.head()


# In[17]:


df.rename(columns={'ssc_p':'SSC_P',
                  'hsc_p':'HSC_P',
                  'degree_p':'Degree_P',
                  'etest_p':'Etest_P',
                  'mba_p':'MBA_P',
                  'salary':'Salary'},inplace=True)


# In[18]:


df.columns


# In[19]:


df = df[['Gender','SSC_P','SSC_Board','HSC_P','HSC_Board',
         'HSC_Stream','Degree','Degree_P','Experience',
         "Etest_P","Specialization","Status","Salary"]]


# In[20]:


df


# In[21]:


X = df.drop("Status",axis=1)
Y = df.Status


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,
                                              test_size=0.2)


# In[24]:


x_train.shape, x_test.shape


# In[25]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[26]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch


# In[27]:


def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model


# In[28]:


tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='project33',
    project_name='Placement')


# In[29]:


tuner.search(x_train, y_train,
             epochs=5,
             validation_data=(x_test, y_test))


# In[30]:


tuner.results_summary()


# In[32]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu,sigmoid


# In[33]:


def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer='glorot_uniform',
                    activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# In[34]:


model = KerasClassifier(build_fn=create_model, verbose=1)


# In[35]:


layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']


# In[36]:


param_grid = dict(layers=layers, activation=activations,
                  batch_size = [128, 256], epochs=[30])


# In[37]:


grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)


# In[38]:


grid_result = grid.fit(x_train, y_train)


# In[39]:


[grid_result.best_score_,grid_result.best_params_]


# In[51]:


# Initialising the ANN
ann = Sequential()

ann.add(Dense(units = 12,
                     kernel_initializer = 'he_uniform',
                     activation='relu',
                     input_dim = 12))
ann.add(Dropout(0.2))

ann.add(Dense(units = 10,
                     kernel_initializer='he_uniform',
                     activation='relu'))
ann.add(Dropout(0.2))

ann.add(Dense(units = 8,
                     kernel_initializer='he_uniform',
                     activation='relu'))

ann.add(Dense(units = 1,
                     kernel_initializer = 'glorot_uniform',
                     activation = 'sigmoid'))


# In[52]:


ann.compile(optimizer = 'Adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


# In[53]:


ann.fit(x_train,y_train,validation_split=0.33,
        batch_size = 10,epochs = 100)


# In[54]:


y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)


# In[55]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[56]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score)


# In[57]:


#Lets predict the dataset value


# In[58]:


X.head(1)


# In[59]:


Y.head(1)


# In[60]:


print(ann.predict(sc.transform([[1,67,1,91,1,1,2,58,0,55,1,270000]])) > 0.5)


# In[50]:


#Our Networks predicts correctly

