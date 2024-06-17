#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')


# In[8]:


get_ipython().system('pip install scikit-learn')


# In[15]:


from IPython import get_ipython


# In[9]:


from pickle import dump


# In[10]:


from pickle import load
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression


# In[11]:


st.title('Model Deployment: Pre-built LogistiRegression')
st.sidebar.header('user Input parameters')


# In[12]:


def user_input_feature():
    CLMSEX=st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR=st.sidebar.selectbox('Insurence',('1','0'))
    SEATBELT=st.sidebar.selectbox('seatbelt',('1','0'))
    CLMAGE= st.sidebar.number_input('Insert the age')
    LOSS = st.sidebar.number_input('Insert loss')
    data={'CLMSEX':CLMSEX,'CLMINSUR':CLMINSUR,'SEATBELT':SEATBELT,'CLMAGE':CLMAGE,'LOSS':LOSS}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_feature()
st.subheader('user inputs parameters')
st.write(df)


# In[13]:


#Load the model from disk


# In[14]:


load_model=load(open('Logistic_model.pkl','rb'))
prediction=load_model.predict(df)
prediction_prob=load_model.predict_proba(df)
st.subheader('predicted results')
st.write('yes' if prediction[0]==0 else
         'claiments will not hire')
st.subheader('prediction probabulity')
st.write(prediction_prob)


# In[ ]:





# In[ ]:




