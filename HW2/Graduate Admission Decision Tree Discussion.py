#!/usr/bin/env python
# coding: utf-8

# In[1041]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO   
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import pydotplus
import numpy


# In[1042]:


df = pd.read_csv('./Admission_Predict_Ver1.1.csv')
df.head()


# ## Design a set of rules to create admit column

# ![rule.png](attachment:rule.png)

# In[1043]:


def review(df):
    if (df['CGPA'] <= 8):
        if (df['SOP'] <= 4):
            if (df['Research'] <= 0.5):
                return 0
            else:
                return 1
        else:
            return 1
    else:
        if (df['GRE Score'] <= 300):
            return 0
        else:
            return 1
df['Admit'] = df.apply(review, axis =1)
df['Admit'].value_counts()


# ## Drop useless attributes

# In[1044]:


df = df.drop('Serial No.', axis = 1)
df = df.drop('TOEFL Score', axis = 1)
df = df.drop('University Rating', axis = 1)
df = df.drop('LOR ', axis = 1)
df = df.drop('Chance of Admit ', axis = 1)


# In[1045]:


df.head(10)


# ## Split attributes and label

# In[1046]:


x = df.drop('Admit', axis = 1)
y = df['Admit']


# ## Create decision tree

# In[1047]:


dtree = DecisionTreeClassifier(max_depth = 3)
dtree.fit(x, y)

dot_data = StringIO()
export_graphviz(dtree, 
                out_file=dot_data,  
                filled=True, 
                feature_names=list(x),
                class_names=['Reject','Admit'],
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("tree.pdf")


# ![tree.png](attachment:tree.png)

# # Compare my set of rules and the decision tree created by usnig sklearn 

# ## 1. Look over the structure of two trees

# ![compare.png](attachment:compare.png)

# ###  right hand side  :
# The order of attributes are same from top to buttom.
# ###  left hand side :
# SOP and Research are in reverse order.
# 
# ### The observations above will be discussed below

# ## 2. Find out useful informations like gini index, value, samples amount on my set of rules

# ###  Caculate gini at the block (CGPA <= 8)  and it's children

# In[1070]:


df['Admit'].value_counts()


# In[1050]:


df[ df['CGPA'] <= 8].count()


# In[1067]:


1-(((93/500)**2) + ((407/500)**2))


# Gini(parent) = 0.302

# In[1052]:


df_left = df[ df['CGPA'] <= 8]
df_left['Admit'].value_counts()


# In[1053]:


df_right = df[ df['CGPA'] > 8]
df_right['Admit'].value_counts()


# In[1054]:


1-(((27/93)**2) + (66/93)**2)


# gini(left) = 0.412

# In[1055]:


1-(((393/407)**2) + (14/407)**2)


# gini(right) = 0.066

# ### Caculate gini at the block (SOP <= 4) and it's children

# In[1057]:


df_left.count()


# In[1058]:


df_left[ df_left['SOP'] <=4 ].count()


# In[1059]:


df_L = df_left[ df_left['SOP'] <=4]
df_L['Admit'].value_counts()


# In[1060]:


df_R = df_left[ df_left['SOP'] >4 ]
df_R['Admit'].value_counts()


# In[1061]:


1-(((66/91)**2) + (25/91)**2)


# gini(left) = 0.399

# In[1062]:


1-(((0/2)**2) + (2/2)**2)


# gini(right) = 0.0

# ## 3. Compare two trees again with extra informations we already got.

# ![compare2.png](attachment:compare2.png)

# ### Calculate information gain from chosing SOP and Research as attribute in depth = 2 at left side

# In[1063]:


0.412-(((0.398)*(91) + (0)*(2))/93)


# IG(SOP<=4) = 0.022

# In[1005]:


0.438-(((0.101)*(75) + (0)*(30))/105)


# IG(Research<=0.5) = 0.366

# ### As you can see, information gain from (Research<=0.5)is greatly less than (SOP<=4).
# 
# 
# ### It means choosing (Research<=0.5) at this step is much more better.
# 
# ### That is why sklearn decsio tree algorithm chose this attribute, which is different from my own set of rules.
