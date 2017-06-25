
# coding: utf-8

# In this notebook, we will try and explore the basic information about the dataset given. The dataset for this competition is a relational set of files describing customers' orders over time. 
# 
# **Objective:** 
# 
# The goal of the competition is to predict which products will be in a user's next order. The dataset is anonymized and contains a sample of over 3 million grocery orders from more than 200,000 Instacart users.
# 
# For each user, 4 and 100 of their orders are given, with the sequence of products purchased in each order
# 
# Let us start by importing the necessary modules.

# In[1]:

import numpy as np # linear algebra
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
from itertools import *
from collections import Counter

# In[14]:

order_products_train_df = pd.read_csv("input/order_products__train.csv")
order_products_prior_df = pd.read_csv("input/order_products__prior.csv")
orders_df = pd.read_csv("input/orders.csv")
products_df = pd.read_csv("input/products.csv")
aisles_df = pd.read_csv("input/aisles.csv")
departments_df = pd.read_csv("input/departments.csv")
sample_sub= pd.read_csv("input/sample_submission.csv")


order_products_prior_df
print('computing prior product info')
prods = pd.DataFrame()
prods['orders'] = order_products_prior_df.groupby(order_products_prior_df.product_id).size().astype(np.int32)
prods['reorders'] = order_products_prior_df['reordered'].groupby(order_products_prior_df.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products_df.join(prods, on='product_id')
#products.set_index('product_id', drop=False, inplace=True)
del products['product_name']
del prods
print(order_products_train_df.shape)
order_products_train_df = order_products_train_df.join(products,on=['product_id'],how='left', lsuffix='_left', rsuffix='_right')
del order_products_train_df['product_id_left']
order_products_train_df['product_id']=order_products_train_df['product_id_right']
del order_products_train_df['product_id_right']

#--------------GLOBAL EDGELIST GENERATOR---------------
#temp=order_products_prior_df[['order_id','product_id']]
#global_order_info=temp.groupby('order_id')
#edgelist_full=[]
#for name, group in global_order_info:
#    items= np.sort(list(group['product_id']))
#    for edge in combinations(items, 2):
#        edgelist_full.append(edge)

#pickle.dump( edgelist_full, open( "edgelist_full.p", "wb" ) )
#print(order_products_train_df.shape)
edgelist_full=pickle.load("edgelist_full.p")


print('add order info to priors')
orders_df.set_index('order_id', inplace=True, drop=False)
order_products_prior_df = order_products_prior_df.join(orders_df, on='order_id', rsuffix='_')
order_products_prior_df.drop('order_id_', inplace=True, axis=1)


print('computing prior user info')
usr = pd.DataFrame()
usr['average_days_between_orders'] = orders_df.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders_df.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
users['total_items'] = order_products_prior_df.groupby('user_id').size().astype(np.int16)
users['all_products'] = order_products_prior_df.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print('user f', users.shape)

#del orders_df['order_id']
#orders_df=orders_df.reset_index()

print('computing order info')
print(order_products_train_df.shape)
print('join order table')
order_products_train_df = order_products_train_df.join(orders_df,on=['order_id'],how='left', lsuffix='_left', rsuffix='_right')
del order_products_train_df['order_id_left']
order_products_train_df['order_id']=order_products_train_df['order_id_right']
del order_products_train_df['order_id_right']
print(order_products_train_df.shape)
del order_products_train_df['eval_set']
del order_products_train_df['days_since_prior_order']
print('join user table')
order_products_train_df = order_products_train_df.join(users,on=['user_id'],how='left', lsuffix='_left', rsuffix='_right')


y=order_products_train_df['reordered'].values
del order_products_train_df['reordered']
del order_products_train_df['order_id']
del order_products_train_df['product_id']
del order_products_train_df['user_id']
del order_products_train_df['all_products']


X=order_products_train_df.values
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dtrain = xgb.DMatrix(X_train)
dtrain.set_label(y_train)
dtest = xgb.DMatrix(X_test)

param = {'eta': 0.03, 'silent': 1, 'eval_metric': 'logloss', 'max_depth':10, 'n_estimators': 1000,
          'objective': 'binary:logistic','updater':'grow_gpu_hist'}
bst = xgb.train(param, dtrain,num_boost_round=2000)

#model = XGBClassifier()
#model.fit(X_train, y_train)
y_pred_train = bst.predict(dtrain)
y_pred = bst.predict(dtest)
#92aba2137
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("log error train", log_loss(y_train, y_pred_train))
print("log error", log_loss(y_test, y_pred))