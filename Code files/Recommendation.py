# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

path1 = "googleplaystore.csv"
apps = pd.read_csv(path1)
apps.dropna(inplace=True)

n_items_base = apps['Size']

# Getting unique values for all features
unique_category = apps['Category'].unique()
unique_genre = apps['Genres'].unique()
unique_type = apps['Type'].unique()
unique_content = apps['Content Rating'].unique()
unique_installs = apps['Installs'].unique()

#Number of unique values for each feature
n_category = len(unique_category)
n_genre = len(unique_genre)
n_type = len(unique_type)
n_content = len(unique_content)
n_installs = len(unique_installs)

number_features = n_category + n_genre + n_type + n_content + n_installs + 10;
# print(number_feature)
# print(n_category)
# print(n_genre)
# print(n_type)
# print(n_content)
# print(n_installs)

# Building the utility matrix
utility_matrix = np.zeros((number_features, n_items_base.size))
i = 0
for each in apps['App']:
    utility_matrix[unique_category.tolist().index(apps['Category'].iat[i])][i] = 1
    utility_matrix[unique_genre.tolist().index(apps['Genres'].iat[i])+n_category][i] = 1
    utility_matrix[unique_type.tolist().index(apps['Type'].iat[i])+n_category+n_genre][i] = 1
    utility_matrix[unique_content.tolist().index(apps['Content Rating'].iat[i]) + n_category + n_genre + n_type][i] = 1
    if apps['Rating'].iat[i] > 0 and apps['Rating'].iat[i] < 0.5:
        utility_matrix[n_category + n_genre + n_type + n_content][i] = 10
    if apps['Rating'].iat[i] >= 0.5 and apps['Rating'].iat[i] < 1:
            utility_matrix[n_category + n_genre + n_type + n_content + 1][i] = 5
    if apps['Rating'].iat[i] >= 1 and apps['Rating'].iat[i] < 1.5:
            utility_matrix[n_category + n_genre + n_type + n_content + 2][i] = 1
    if apps['Rating'].iat[i] >= 1.5 and apps['Rating'].iat[i] < 2:
            utility_matrix[n_category + n_genre + n_type + n_content + 3][i] = 1
    if apps['Rating'].iat[i] >= 2 and apps['Rating'].iat[i] < 2.5:
            utility_matrix[n_category + n_genre + n_type + n_content + 4][i] = 1
    if apps['Rating'].iat[i] >= 2.5 and apps['Rating'].iat[i] < 3:
            utility_matrix[n_category + n_genre + n_type + n_content + 5][i] = 1
    if apps['Rating'].iat[i] >= 3 and apps['Rating'].iat[i] < 3.5:
            utility_matrix[n_category + n_genre + n_type + n_content + 6][i] = 1
    if apps['Rating'].iat[i] >= 3.5 and apps['Rating'].iat[i] < 4:
            utility_matrix[n_category + n_genre + n_type + n_content + 7][i] = 1
    if apps['Rating'].iat[i] >= 4 and apps['Rating'].iat[i] < 4.5:
            utility_matrix[n_category + n_genre + n_type + n_content + 8][i] = 1
    if apps['Rating'].iat[i] >= 4.5 and apps['Rating'].iat[i] < 5:
            utility_matrix[n_category + n_genre + n_type + n_content + 9][i] = 1
    utility_matrix[unique_installs.tolist().index(apps['Installs'].iat[i]) + n_category + n_genre + n_type + 10][i] = 1
    i = i + 1

utility_matrix = utility_matrix.T
# print(utility_matrix.shape)

n_top_similar_apps = 15
input_app_name = input("Input the app name: ")
inp_app_index = apps['App'].tolist().index(input_app_name)
n_recommendations = int(input("Input the number of recommendations: "))
item_similarity = cosine_similarity(utility_matrix, Y=None)
similar_n = item_similarity.argsort()[:,-n_top_similar_apps:][:,::-1]

recommended_apps_indexes = similar_n[inp_app_index]
print('Top ', n_recommendations,' recommended apps:')
recommended_apps_list = []
for each in recommended_apps_indexes:
    recommended_apps_list.append(apps['App'].iat[each])
#print(recommended_apps_list)
#print(recommended_apps_list)
# unique_recommended_apps_list = []
# for x in recommended_apps_list:
#         if x not in unique_recommended_apps_list and x!=apps['App'].iat[inp_app_index]:
#             unique_recommended_apps_list.append(x)

for i in range (1,n_recommendations+1):
    print(recommended_apps_list[i])

