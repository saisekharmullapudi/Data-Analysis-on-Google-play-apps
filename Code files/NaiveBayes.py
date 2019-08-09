import numpy as np
import pandas as pd
import pyphen
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score


import matplotlib.pyplot as plt

##################################################################################################################
path1 = "googleplaystore.csv"
apps = pd.read_csv(path1)

apps['Syllabul_Count'] = 0

def syllabu(str1):
    dic = pyphen.Pyphen(lang='en')
    # print(dic.inserted('str1'))
    st = dic.inserted(str1)
    p = len(dic.inserted(str1))
    count = 0;
    for i in range(0, p, 1):
        if (st[i] == '-'):
            count = count + 1
    return count+1



for i in range(0,len(apps),1):
    # a=f1.at[i,"App"]
    # a.replace(' ', '-')
    ret=syllabu(apps.at[i,"App"])
    apps.at[i,"Syllabul_Count"]=ret

##################################################################################################################




apps.dropna(inplace=True)


######################Vectorise Category############################################################################################
catval = apps['Category'].unique()
dict1 = {}
i = 0
for each in catval:
    dict1[each] = i
    i = i+1
# print(dict1)
apps["CategoryValues"] = apps["Category"].map(dict1).astype(int)


###################Installs count###############################################################################################
apps['Installs'] = [int(i[:-1].replace(',','')) for i in apps['Installs']]
installs = apps["Installs"]
installs = np.array(installs)
# print(installs)
def znormalization(ts):
    mean = ts.mean(axis = 0)
    std = ts.std(axis = 0)
    return (ts - mean) / std
normalized = znormalization(installs)
# print(normalized)

###################Size convesrion###############################################################################################

def convertsize(size1):
    if 'M' in size1:
        size = size1[:-1]
        size = float(size) * 1024
        return size
    elif 'K' in size1:
        size = size1[:-1]
        size = float(size) * 1
        return size
    else:
        return 1000
apps['Size'] = apps['Size'].map(convertsize)


######################Type############################################################################################

def typetobool(type):
    if type == "Free":
        return 0
    else:
        return 1
apps['Type'] = apps['Type'].map(typetobool)

#########################Price#########################################################################################

def convertprice(price):
    if "$" in price:
        price1 = price[1:]
        return float(price1)
    else:
        return float(price)
apps['Price'] = apps['Price'].map(convertprice)

#########################Reviews#########################################################################################
rev=apps['Reviews'].tolist()
# print(type(int(rev[4])))
results = list(map(int, rev))
# print(max(results))
# print(min(results))
def convertnum(num):
    if(num<10):
        return 0
    elif(num>=10 and num<100):
        return 1
    elif (num >= 100 and num < 1000):
        return 2
    elif (num >= 1000 and num < 10000):
        return 3
    elif (num >= 10000 and num < 100000):
        return 4
    elif (num >= 100000 and num < 1000000):
        return 5
    elif (num >= 1000000 and num < 5000000):
        return 6
    elif (num >= 5000000 and num < 10000000):
        return 7
    elif (num >= 10000000 and num < 20000000):
        return 8
    elif (num >= 20000000 and num < 40000000):
        return 9
    elif (num >= 40000000 and num < 60000000):
        return 10
    elif (num >= 60000000 and num < 80000000):
        return 11
    else:
        return 12
rev_num=[]
for each in results:
    rev_num.append(convertnum(each))

apps['Reviews']=rev_num


#########################Content Rating#########################################################################################
contrating=apps['Content Rating'].unique()
cont={}
j=0
for each in contrating:
    cont[each]=j
    j=j+1
apps['Content Rating']=apps['Content Rating'].map(cont).astype(int)

######################Genres############################################################################################

genreval = apps['Genres'].unique()
# print(genreval)
genresdict = {}
j = 0
for each in genreval:
    genresdict[each] = j
    j = j + 1
apps['GenreValues'] = apps['Genres'].map(genresdict).astype(int)


##################################################################################################################


# print(apps)
app1=apps.copy(deep=False)
nam=list(app1.columns.values)

print(nam)
app_y=app1['CategoryValues']
# print(app_y.head())


app_x=app1.drop(['App','Category','CategoryValues','Android Ver','Last Updated','Genres','Current Ver','Installs','Size',  'Content Rating', 'Type',],axis=1)
na=list(app_x.columns.values)
print(na)
##################################################################################################################

model = GaussianNB()
X=np.array(app_x)
Y=np.array(app_y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model.fit(X_train, y_train)

predicted= model.predict(X_test)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, predicted)*100)
######################################################################################################################

# app2=apps.copy(deep=False)
# nam=list(app2.columns.values)
#
# # print(nam)
# app_y1=app2['CategoryValues']
# print(app_y.head())


# app_x1=app2.drop(['App','Category','CategoryValues','Android Ver', 'Last Updated','Genres','Current Ver', 'Size','GenreValues' ],axis=1)
# na=list(app_x1.columns.values)
# print(na)
# X1=np.array(app_x1)
# Y1=np.array(app_y1)
#
# X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.2)
#
#
#
# #########################################################################################################################
#
# classifier = KNeighborsClassifier(n_neighbors=4)
# classifier.fit(X_train1, y_train1)
# y_pred1 = classifier.predict(X_test1)
# con_mat = accuracy_score(y_test1,y_pred1)
# print("KNN model accuracy",con_mat*100)
#########################################################################
############################################################################
