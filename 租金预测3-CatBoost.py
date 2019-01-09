
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

house = pd.read_csv("train.csv")
houseTest = pd.read_csv("test.csv")

Id = houseTest["id"]
houseTest = houseTest.drop(["id"],axis=1)


# house
print("-"*50+ "train简单查看统计学信息" +"-"*50)
print(house.info())
print("-"*50+ "test简单查看统计学信息" +"-"*50)
print(houseTest.info())

print("-"*50 + "train查看缺失值" + "-"*50)
print(house.isnull().sum().sort_values(ascending=False).head(10))
print("-"*50 + "test查看缺失值" + "-"*50)
print(houseTest.isnull().sum().sort_values(ascending=False).head(10))


# In[2]:


house["位置"][house["位置"].isnull()] = house["位置"].dropna().mode().values
house["区"][house["区"].isnull()] = house["区"].dropna().mode().values
house["小区房屋出租数量"][house["小区房屋出租数量"].isnull()] = house["小区房屋出租数量"].dropna().mode().values
houseTest["位置"][houseTest["位置"].isnull()] = houseTest["位置"].dropna().mode().values
houseTest["区"][houseTest["区"].isnull()] = houseTest["区"].dropna().mode().values
houseTest["小区房屋出租数量"][houseTest["小区房屋出租数量"].isnull()] = houseTest["小区房屋出租数量"].dropna().mode().values

print("-"*50 + "train查看缺失值" + "-"*50)
print(house.isnull().sum().sort_values(ascending=False).head(10))
print("-"*50 + "test查看缺失值" + "-"*50)
print(houseTest.isnull().sum().sort_values(ascending=False).head(10))


# In[3]:


# 装修情况肯定对租金价格有较大影响，我们现在有两万的装修数据，可以尝试下预测
# 居住状态不太好预测，这个属于人为因素，但是应该有人居住的房子一定租金较低
# 出租方式，妥妥的有影响，中介收费不一，朋友介绍肯定便宜
# 距离应该是地铁站点到房子的距离，交通的影响
# 地铁站这个不太好预测，因为住房附近可能压根没地铁
# 小区出租数量，位置，区 可以直接干掉


# In[4]:


# 特征工程

# 房屋朝向
print("-"*50+"house房屋朝向"+"-"*50)
house_orientation_unique = house["房屋朝向"].unique()
print(house["房屋朝向"].unique())
# print(house["房屋朝向"].value_counts())
print("-"*50+"houseTest房屋朝向"+"-"*50)
houseTest_orientation_unique = houseTest["房屋朝向"].unique()
print(houseTest["房屋朝向"].unique())
# print(houseTest["房屋朝向"].value_counts())
print()
print("house 有 houseTest没有:")
print ([i for i in house_orientation_unique if i not in houseTest_orientation_unique])
print("house 没有 houseTest有:")
print ([i for i in houseTest_orientation_unique if i not in house_orientation_unique])
print("house 有 houseTest有:")
print ([i for i in houseTest_orientation_unique if i in house_orientation_unique])

house_orientation = pd.DataFrame({'东':[0]*196539,'南':[0]*196539,
                                 '西':[0]*196539,'北':[0]*196539,
                                 '东南':[0]*196539,'东北':[0]*196539,
                                 '西南':[0]*196539,'西北':[0]*196539,})
houseTest_orientation = pd.DataFrame({'东':[0]*196539,'南':[0]*196539,
                                 '西':[0]*196539,'北':[0]*196539,
                                 '东南':[0]*196539,'东北':[0]*196539,
                                 '西南':[0]*196539,'西北':[0]*196539,})
house = house.join(house_orientation)
# house

# 将房屋朝向整成八个特征
def orientation_dong(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    dong = "东"
    if dong in lst:
        return 1
    else:
        return 0

def orientation_nan(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    nan = "南"
    if nan in lst:
        return 1
    else:
        return 0

    
def orientation_xi(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    xi = "西"
    if xi in lst:
        return 1
    else:
        return 0


def orientation_bei(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    bei = "北"
    if bei in lst:
        return 1
    else:
        return 0

    
def orientation_dongNan(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    dongNan = "东南"
    if dongNan in lst:
        return 1
    else:
        return 0

    
def orientation_dongBei(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    dongBei = "东北"
    if dongBei in lst:
        return 1
    else:
        return 0

    
def orientation_xiNan(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    xiNan = "西南"
    if xiNan in lst:
        return 1
    else:
        return 0

    
def orientation_xiBei(x):
    lst = []
    for i in x.split(" "):
        lst.append(i)
    xiBei = "西北"
    if xiBei in lst:
        return 1
    else:
        return 0

    
house['东'] = house['房屋朝向'].apply(lambda x : orientation_dong(x) )
house['南'] = house['房屋朝向'].apply(lambda x : orientation_nan(x) )
house['西'] = house['房屋朝向'].apply(lambda x : orientation_xi(x) )
house['北'] = house['房屋朝向'].apply(lambda x : orientation_bei(x) )
house['东南'] = house['房屋朝向'].apply(lambda x : orientation_dongNan(x) )
house['东北'] = house['房屋朝向'].apply(lambda x : orientation_dongBei(x) )
house['西南'] = house['房屋朝向'].apply(lambda x : orientation_xiNan(x) )
house['西北'] = house['房屋朝向'].apply(lambda x : orientation_xiBei(x) )

houseTest['东'] = houseTest['房屋朝向'].apply(lambda x : orientation_dong(x) )
houseTest['南'] = houseTest['房屋朝向'].apply(lambda x : orientation_nan(x) )
houseTest['西'] = houseTest['房屋朝向'].apply(lambda x : orientation_xi(x) )
houseTest['北'] = houseTest['房屋朝向'].apply(lambda x : orientation_bei(x) )
houseTest['东南'] = houseTest['房屋朝向'].apply(lambda x : orientation_dongNan(x) )
houseTest['东北'] = houseTest['房屋朝向'].apply(lambda x : orientation_dongBei(x) )
houseTest['西南'] = houseTest['房屋朝向'].apply(lambda x : orientation_xiNan(x) )
houseTest['西北'] = houseTest['房屋朝向'].apply(lambda x : orientation_xiBei(x) )

house = house.drop("房屋朝向",axis=1)
houseTest = houseTest.drop("房屋朝向",axis=1)
# house


# In[5]:


print("-"*50 + "train查看缺失值" + "-"*50)
print(house.isnull().sum().sort_values(ascending=False).head(10))
print("-"*50 + "test查看缺失值" + "-"*50)
print(houseTest.isnull().sum().sort_values(ascending=False).head(10))


# In[6]:


# house['装修情况'] = house['装修情况'].fillna(house['装修情况'].mean())
# house['居住状态'] = house['居住状态'].fillna(house['居住状态'].mean())
# house['出租方式'] = house['出租方式'].fillna(house['出租方式'].mean())
# house['距离'] = house['距离'].fillna(0)
# house['地铁站点'] = house['地铁站点'].fillna(0)
# house['地铁线路'] = house['地铁线路'].fillna(0)

# houseTest['装修情况'] = houseTest['装修情况'].fillna(houseTest['装修情况'].mean())
# houseTest['居住状态'] = houseTest['居住状态'].fillna(houseTest['居住状态'].mean())
# houseTest['出租方式'] = houseTest['出租方式'].fillna(houseTest['出租方式'].mean())
# houseTest['距离'] = houseTest['距离'].fillna(0)
# houseTest['地铁站点'] = houseTest['地铁站点'].fillna(0)
# houseTest['地铁线路'] = houseTest['地铁线路'].fillna(0)


# In[7]:


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn import model_selection

x = house.drop(["月租金"],axis=1)
y = house.月租金

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=10)


# In[ ]:


from catboost import CatBoostRegressor

clf_est =CatBoostRegressor(random_seed=1314)
# iterations=500,loss_function='RMSE',custom_metric='RMSE',eval_metric='RMSE',
#                        leaf_estimation_method='Gradient',depth=10,learning_rate=0.20
clf_param_grid = {'learning_rate':[0.15,0.2],
                  'loss_function': ['RMSE'], # 'Logloss' 
                 'custom_metric': ['RMSE'], 
                  'eval_metric': ['RMSE'], 
                  'leaf_estimation_method':['Newton'], 
                  'depth':[10]}
clf_grid = model_selection.GridSearchCV(clf_est, clf_param_grid, n_jobs=3, cv=3, verbose=1)
print("fitting")
clf_grid.fit(x_train, y_train)

print('BestParams: ' + str(clf_grid.best_params_))


# In[16]:


print('Training:')
clf = CatBoostRegressor(
                    learning_rate = clf_grid.best_estimator_.learning_rate_,
                   loss_function = 'RMSE',
                   custom_metric = 'RMSE',
                   eval_metric = 'RMSE',
                   leaf_estimation_method = 'Newton',
                   random_seed=1314)
clf.fit(x_train, y_train)
print('Training Finished')


# In[17]:


predictions = clf.predict(houseTest.astype(float))
Submission = pd.DataFrame({'id': Id, 
                           'price': predictions})
Submission.to_csv('Submission.csv',index=False,sep=',')
print("I LOVE YOUU")


print("评分：")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf,x_test,y_test)
print("Boost Score: ",scores.mean())
from sklearn import metrics

y_pred_xgb = clf.predict(x_test)
# 均方根误差
print("均方根误差: ",metrics.mean_squared_error(y_test,y_pred_xgb))


# In[ ]:




# clf = joblib.load('clf2.09.pkl')    # 下载模型
# print(clf)

# print("评分：")
# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(clf,x_test,y_test)
# print("Boost: ",scores.mean())


# In[ ]:


from sklearn.externals import joblib

# joblib.dump(clf, 'clf2.09.pkl')       # 保存模型，需先建立saved_model文件夹

# clf = joblib.load('clf2.09.pkl')    # 下载模型
# print(clf)

