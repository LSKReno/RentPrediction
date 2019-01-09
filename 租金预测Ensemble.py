
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


# In[4]:


qu_dummies = pd.get_dummies(house.区)
quTest_dummies = pd.get_dummies(houseTest.区)

house = house.join(qu_dummies)
houseTest = houseTest.join(quTest_dummies)


# In[5]:



print("-"*50 + "train查看缺失值" + "-"*50)
print(house.isnull().sum().sort_values(ascending=False).head(10))
print("-"*50 + "test查看缺失值" + "-"*50)
print(houseTest.isnull().sum().sort_values(ascending=False).head(10))


# In[6]:


house['装修情况'] = house['装修情况'].fillna(0)
house['居住状态'] = house['居住状态'].fillna(0)
house['出租方式'] = house['出租方式'].fillna(-9999)
house['距离'] = house['距离'].fillna(0)
house['地铁站点'] = house['地铁站点'].fillna(0)
house['地铁线路'] = house['地铁线路'].fillna(0)

houseTest['装修情况'] = houseTest['装修情况'].fillna(0)
houseTest['居住状态'] = houseTest['居住状态'].fillna(0)
houseTest['出租方式'] = houseTest['出租方式'].fillna(-9999)
houseTest['距离'] = houseTest['距离'].fillna(0)
houseTest['地铁站点'] = houseTest['地铁站点'].fillna(0)
houseTest['地铁线路'] = houseTest['地铁线路'].fillna(0)


# In[7]:


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn import model_selection

houseTest = houseTest.drop(["区"],axis=1)
x = house.drop(["月租金","区"],axis=1)
y = house.月租金

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=10)


# In[8]:


from sklearn.model_selection import KFold

SEED = 1314
NFOLDS = 10
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

ntrain = x_train.shape[0]
ntest = houseTest.shape[0]
print('ntrain: '+str(ntrain) +'   '+ 'ntest: '+str(ntest))

def get_out_fold(clf, x_train, y_train, houseTest):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    for i,(train_index, test_index) in enumerate(kf.split(x_train) ):
        print('train_index:%s , test_index: %s ' %(train_index,test_index))
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(houseTest)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


SEED = 1314
NFOLDS = 10
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

ntrain1 = x_train.shape[0]
ntest1 = x_test.shape[0]
print('ntrain1: '+str(ntrain1) +'   '+ 'ntest1: '+str(ntest1))

def get_out_fold1(clf, x_train, y_train, houseTest):
    oof_train = np.zeros((ntrain1,))
    oof_test = np.zeros((ntest1,))
    oof_test_skf = np.empty((NFOLDS, ntest1))
    
    for i,(train_index, test_index) in enumerate(kf.split(x_train) ):
        print('train_index:%s , test_index: %s ' %(train_index,test_index))
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(houseTest)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[9]:


from sklearn import model_selection
from lightgbm import LGBMRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# rf = RandomForestRegressor(n_estimators=500, max_features='sqrt',max_depth=20,
#                             min_samples_split=6, min_samples_leaf=5, n_jobs=4, verbose=0)

# et = ExtraTreesRegressor(n_estimators=500, n_jobs=4, max_depth=20, min_samples_leaf=6, verbose=0)

# gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.2, min_samples_split=5,
#                                 min_samples_leaf=6, max_depth=20, verbose=0)
print("Layer1 Start:")

# cb = CatBoostRegressor(iterations=500,loss_function='RMSE',custom_metric='RMSE',eval_metric='RMSE',
#                        leaf_estimation_method='Gradient',depth=10,learning_rate=0.20)

# BestParams: {'colsample_bytree': 0.6, 'gamma': 0.9, 
# #              'learning_rate': 0.15, 'max_depth': 25, 'min_child_weight': 7, 'n_estimators': 800, 'subsample': 0.95}
xgb = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=25, min_child_weight= 6, gamma=0.92, subsample=0.95,
                  colsample_bytree=0.6, nthread=-1)

lgb = LGBMRegressor(n_estimators=800,learning_rate=0.15,max_depth=30,bagging_fraction=0.7,
                    max_bin=300,num_leaves=850,alpha=0.06)

x_train = pd.DataFrame(x_train.values)
y_train = pd.DataFrame(y_train.values)
x_test = pd.DataFrame(x_test.values)
y_test = pd.DataFrame(y_test.values)
houseTest = pd.DataFrame(houseTest.values)

# print("x_train: ")
# print(x_train)
xgb_oof_train, xgb_oof_test = get_out_fold(xgb, x_train, y_train, houseTest) # XGBboost
# cb_oof_train, cb_oof_test = get_out_fold(cb, x_train, y_train, houseTest) # CatBoost
lgb_oof_train, lgb_oof_test = get_out_fold(lgb, x_train, y_train, houseTest) # LGBboost


# In[10]:


# 为了得到维度变换后的x_test
xgb_oof_train1, xgb_oof_test1 = get_out_fold1(xgb, x_train, y_train, x_test) # XGBboost
lgb_oof_train1, lgb_oof_test1 = get_out_fold1(lgb, x_train, y_train, x_test) # LGBboost

print("Training is completed.")
x_train = np.concatenate((xgb_oof_train,lgb_oof_train), axis=1)
houseTest = np.concatenate((xgb_oof_test,lgb_oof_test), axis=1)

x_test = np.concatenate((xgb_oof_test1,lgb_oof_test1), axis=1)

print("Layer1 Finished")


# In[19]:


# x_train = np.concatenate((xgb_oof_train,lgb_oof_train), axis=1)
# houseTest = np.concatenate((xgb_oof_test,lgb_oof_test), axis=1)
# x_test = np.concatenate((xgb_oof_test1,lgb_oof_test1), axis=1)

x_train = pd.DataFrame(x_train)
houseTest = pd.DataFrame(houseTest)
x_test = pd.DataFrame(x_test)


# In[20]:


x_train.to_csv('x_train.csv')
houseTest.to_csv('houseTest.csv')
x_test.to_csv('x_test.csv')


# In[21]:


# 将其求平均值
x_train = pd.read_csv('x_train.csv')
houseTest = pd.read_csv('houseTest.csv')
x_test = pd.read_csv('x_test.csv')

x_train["2"] = (x_train["0"]+x_train["1"] )/2
houseTest["2"] = (houseTest["0"]+houseTest["1"] )/2
x_test["2"] = (x_test["0"]+x_test["1"] )/2


# In[183]:


print("Layer2 Start:")
print("Finding best parameters:")

stacker_est =LGBMRegressor(random_state=1314)

stacker_param_grid = {'n_estimators': [800],
                  'learning_rate': [0.013],
                 'max_depth':[3],
                 'bagging_fraction':[0.6],
                  'max_bin':[10000],
                  'num_leaves':[6],
                  'alpha':[0.06]
                 }

stacker_grid = model_selection.GridSearchCV(stacker_est, stacker_param_grid, n_jobs=4, cv=2, verbose=1)
print("fitting")
stacker_grid.fit(x_train, y_train)

print('BestParams: ' + str(stacker_grid.best_params_))
print('Training:')
stacker = LGBMRegressor(n_estimators = stacker_grid.best_estimator_.n_estimators,
                   learning_rate = stacker_grid.best_estimator_.learning_rate,
                    max_depth = stacker_grid.best_estimator_.max_depth,
                    bagging_fraction = stacker_grid.best_estimator_.bagging_fraction,
                    max_bin = stacker_grid.best_estimator_.max_bin,
                    num_leaves = stacker_grid.best_estimator_.num_leaves,
                    alpha = stacker_grid.best_estimator_.alpha,
                   random_state=1314)
print('Training Finished')
# stacker = XGBRegressor(n_estimators=500, max_depth= 20, min_child_weight= 10,gamma=0.9, subsample=0.95,
#                   colsample_bytree=0.6, nthread=-1)
# lgb = LGBMRegressor(n_estimators=1000,learning_rate=0.3,max_depth=30,bagging_fraction=0.6,
#                     max_bin=200,num_leaves=200,alpha=0.08)

stacker.fit(x_train, y_train)
print("Layer2 Finished")


# In[184]:


print("评分：")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(stacker,x_test,y_test)
print("Boost: ",scores.mean())
from sklearn import metrics

y_pred_xgb = stacker.predict(x_test)
# 均方根误差
print(metrics.mean_squared_error(y_test,y_pred_xgb))


# In[185]:


predictions = stacker.predict(houseTest.astype(float))
Submission = pd.DataFrame({'id': Id, 
                           'price': predictions})
Submission.to_csv('StackingSubmission.csv',index=False,sep=',')
print("I LOVE YOUU Stacking")


# In[14]:


# clf_est =XGBRegressor(random_state=11)
# clf_param_grid = {'n_estimators': [500,800],
#                   'learning_rate': [0.2], 
#                  'max_depth': [20], 
#                   'min_child_weight': [10], 
#                   'gamma':[0.9,0.92], 
#                   'subsample':[0.95],
#                   'colsample_bytree':[0.6]}
# clf_grid = model_selection.GridSearchCV(clf_est, clf_param_grid, n_jobs=3, cv=2, verbose=1)
# print("fitting")
# clf_grid.fit(x_train, y_train)

# print('BestParams: ' + str(clf_grid.best_params_))
# clf = XGBRegressor(n_estimators = clf_grid.best_estimator_.n_estimators,
#                    learning_rate = clf_grid.best_estimator_.learning_rate,
#                    max_depth = clf_grid.best_estimator_.max_depth,
#                    min_child_weight = clf_grid.best_estimator_.min_child_weight,
#                    gamma = clf_grid.best_estimator_.gamma,
#                    subsample = clf_grid.best_estimator_.subsample,
#                    colsample_bytree = clf_grid.best_estimator_.colsample_bytree,
#                    random_state=11)
# clf.fit(x_train, y_train)

# # clf = joblib.load('clf2.09.pkl')    # 下载模型
# # print(clf)

# print("评分：")
# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(clf,x_test,y_test)
# print("Boost: ",scores.mean())


# In[15]:


# predictions = clf.predict(houseTest.astype(float))
# Submission = pd.DataFrame({'id': Id, 
#                            'price': predictions})
# Submission.to_csv('Submission.csv',index=False,sep=',')
# print("I LOVE YOUU")

# from sklearn import metrics

# y_pred_xgb = clf.predict(x_test)
# # 均方根误差
# print(metrics.mean_squared_error(y_test,y_pred_xgb))


# In[16]:


# from sklearn.externals import joblib

# # joblib.dump(clf, 'clf2.09.pkl')       # 保存模型，需先建立saved_model文件夹

# # clf = joblib.load('clf2.09.pkl')    # 下载模型
# # print(clf)

