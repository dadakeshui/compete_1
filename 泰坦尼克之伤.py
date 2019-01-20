import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from numpy import float16
from sklearn import linear_model


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Embarked','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Embarked.notnull()].values
    unknown_age = age_df[age_df.Embarked.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Embarked.isnull()), 'Embarked' ] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = 1
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = 0
    return df

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages2(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass','Embarked']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr


def set_missing_ages3(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Fare','Sex', 'Parch', 'SibSp','Embarked', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Fare.notnull()].values
    unknown_age = age_df[age_df.Fare.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Fare.isnull()), 'Fare' ] = predictedAges

    return df, rfr






#看一下数据，统计一下，在进行图形分析
data_train = pd.read_csv("H:/kaggle/train.csv")  #读取训练数据


dict = {"S":1, "C":2,"Q":3}
data_train.Embarked = [dict[x] if x in dict else x for x in data_train.Embarked]
'''
data_train.loc[ (data_train.Embarked == "S"), 'Embarked' ] = int("1")
data_train.loc[ (data_train.Embarked == "C"), 'Embarked' ] = int("2")
data_train.loc[ (data_train.Embarked == "Q"), 'Embarked' ] = int("3")

'''
dict2 = {"male":1, "female":2}
data_train.Sex = [dict2[x] if x in dict2 else x for x in data_train.Sex]



#处理数据完成
data_train, rfr = set_missing_ages(data_train)
data_train, rfr = set_missing_ages2(data_train)
data_train = set_Cabin_type(data_train)

data_train.loc[ (data_train.Embarked.isnull()), 'Embarked' ] = 0


pd.set_option('display.width',None)
print(data_train)
print(data_train.info())




df = data_train
df.drop([ 'Name', 'Ticket'], axis=1, inplace=True)

known_age = df[df.Age.notnull()].values
y_age = known_age[:, 4].reshape(-1, 1)
x_fare = known_age[:, 7].reshape(-1, 1)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(y_age)
x1= scaler.fit_transform(y_age, age_scale_param)
fare_scale_param = scaler.fit(x_fare)
x2 = scaler.fit_transform(x_fare, fare_scale_param)

df.loc[ (df.Age.notnull()), 'Age' ] = x1.reshape(-1, 1)
df.loc[ (df.Fare.notnull()), 'Fare' ] = x2.reshape(-1, 1)


#train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_df = df[['Survived','Age','SibSp','Parch', 'Fare', 'Cabin','Embarked','Sex', 'Pclass']]
train_np = train_df.values


y = train_np[:, 0]

# X即特征属性值
X = train_np[:,1::]
#X.round(5)


clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
print(clf)


data_test = pd.read_csv("H:/kaggle/test.csv")  #读取训练数据

dict = {"S":1, "C":2,"Q":3}
data_train.Embarked = [dict[x] if x in dict else x for x in data_train.Embarked]


data_test.loc[ (data_test.Embarked == "S"), 'Embarked' ] = 1
data_test.loc[ (data_test.Embarked == "C"), 'Embarked' ] = 2
data_test.loc[ (data_test.Embarked == "Q"), 'Embarked' ] = 3



dict3 = {"male":1, "female":2}
data_test.Sex = [dict3[x] if x in dict3 else x for x in data_test.Sex]

data_test, rfr = set_missing_ages3(data_test)
data_test, rfr = set_missing_ages2(data_test)
data_test = set_Cabin_type(data_test)

dict = {"1":1, "2":2,"3":3}
data_train.Embarked = [dict[x] if x in dict else x for x in data_train.Embarked]


df2 = data_test
df2.drop([ 'Name', 'Ticket'], axis=1, inplace=True)


known_age2 = df2[df2.Age.notnull()].values
y_age2 = known_age2[:, 4].reshape(-1, 1)
x_fare2 = known_age2[:, 7].reshape(-1, 1)

scaler2 = preprocessing.StandardScaler()
age_scale_param2 = scaler2.fit(y_age2)
x1_2= scaler2.fit_transform(y_age2, age_scale_param2)
fare_scale_param2 = scaler2.fit(x_fare2)
x2_2 = scaler2.fit_transform(x_fare2, fare_scale_param2)

df2.loc[ (df2.Age.notnull()), 'Age' ] = x1_2.reshape(-1, 1)
df2.loc[ (df2.Fare.notnull()), 'Fare' ] = x2_2.reshape(-1, 1)


test_df = df2[['Age','SibSp','Parch', 'Fare', 'Cabin','Embarked','Sex', 'Pclass']]
test_np = test_df.values

predictions = clf.predict(test_np)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("H:/kaggle/logistic_regression_predictionss.csv", index=False)



'''
#显示数据
pd.set_option('display.width',None)
print(data_train)
print(data_train.info())
'''






