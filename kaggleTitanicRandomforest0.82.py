import numpy as np
import pandas as pd

# Go to philipkalinda.com
print('Reading Data...')
train_data = pd.read_csv("./tai/train.csv", dtype={"Age": np.float64}, )
test_data = pd.read_csv("./tai/test.csv", dtype={"Age": np.float64}, )#浮点化

import re
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor


# Functions

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


print('Cleaning Data...')#阶段打印非常好

combined2 = pd.concat([train_data, test_data], axis=0)#统一处理
combined2.Embarked.fillna('S', inplace=True)#填充
combined2.Fare.fillna(np.median(combined2.Fare[combined2.Fare.notnull()]), inplace=True)#费用中位数
combined2['Title'] = combined2["Name"].apply(get_title)#获得名字特征
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 7, "Dona": 10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7,
                 "Ms": 2}#类别化
combined2["TitleCat"] = combined2.loc[:, 'Title'].map(title_mapping)
combined2['CabinCat'] = pd.Categorical.from_array(combined2.Cabin.fillna('0').apply(lambda x: x[0])).codes#类别化
combined2.Cabin.fillna('0', inplace=True)
combined2['EmbarkedCat'] = pd.Categorical.from_array(combined2.Embarked).codes
combined2.drop(['Ticket'], axis=1, inplace=True)#删除



print('Consolidating Data...')

full_data = pd.concat([combined2.drop(['Survived'], axis=1),
                       pd.get_dummies(combined2.Sex, prefix='Sex'),#离散化并添入sex
                       combined2.Survived], axis=1)

print('Generating Features...')

full_data['FamilySize'] = full_data["SibSp"] + full_data["Parch"] #合归为家庭规模
full_data['NameLength'] = full_data.Name.apply(lambda x: len(x))  #增加姓名长度特征

import operator

family_id_mapping = {}


def get_family_id(row):#计算家庭
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


family_ids = full_data.apply(get_family_id, axis=1)
# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[full_data["FamilySize"] < 3] = -1 #将人少的归成一类id
full_data["FamilyId"] = family_ids #按家庭归类

#### Person Label
child_age = 14


def get_person(passenger):#分成男女小孩
    age, sex = passenger
    if (age < child_age):
        return 'child'
    elif (sex == 'female'):
        return 'female_adult'
    else:
        return 'male_adult'


full_data = pd.concat(
    [full_data, pd.DataFrame(full_data[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])], axis=1)#新增特征
dummies = pd.get_dummies(full_data['person'])#独热
full_data = pd.concat([full_data, dummies], axis=1)


def process_surname(nm):
    return nm.split(',')[0].lower()


full_data['surname'] = full_data['Name'].apply(process_surname)

#### Persihing Females
perishing_female_surnames = list(set(full_data[(full_data.female_adult == 1.0) &
                                               (full_data.Survived == 0.0) &
                                               ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname'].values))#通过死去有家人的女性形成特征


def perishing_mother_wife(passenger):
    surname, Pclass, person = passenger
    return 1.0 if (surname in perishing_female_surnames) else 0.0


full_data['perishing_mother_wife'] = full_data[['surname', 'Pclass', 'person']].apply(perishing_mother_wife, axis=1)

#### Survivng Males
surviving_male_surnames = list(set(full_data[(full_data.male_adult == 1.0) &
                                             (full_data.Survived == 1.0) &
                                             ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname'].values))#活着的有家人的成年男子


def surviving_father_husband(passenger):#函数处理很有意思
    surname, Pclass, person = passenger
    return 1.0 if (surname in surviving_male_surnames) else 0.0


full_data['surviving_father_husband'] = full_data[['surname', 'Pclass', 'person']].apply(surviving_father_husband,
                                                                                         axis=1)

classers = ['Fare', 'Parch', 'Pclass', 'SibSp', 'TitleCat', 'CabinCat', 'Sex_female', 'Sex_male', 'EmbarkedCat',
            'FamilySize', 'NameLength', 'FamilyId']#回归所用特征
age_et = ExtraTreesRegressor(n_estimators=200)#额外回归树
X_train = full_data.loc[full_data.Age.notnull(), classers]
Y_train = full_data.loc[full_data.Age.notnull(), ['Age']]
X_test = full_data.loc[full_data.Age.isnull(), classers]
age_et.fit(X_train, np.ravel(Y_train))
age_preds = age_et.predict(X_test)
full_data.loc[full_data.Age.isnull(), ['Age']] = age_preds#加入预测后年龄

######################################################################牛逼分界线
######################################################################


print('Building Model...')

#### Model Build - Random Forest (Categorical Features)
model_dummys = ['Age', 'male_adult', 'female_adult', 'child', 'perishing_mother_wife', 'surviving_father_husband',
                'Fare', 'Parch', 'Pclass', 'SibSp', 'TitleCat', 'CabinCat', 'Sex_female', 'Sex_male', 'EmbarkedCat',
                'FamilySize', 'NameLength', 'FamilyId']
model_rf = RandomForestClassifier(n_estimators=600,min_samples_leaf=5,class_weight={0:0.745,1:0.255})
X_data = full_data.iloc[:891, :]
X_train = X_data.loc[:, model_dummys]#选择训练的特征
Y_data = full_data.iloc[:891, :]
Y_train = Y_data.loc[:, ['Survived']]
X_t_data = full_data.iloc[891:, :]
X_test = X_t_data.loc[:, model_dummys]
model_rf.fit(X_train, np.ravel(Y_train))

print('Generating Predictions...')

model_results = model_rf.predict(X_test)

print('Processing Submission File...')

model_results = [str(int(x)) for x in model_results]
submission = pd.DataFrame()
submission['PassengerId'] = X_t_data.PassengerId#测试集id
submission['Survived'] = model_results
submission.set_index(['PassengerId'], inplace=True, drop=True)
submission.head(3)
submission.to_csv('titanictest3.csv')#生成结果

print('Done.')