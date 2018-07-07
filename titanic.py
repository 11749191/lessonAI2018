import matplotlib
import sklearn
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


train = pd.read_csv('./train.csv', index_col = 'PassengerId') #返回一个pd.DataFrame
test = pd.read_csv('./test.csv', index_col = 'PassengerId')
submit = pd.read_csv('./gender_submission.csv', index_col = 'PassengerId')

from sklearn.model_selection import train_test_split
X_ = train.drop('Survived', axis =1)
Y_ = train['Survived']

X_train, X_vali, Y_train, Y_vali = \
train_test_split(X_, Y_, test_size = 0.2, random_state = 31)
# # X_train.head()

# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_index, vali_index = list(split.split(X_, X_.Sex))[0]
# X_train = X_.reindex(train_index)
# X_vali = X_.reindex(vali_index)
# Y_train = Y_.reindex(train_index)
# Y_vali = Y_.reindex(vali_index)
# print('rate of male/femate %f' % (X_.Sex.value_counts()['male']/X_.Sex.value_counts()['female']))
# print('rate of male/femate %f' % (X_train.Sex.value_counts()['male']/X_train.Sex.value_counts()['female']))

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline,FeatureUnion,Parallel
# print(X_.isna().sum())
from sklearn.base import BaseEstimator, TransformerMixin
class Deal_NAs(BaseEstimator, TransformerMixin):
    def __init__(self, drop_Cabin = True, strategy = 'most_frequent'):
        self.drop_Cabin = drop_Cabin
        self.fillna_with = 'NA'
        self.strategy = strategy
        self.imputer = Imputer(strategy = self.strategy)
    def fit(self, X_):
        numeric_cols = X_.columns[(X_.dtypes != 'object').values]
        numeric_ = X_[numeric_cols]
        self.imputer.fit(numeric_)
        return self
    def transform(self, X_):
        numeric_cols = X_.columns[(X_.dtypes != 'object').values]
        numeric_ = X_[numeric_cols]
        trans_numeric = self.imputer.transform(numeric_)
        X_[numeric_cols] = trans_numeric
        if self.drop_Cabin:
            X_ = X_.drop('Cabin',axis = 1)
        X_ = X_.fillna(self.fillna_with)
        return X_
dn = Deal_NAs()
X_withoutNA = dn.fit_transform(X_train)
# print(X_withoutNA.isna().sum())

from sklearn.preprocessing import LabelBinarizer,LabelEncoder
class RobustLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = []
        self.set_classes_ = None
        self.unseen_tag = 'unseen'
    def fit(self, list_):
        self.set_classes_ = set(list_)
        self.classes_ = list(self.set_classes_) + [self.unseen_tag]
        return self
    def transform(self, list_):
        list_ = [obj if obj in self.set_classes_ else self.unseen_tag for obj in list_]
        dct = dict(zip(self.classes_, range(len(self.classes_))))
        res = [dct[obj] for obj in list_]
        return res
class Encode_CatCols(BaseEstimator, TransformerMixin):
    def __init__(self, onehot = True, drop = []):
        self.onehot = onehot
        self.drop = drop
        self.encoders = {}
        self.catCols_ = None
        if self.onehot:
            self.oh_encoders = {}
    def fit(self, X_):
        catCols = [colname for colname in X_.columns if X_[colname].dtype == 'object']
        
        for col in self.drop:
            catCols.remove(col)
            
        self.catCols = catCols
        
        for col in catCols:
            encoder = RobustLabelEncoder()
            tmp = encoder.fit_transform(X_[col].tolist())
            self.encoders[col] = encoder
            if self.onehot:
                oh_encoder = LabelBinarizer() #不训练
                oh_encoder.classes_ = np.array(range(len(encoder.classes_)))
                #print(encoder.classes_)
                self.oh_encoders[col] = oh_encoder
        return self
    
    def transform(self, X_):
        
        for col in self.drop:
            X_ = X_.drop(col, axis = 1)
            
        if self.onehot:
            new_cols = [X_]
            
        for col in self.catCols:
            encoder = self.encoders[col]
            X_[col] = encoder.transform(X_[col].tolist())
            if self.onehot:
                binary_colnames = [col+'_'+class_ for class_ in encoder.classes_]
                #if len(binary_colnames) == 2: binary_colnames = [binary_colnames[0]]
                values = self.oh_encoders[col].transform(X_[col].tolist())
                new_cols.append(pd.DataFrame(values, index = X_.index, columns = binary_colnames))
                
        if self.onehot:
            
            new_cols = pd.concat(new_cols, axis = 1)
            X_ = new_cols.drop(self.catCols, axis = 1)
        return X_
ec = Encode_CatCols(onehot = True, drop = ['Name','Ticket'])
# print(ec.fit_transform(X_withoutNA).head())

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('deal_na', Deal_NAs()),('encode_cat', Encode_CatCols(onehot = False, drop = ['Name','Ticket']))])
X_prepared = pipeline.fit_transform(X_)
print(X_prepared.head(10))

#X_prepared = pipeline.fit_transform(X_)
X_train_p = pipeline.fit_transform(X_train)
X_vali_p = pipeline.transform(X_vali)

from sklearn.preprocessing import StandardScaler
class Scale_NumCols(BaseEstimator, TransformerMixin):
    def __init__(self, colnames = None, take_log = False):
        self.cols_to_transform = colnames
        self.take_log = take_log
        self.scaler = StandardScaler()
    def fit(self, X_):
        X_ = X_.copy()
        if self.cols_to_transform is None:
            self.cols_to_transform = [col for col in X_.columns if X_[col].dtype != 'object']
        if type(self.take_log) == bool:
            self.take_log = [self.take_log for col in self.cols_to_transform]
        else:
            assert len(self.take_log)==len(self.cols_to_transform)
        for col, log in zip(self.cols_to_transform, self.take_log):
            if log:
                X_.loc[:,col] = np.log(X_[col]+1)
        self.scaler.fit(X_[self.cols_to_transform])
        return self
    def transform(self, X_):
        for col, log in zip(self.cols_to_transform, self.take_log):
            if log:
                X_.loc[:,col] = np.log(X_[col]+1)
        X_.loc[:,self.cols_to_transform] = self.scaler.transform(X_[self.cols_to_transform])
        return X_
scale_num = ('scale_num', Scale_NumCols(['Age', 'SibSp', 'Parch', 'Fare'], take_log = True))
pipeline = Pipeline([('deal_na', Deal_NAs()),('encode_cat', Encode_CatCols(drop = ['Name','Ticket'])),scale_num])
#X_prepared = pipeline.fit_transform(X_)
X_train_p = pipeline.fit_transform(X_train)
X_vali_p = pipeline.transform(X_vali)

from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.ensemble import GradientBoostingClassifier as gbc
model = lr(C = 1)
model = dtc(min_samples_split = 10, max_features = 5)
model = abc(dtc(max_depth = 4), n_estimators=100)
model = gbc(n_estimators= 200)
#model = rfc(n_estimators=200 ,min_samples_split = 5)
model.fit(X_train_p, Y_train)
# print(model.score(X_train_p, Y_train))
# print(model.score(X_vali_p, Y_vali))
# coef_df = pd.DataFrame({'name':X_train_p.columns.tolist(), 'coef':model.coef_[0]})
# coef_df.sort_values('coef', ascending = False)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
Y_pred = model.predict(X_vali_p)
print(classification_report(Y_vali, Y_pred))

print(submit.head())

test_p = pipeline.transform(test)
Y_test_pred = model.predict(test_p)

test['Survived'] = Y_test_pred
result = test[['Survived']]
result.to_csv('result.csv', header = True, index = True)

cfm = confusion_matrix(Y_vali, Y_pred, labels = [0,1])
dff = pd.DataFrame(cfm, columns = ['predict_die','predict_survive'], index = ['true_die','true_survive'])

True_Negative = dff.iloc[0,0]
True_Positive = dff.iloc[1,1]
False_Negative = dff.iloc[1,0]
False_Positive = dff.iloc[0,1]

RECALL = True_Positive/(True_Positive + False_Negative)
print("recall:",RECALL)
PRECISION = True_Positive/(True_Positive + False_Positive)
print(PRECISION)
F1 = 2/(1./RECALL+1./PRECISION)
print(F1)