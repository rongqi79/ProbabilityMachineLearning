import pandas as pd;
import numpy as np;
from sklearn import preprocessing;
from sklearn.datasets import load_iris;
import scipy.stats as sp
from sklearn import decomposition, svm,naive_bayes
from sklearn.model_selection import train_test_split,GridSearchCV
import Classifier
import sklearn.ensemble as ske
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from time import time
from operator import itemgetter
import pickle
import os

def dataPreprocessing(df):

    #imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    #imp.fit(df)
    df = df.apply(preprocessing.LabelEncoder().fit_transform)
    #df=df.apply(preprocessing.StandardScaler().fit_transform)
    #df=df.apply(preprocessing.MinMaxScaler().fit_transform)
    return df

def featureSelection(df, k):
    # feature selection based on chi square
    listPVal = []
    for i in range(len(df.columns) - 1):
        freqtab = pd.crosstab(df.iloc[:, i], df.iloc[:, -1])
        chi2, pval, dof, expected = sp.chi2_contingency(freqtab)
        listPVal.append(pval)
    for i in range(len(df.columns) - 1):
        if (listPVal[i] > 0.005):
            df.drop(df.columns[i], axis=1, inplace=True)
    print("now the column number is ", df.shape[1])
    data = df.iloc[:,:len(df.columns)-1]
    Y=df.iloc[:,len(df.columns)-1]
    Xdata=pd.DataFrame(data)
    pca = decomposition.PCA(n_components=k)
    pca.fit(data)
    Xdata = pd.DataFrame(pca.transform(data))
    Xdata[k]=Y
    return Xdata


dataSet = input("What's your choice of dataset?\n"
                "1: bank;"
                "2: titanic;"
                "3: iris;"
                )

if dataSet=='1':
    df = pd.read_csv('./bank/bank.csv', delimiter=';')
    extra_name='bank'
elif dataSet=='2':
    df = pd.read_csv('titanic_data.csv')
    extra_name = 'titanic'

elif dataSet=='3':
    iris = load_iris()
    df=pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                 columns=iris['feature_names'] + ['target'])
    extra_name = 'iris'
else:
    print("dataset could not be found")

print (df.head())
df=dataPreprocessing(df)

print (df.head())

k= int(input("choose the number of features you want to keep(integer >1)"))

df=featureSelection(df,k)
print(df.head())

Y = df[k].values.astype(int)
X=df.drop(k, 1).values

#data_train, data_test, target_train, target_test = train_test_split(X, Y,test_size=0.2, random_state=43)
data_train, data_test, target_train, target_test = train_test_split(X, Y,test_size=0.2)
'''
# random forrest
cv_num_estimators=0
while True:
    num_estimators=int(input("Let's start with random forrest with cross validation.\n"
                             "Choose number of estimators you want to use for random forrest:"))
    clf_rf = ske.RandomForestClassifier(n_estimators=num_estimators)
    scoresRF = cross_val_score(clf_rf, data_train, target_train, cv=5)
    print(scoresRF)
    crossValResult= int(input("do you think this cross validation result is good?\n"
             "input 1 if yes; input 2 if no"))
    if (crossValResult==1):
        cv_num_estimators=num_estimators
        break

clf_rf = ske.RandomForestClassifier(n_estimators=cv_num_estimators)
clf_rf.fit(data_train,target_train)
predictrf=clf_rf.predict(data_test)
accuracyrf = accuracy_score(target_test, predictrf)
#print(predictrf)
print(accuracyrf)


#Logistic Regression
while True:
    num_estimators=input("Let's try logistic regression with cross validation.\n")
    clf_lr = LogisticRegression()
    scoresRF = cross_val_score(clf_lr, data_train, target_train, cv=5)
    print(scoresRF)
    crossValResult= int(input("do you think this cross validation result is good?\n"
             "input 1 if yes; input 2 if no"))
    if (crossValResult==1):
        cv_num_estimators=num_estimators
        break
clf_lr = LogisticRegression()
clf_lr.fit(data_train,target_train)
predictlr=clf_lr.predict(data_test)
accuracylr = accuracy_score(target_test, predictlr)
#print(predictlr)
print("logistic regression accuracy of test set is ",accuracylr)

'''

'''
for i in range(len(target_test)):
    if ((predictlr[i]!=target_test[i]) | (predictrf[i]!=target_test[i])):
        print("index ", i, "   result should be: " ,target_test[i] , "   RF predict: ",
              predictrf[i] , "   LR predict ",predictlr[i] )


#svm
svmParameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, svmParameters)
clf.fit(data_train,target_train) 
'''

#naive bayers
clf_nb= naive_bayes.GaussianNB()
nb_result = cross_val_score(clf_nb, data_train, target_train, cv=5).mean()
predict_nb= cross_val_predict(clf_nb, data_train, target_train, cv=5)
print("naive bayes result is : ", nb_result)




def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)
    filename=extra_name+str(clf)[:6]
    cwd = os.getcwd()
    filename_suffix='model'
    pathfile=os.path.join(cwd, filename + "." + filename_suffix)
    with open(pathfile, "wb") as fp:
        pickle.dump(grid_search, fp)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                                         len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return top_params

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters


dt_old = DecisionTreeClassifier(min_samples_split=20,
                                random_state=99)
dt_old.fit(data_train, target_train)
scores = cross_val_score(dt_old, data_train, target_train, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()), end="\n\n")


print("-- Decision Tree Grid Parameter Search via 10-fold CV")

# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

dt = DecisionTreeClassifier()
ts_gs = run_gridsearch(data_train, target_train, dt, param_grid, cv=10)

print("\n-- Best Parameters of Decision Tree:")
for k, v in ts_gs.items():
    print("parameter: {:<20s} setting: {}".format(k, v))


# Load model from file
filename=extra_name+'Decisi'
cwd = os.getcwd()
filename_suffix='model'
pathfile=os.path.join(cwd, filename + "." + filename_suffix)
with open(pathfile, "rb") as fp:
    grid_search_load = pickle.load(fp)

# Predict new data with model loaded from disk
dt_result = grid_search_load.best_estimator_.score(data_test, target_test)
predict_dt = grid_search_load.best_estimator_.predict(data_test)
print(dt_result)
accuracydt = accuracy_score(target_test, predict_dt)
print("decision tree accuracy is : ", accuracydt)

#svm
'''
print("-- svm Grid Parameter Search via 10-fold CV")
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svmGrid = run_gridsearch(data_train, target_train, svm.SVC(C=1), tuned_parameters, cv=10)

print("\n-- Best Parameters:")
for k, v in svmGrid.items():
    print("parameter: {:<20s} setting: {}".format(k, v))
'''

#filename =str(svm.SVC)[:6]
filename = 'SVC(C='
cwd = os.getcwd()
filename_suffix = 'model'
pathfile = os.path.join(cwd, filename + "." + filename_suffix)
with open(pathfile, "rb") as fp:
    grid_search_load = pickle.load(fp)

# Predict new data with model loaded from disk
svm_result = grid_search_load.best_estimator_.score(data_test, target_test)
predict_svm = grid_search_load.best_estimator_.predict(data_test)
print(svm_result)
accuracysvm = accuracy_score(target_test, predict_svm)
print("svm accuracy is : ", accuracysvm)


print("-- logistic regression Grid Parameter Search via 10-fold CV")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
clf_lr = LogisticRegression()
lrGrid = run_gridsearch(data_train, target_train, clf_lr, param_grid, cv=10)
print("\n-- Logistic Regression Best Parameters:")
for k, v in lrGrid.items():
    print("parameter: {:<20s} setting: {}".format(k, v))


filename = extra_name+'Logist'
cwd = os.getcwd()
filename_suffix = 'model'
pathfile = os.path.join(cwd, filename + "." + filename_suffix)
with open(pathfile, "rb") as fp:
    grid_search_load = pickle.load(fp)

# Predict new data with model loaded from disk
lr_result = grid_search_load.best_estimator_.score(data_test, target_test)
predict_lr = grid_search_load.best_estimator_.predict(data_test)
accuracyLr = accuracy_score(target_test, predict_lr)
print("logistic regression accuracy is : ", accuracyLr)

print("-- random forrest Grid Parameter Search via 10-fold CV")
param_grid_rf = {
    'n_estimators': [10, 50, 200],
    'max_features': ['auto', 'sqrt', 'log2']
}

clf_rf = ske.RandomForestClassifier()
rfGrid = run_gridsearch(data_train, target_train, clf_rf, param_grid_rf, cv=10)
print("\n-- Random Forrest Best Parameters:")
for k, v in lrGrid.items():
    print("parameter: {:<20s} setting: {}".format(k, v))

filename = extra_name+ str(clf_rf)[:6]
cwd =os.getcwd()
filename_suffix = 'model'
pathfile = os.path.join(cwd, filename + "." + filename_suffix)
with open(pathfile, "rb") as fp:
    grid_search_load = pickle.load(fp)

# Predict new data with model loaded from disk
rf_result = grid_search_load.best_estimator_.score(data_test, target_test)
predict_rf = grid_search_load.best_estimator_.predict(data_test)
accuracyrf = accuracy_score(target_test, predict_rf)
print("random forrest accuracy is : ",rf_result )

result={'score': [svm_result,rf_result,lr_result,dt_result,nb_result],
        'predict':[predict_svm,predict_rf,predict_lr,predict_dt, predict_nb]}
dfResult = pd.DataFrame(result,index = ['svm','random forrest','logistic regression','decision tree','naive bays' ])
print(dfResult)
dfResult.sort_values(by=['score'], inplace=True, ascending=0)
print(dfResult)
k=3
predicitonTopK =dfResult.iloc[0:k+1, 0]


s2=[]
for i in range(len(target_test)):
    temp=0
    for j in range(k):
        temp=temp+predicitonTopK[j][i]
    if temp>k/2:
        s2.append(1)
    else:
        s2.append(0)


'''s2=[]
for i in range(len(target_test)):
    temp= [0] * 3
    for j in range(k):
        if (predicitonTopK[j][i]==0):
            temp[0]=temp[0]+1;
        elif(predicitonTopK[j][i]==1):
            temp[1] = temp[1] + 1;
        else:
            temp[2] = temp[2] + 1;
    if (temp[0]>temp[1] & temp[0]>temp[2]):
        s2.append(0)
    elif (temp[1]>temp[0] & temp[1]>temp[2]):
        s2.append(1)
    elif (temp[2]>temp[0] & temp[2]>temp[1]):
        s2.append(2)
    else:
        s2.append(predicitonTopK[k][i])
'''


myarray = np.asarray(s2)
myAccuracy = accuracy_score(target_test, myarray)
topPrediction=dfResult.iloc[0, 1]
print("the top accuracy of machine learning algorithms is : ",topPrediction )

print("my accuracy is :",myAccuracy)
