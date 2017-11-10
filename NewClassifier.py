import pandas as pd;
import numpy as np;
from sklearn import decomposition, svm,naive_bayes
from sklearn.model_selection import GridSearchCV
import sklearn.ensemble as ske
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from time import time
from operator import itemgetter
import pickle
import os
import util



class NewClassifier():

    def __init__(self, data_name):
       self.data_name=data_name



    def fit(self, X, Y):
        self.feature_number=X.shape[1]



        def fileExist(data_name, feature_number, algo):
            filename = data_name + algo + str(feature_number)
            cwd = os.getcwd()
            filename_suffix = 'model'
            pathfile = os.path.join(cwd, filename + "." + filename_suffix)
            return os.path.isfile(pathfile)

        # naive bayers
        clf_nb = naive_bayes.GaussianNB()
        nb_result = cross_val_score(clf_nb, X, Y, cv=5).mean()
        predict_nb = cross_val_predict(clf_nb, X, Y, cv=5)
        print("naive bayes result is : ", nb_result)

        clf_nb.fit(X, Y)
        if (fileExist(self.data_name, self.feature_number, "Gaussi") == False):
            extra_name = self.data_name
            filename = extra_name + str(clf_nb)[:6] + str(self.feature_number)
            cwd = os.getcwd()
            filename_suffix = 'model'
            pathfile = os.path.join(cwd, filename + "." + filename_suffix)
            with open(pathfile, "wb") as fp:
                pickle.dump(clf_nb, fp)


        def run_gridsearch(X, y, clf, param_grid,cv=5 ):
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
            grid_search = GridSearchCV(clf,param_grid=param_grid,cv=cv)
            start = time()
            grid_search.fit(X, y)
            extra_name=self.data_name
            filename = extra_name + str(clf)[:6]+str(self.feature_number)
            cwd = os.getcwd()
            filename_suffix = 'model'
            pathfile = os.path.join(cwd, filename + "." + filename_suffix)
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
                   "{0:.3f} (std: {1:.3f})").format(score.mean_validation_score,np.std(score.cv_validation_scores)))
                print("Parameters: {0}".format(score.parameters))
                print("")

            return top_scores[0].parameters

    #decision tree
        if (fileExist(self.data_name, self.feature_number, "Decisi")==False):
            print("-- Decision Tree Grid Parameter Search via 10-fold CV")
            # set of parameters to test
            param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_split": [2, 10, 20],
                  "max_depth": [None, 2, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "max_leaf_nodes": [None, 5, 10, 20],
                  }

            dt = DecisionTreeClassifier()
            ts_gs = run_gridsearch(X, Y, dt, param_grid, cv=10)
            print("\n-- Best Parameters of Decision Tree:")
            for k, v in ts_gs.items():
                print("parameter: {:<20s} setting: {}".format(k, v))

    # svm
        if (fileExist(self.data_name, self.feature_number, "SVC(C=") == False):
            print("-- svm Grid Parameter Search via 10-fold CV")
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

            svmGrid = run_gridsearch(X, Y, svm.SVC(C=1), tuned_parameters, cv=10)

            print("\n-- svm Best Parameters:")
            for k, v in svmGrid.items():
                print("parameter: {:<20s} setting: {}".format(k, v))

    #logistic regression
        if (fileExist(self.data_name, self.feature_number, "Logist")==False):
            print("-- Logistic Regression Grid Parameter Search via 10-fold CV")
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            clf_lr = LogisticRegression()
            lrGrid = run_gridsearch(X, Y, clf_lr, param_grid, cv=10)
            print("\n-- Logistic Regression Best Parameters:")
            for k, v in lrGrid.items():
                print("parameter: {:<20s} setting: {}".format(k, v))
    #Random Forrest

        if (fileExist(self.data_name, self.feature_number, "Random") == False):
            print("-- random forrest Grid Parameter Search via 10-fold CV")
            param_grid_rf = {
            'n_estimators': [10, 50, 200],
            'max_features': ['auto', 'sqrt', 'log2']
            }

            clf_rf = ske.RandomForestClassifier()
            rfGrid = run_gridsearch(X, Y, clf_rf, param_grid_rf, cv=10)
            print("\n-- Random Forrest Best Parameters:")
            for k, v in rfGrid.items():
                print("parameter: {:<20s} setting: {}".format(k, v))



    def sortedResult(self, X, Y):
        # Load model from file
        filename = self.data_name + "Decisi" + str(self.feature_number)
        cwd = os.getcwd()
        filename_suffix = 'model'
        pathfile = os.path.join(cwd, filename + "." + filename_suffix)
        with open(pathfile, "rb") as fp:
            grid_search_load = pickle.load(fp)

        # Predict new data with model loaded from disk
        dt_result = grid_search_load.best_estimator_.score(X, Y)
        predict_dt = grid_search_load.best_estimator_.predict(X)
        print(dt_result)
        accuracydt = accuracy_score(Y, predict_dt)
        print("decision tree accuracy is : ", accuracydt)

        # Predict naive bayes with model loaded from disk
        # Load model from file
        filename = self.data_name + "Gaussi" + str(self.feature_number)
        cwd = os.getcwd()
        filename_suffix = 'model'
        pathfile = os.path.join(cwd, filename + "." + filename_suffix)
        with open(pathfile, "rb") as fp:
            grid_search_load = pickle.load(fp)
        nb_result = grid_search_load.score(X, Y)
        predict_nb = grid_search_load.predict(X)
        print(nb_result)
        accuracynb = accuracy_score(Y, predict_nb)
        print("naive bayes accuracy is : ", accuracynb)



        # Load model from random Forrest
        filename = self.data_name + "Random" + str(self.feature_number)
        cwd = os.getcwd()
        filename_suffix = 'model'
        pathfile = os.path.join(cwd, filename + "." + filename_suffix)
        with open(pathfile, "rb") as fp:
            grid_search_load = pickle.load(fp)

        # Predict new data with random forrest model loaded from disk
        rf_result = grid_search_load.best_estimator_.score(X, Y)
        predict_rf = grid_search_load.best_estimator_.predict(X)
        print(rf_result)
        accuracyrf = accuracy_score(Y, predict_rf)
        print("random forrest accuracy is : ", accuracyrf)

        # Load model from logistic regression
        filename = self.data_name + "Logist" + str(self.feature_number)
        cwd = os.getcwd()
        filename_suffix = 'model'
        pathfile = os.path.join(cwd, filename + "." + filename_suffix)
        with open(pathfile, "rb") as fp:
            grid_search_load = pickle.load(fp)

        # Predict new data with logistic regression model loaded from disk
        lr_result = grid_search_load.best_estimator_.score(X, Y)
        predict_lr = grid_search_load.best_estimator_.predict(X)
        print(lr_result)
        accuracylr = accuracy_score(Y, predict_lr)
        print("logistic regression accuracy is : ", accuracylr)

        # Load model from svm
        filename = self.data_name + "SVC(C="+str(self.feature_number)
        cwd = os.getcwd()
        filename_suffix = 'model'
        pathfile = os.path.join(cwd, filename + "." + filename_suffix)
        with open(pathfile, "rb") as fp:
            grid_search_load = pickle.load(fp)

        # Predict new data with svm model loaded from disk
        svm_result = grid_search_load.best_estimator_.score(X, Y)
        predict_svm = grid_search_load.best_estimator_.predict(X)
        print(svm_result)
        accuracysvm = accuracy_score(Y, predict_svm)
        print("svm accuracy is : ", accuracysvm)



        result = {'score': [svm_result, rf_result, lr_result, dt_result, nb_result],
                  'predict': [predict_svm, predict_rf, predict_lr, predict_dt, predict_nb]}
        self.dfResult = pd.DataFrame(result,
                                index=['svm', 'random forrest', 'logistic regression', 'decision tree', 'naive bayes'])
        # print(dfResult)
        self.dfResult.sort_values(by=['score'], inplace=True, ascending=0)
        print(self.dfResult)
        topPrediction = self.dfResult.iloc[0, 1]
        print("the top accuracy of machine learning algorithms is : ", self.dfResult.first_valid_index() ,topPrediction)


    '''def predict(self, X, Y, k):
        s2 = []
        for i in range(len(Y)):
            temp = 0
            for j in range(k):
                temp = temp + self.predicitonTopK[j][i]
            if temp > k / 2:
                s2.append(1)
            else:
                s2.append(0)
        myarray = np.asarray(s2)
        myAccuracy = accuracy_score(Y, myarray)
        print("my accuracy is :", myAccuracy)'''

    def score(self, Y):
        myScore = accuracy_score(Y, self.prediction )
        print("my accuracy is :", myScore)


    def multiPredict(self, X, Y, k):
        s2 = []
        predicitonTopK = self.dfResult.iloc[0:k + 1, 0]
        for i in range(len(Y)):
            temp=[]
            for j in range(k):
                temp.append(predicitonTopK[j][i])
            s2.append(util.most_common(temp))
        self.prediction = np.asarray(s2)





