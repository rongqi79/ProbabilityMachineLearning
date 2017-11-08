import pandas as pd;
from sklearn import preprocessing;
import scipy.stats as sp;
from sklearn import decomposition;


def dataPreprocessing(df):

    #imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0, verbose=1)
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

def most_common(lst):
    return max(set(lst), key=lst.count)