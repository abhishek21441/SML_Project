import numpy as np
import pandas as pd
import sklearn.decomposition as sk_decomp
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import Lasso

class kmeans_features(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        
        kmeans = KMeans(n_clusters=7, n_init='auto')
        kmeans.fit(X)

        new_X = pd.DataFrame(X)
        new_X['cluster_labels'] = kmeans.labels_

        new_X.columns = new_X.columns.astype(str)

        return new_X


def accuracy(out, true):
    return np.sum(out == true)/len(out)

data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_data = test_data.drop(columns=['ID'])

ids = data['ID']

labels = data['category']
# labels.replace(np.unique(labels), [i for i in range(len(np.unique(labels)))], inplace=True)
data = data.drop(columns=['ID', 'category'])


#15 --> 79.472, 20 --> 80.13, 25 --> 80.04, 30 --> 80.37
lof = LocalOutlierFactor(n_neighbors=4)
outlier = lof.fit_predict(data)

data_cleaned = data[outlier == 1]
labels_cleaned = labels[outlier == 1]

# print(data_cleaned.shape)

# kmeans = KMeans(n_clusters=len(np.unique(labels_cleaned)))
# kmeans.fit(data_cleaned)

# data_cleaned['cluster_labels'] = kmeans.labels_

#pca, lda followed by logistic regression gives 80%
# 0. LDA Being used in all pipes
km = kmeans_features()

lda = LinearDiscriminantAnalysis(tol=1e-5) #80%

#1. Logistic regression
cl = LogisticRegression(max_iter=10000, tol=1e-5) #can toggle class_weight parameter to be default or 'balanced', in my experience default usually does better or the same as 'balanced'
# nb = ComplementNB() #77

#2. Random Forest
rf = RandomForestClassifier(n_estimators=205, random_state=6)

#3. Lasso Regression
# lasso = Lasso(alpha=0.1, random_state=42)

pipe = make_pipeline(sk_decomp.PCA(n_components=370, whiten=True), MinMaxScaler(), lda, km,  MinMaxScaler(), cl, verbose=False) 
pipe2 = make_pipeline(sk_decomp.PCA(n_components=180, whiten=True), MinMaxScaler(), lda, rf, verbose=False)

#pca1 370 , pca2 200 LR 10k , e-5  , rf 200 , 0   ,lda e-5 ====> gave 84.057 on submission
#pca1 375 , pca2 200 LR 10k , e-5  , rf 200 , 0   ,lda e-5 ====> gave 83.091 on submission but very high cross validation
#pca1 380 , pca2 200 LR 10k , e-5  , rf 200 , 0   ,lda e-5 ====> gave 83.091 on submission
#pca1 370 , pca2 200 LR 10k , e-5  , rf 175 , 0   ,lda e-5 ====> gave 83.091 on submission
#pca1 370 , pca2 200 LR 10k , e-5  , rf 185 , 0   ,lda e-5 ====> scored similar to above on CV


# scores = cross_val_score(pipe, data_cleaned, labels_cleaned, cv=5)

# print(scores, np.mean(scores))

pipe.fit(data_cleaned, labels_cleaned)
pipe2.fit(data_cleaned, labels_cleaned)

voting_clf = VotingClassifier(
    estimators=[('cl', pipe),('rf', pipe2)],
    voting='soft'
)
voting_clf.fit(data_cleaned, labels_cleaned)

# scores3 = cross_val_score(pipe, data_cleaned, labels_cleaned, cv=3)
# # # scores5 = cross_val_score(voting_clf, data_cleaned, labels_cleaned, cv=5,verbose=True)

# print(scores3, np.mean(scores3))
# # # print(scores5, np.mean(scores5))


testingplease = pd.DataFrame([i for i in range(test_data.shape[0])], columns=['ID'])
testingplease['Category'] = voting_clf.predict(test_data)

testingplease.to_csv('something_newrf180_4.csv', index=False)

