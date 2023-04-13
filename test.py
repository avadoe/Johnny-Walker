import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.iloc[:, :-1].values
X_test = test_data.iloc[:, :-1].values

y_train = train_data.iloc[:, -1].values
y_test = test_data.iloc[:, -1].values

class ClassificationModels:
    def __init__(self, SGD_Classifier=SGDClassifier(), Decision_Tree_Classifier=DecisionTreeClassifier(),
                 Naive_Bayes_Classifier=GaussianNB(), KNN_Classifier=KNeighborsClassifier(), SVM_Classifier=SVC()):
        self.scores = None
        self.predictions = None
        self.SGD_Classifier = SGD_Classifier
        self.Decision_Tree_Classifier = Decision_Tree_Classifier
        self.Naive_Bayes_Classifier = Naive_Bayes_Classifier
        self.KNN_Classifier = KNN_Classifier
        self.SVM_Classifier = SVM_Classifier

    def fit(self, X, y):
        self.SGD_Classifier.fit(X, y)
        self.Decision_Tree_Classifier.fit(X, y)
        self.Naive_Bayes_Classifier.fit(X, y)
        self.KNN_Classifier.fit(X, y)
        self.SVM_Classifier.fit(X, y)

    def predictScores(self, X, y, scorer=accuracy_score):
        self.predictions = {
            'SGD_Classifier': self.SGD_Classifier.predict(X),
            'Decision_Tree_Classifier': self.Decision_Tree_Classifier.predict(X),
            'Naive_Bayes_Classifier': self.Naive_Bayes_Classifier.predict(X),
            'KNN_Classifier': self.KNN_Classifier.predict(X),
            'SVM_Classifier': self.SVM_Classifier.predict(X)
        }

        self.scores = {
            'SGD_Classifier': scorer(y, self.predictions['SGD_Classifier']),
            'Decision_Tree_Classifier': scorer(y, self.predictions['Decision_Tree_Classifier']),
            'Naive_Bayes_Classifier': scorer(y, self.predictions['Naive_Bayes_Classifier']),
            'KNN_Classifier': scorer(y, self.predictions['KNN_Classifier']),
            'SVM_Classifier': scorer(y, self.predictions['KNN_Classifier'])
        }
        
    def confusionMatrix(self, test_y, predicted_y):
        return pd.crosstab(test_y, predicted_y, rownames=['Actual'], colnames=['Predicted'], margins=True)
        
    def kFoldSplit(self, X, k=5, random_state=None):
        X_index = np.arange(len(X))
        X_groups = np.array_split(X_index, k)
        train_index = []
        test_index = []

        if random_state is not None:
            np.random.seed(random_state)

        for test in X_groups:
            train = X_index[~np.isin(X_index, test)]
            np.random.shuffle(train)
            np.random.shuffle(test)

            train_index.append(train)
            test_index.append(test)

        return train_index, test_index

    def cross_validate_score(self, model, X, y, cv=2, scorer=accuracy_score, random_state=None, return_cfmat=False):
        if cv < 2:
            print(f"ERROR: CV should be > 1, otherwise use score method indirectly")
            return

        scores = []
        confusion_matrices = []

        train_index, test_index = self.kFoldSplit(X, cv, random_state)
        for train_index, test_index in zip(train_index, test_index):
            x_train = X[train_index]
            Y_train = y[train_index]
            x_test = X[test_index]
            Y_test = y[test_index]

            model.fit(x_train, Y_train)

            Y_prediction = model.predict(x_test)
            if return_cfmat is True:
                confusion_matrices.append(pd.crosstab(Y_test, Y_prediction, margins=True))
            scores.append(scorer(Y_test, Y_prediction))

        if return_cfmat is True:
            return np.mean(scores), confusion_matrices
        return np.mean(scores)

    def cross_validate(self, X, y, cv=5, scorer=accuracy_score, return_confusion_matrix=True):
        SGD_scores, SGD_cfm = self.cross_validate_score(SGDClassifier, X, y, cv=cv, scorer=scorer,
                                              return_cfmat=return_confusion_matrix)
        Decision_Tree_scores, Decision_Tree_cfm = self.cross_validate_score(DecisionTreeClassifier, X, y, cv=cv, scorer=scorer,
                                                                  return_cfmat=return_confusion_matrix)
        Naive_Bayes_scores, Naive_Bayes_cfm = self.cross_validate_score(GaussianNB, X, y, cv=cv, scorer=scorer,
                                                              return_cfmat=return_confusion_matrix)
        KNN_scores, KNN_cfm = self.cross_validate_score(KNeighborsClassifier, X, y, cv=cv, scorer=scorer,
                                              return_cfmat=return_confusion_matrix)
        SVM_scores, SVM_cfm = self.cross_validate_score(SVC, X, y, cv=cv, scorer=scorer, return_cfmat=return_confusion_matrix)

        self.cross_validation_scores = {
            'SGD_Classifier': SGD_scores,
            'Decision_Tree_Classifier': Decision_Tree_scores,
            'Naive_Bayes_Classifier': Naive_Bayes_scores,
            'KNN_Classifier': KNN_scores,
            'SVM_Classifier': SVM_scores
        }

        self.cross_validation_cfmat = {
            'SGD_Classifier': SGD_cfm,
            'Decision_Tree_Classifier': Decision_Tree_cfm,
            'Naive_Bayes_Classifier': Naive_Bayes_cfm,
            'KNN_Classifier': KNN_cfm,
            'SVM_Classifier': SVM_cfm
        }

        if return_confusion_matrix is True:
            return self.cross_validation_scores, self.cross_validation_cfmat
        return self.cross_validation_scores
    
    def fit_predict(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        self.predictScores(X_test, y_test)
    
    def print_scores(self):
        for model, score in self.scores.items():
            print(f"{model} Accuracy: {score}")
            
    def print_confusion_matrices(self):
        for model, cfmat_list in self.confusion_matrices.items():
            for i, cfmat in enumerate(cfmat_list):
                print(f"{model} Confusion Matrix {i+1}:\n{cfmat}\n\n")
                
    def print_cross_validation_scores(self):
        for model, score in self.cross_validation_scores.items():
            print(f"{model} Cross Validation Mean Accuracy: {score}\n")



models = ClassificationModels()
models.fit_predict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print(f"Accuracy Scores: {models.print_scores()}")

print(f"Confusion Matrices: ")
models.print_confusion_matrices()

print(f"Cross Validation Scores: {models.print_cross_validation_scores()}")
