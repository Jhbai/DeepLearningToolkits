from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DC
from sklearn.svm import SVC
class ModelSetUp:
    def __init__(self, string, X, y):
        Models = {'RandomForest':RF(n_estimators=1000),
                  'LogisticRegression':LR(),
                   'DecisionTree':DC(),
                   'GradientBoostingDecisionTree':GBDT(n_estimators=1000),
                   'SupportVectorMachine':SVC(C=1.0, kernel='rbf')
                   }
        self.model = Models[string]
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)