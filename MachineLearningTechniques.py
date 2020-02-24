from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

class MachineLearningTechniques:

    # Initialize
    def __init__(self, xtr, ytr, xt, yt, xval, yval):      # initialize for sets with y test data and validation data
        self.xtr  = xtr
        self.ytr  = ytr
        self.xt   = xt
        self.yt   = yt
        self.xval = xval
        self.yval = yval


    def LASSO(self):
        params = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2]}
        lasso = Lasso(random_state=33, alpha=0.01, max_iter=10000000)
        ab = AdaBoostRegressor(base_estimator=lasso, n_estimators=50, learning_rate=1.0, random_state=33)
        gs = GridSearchCV(lasso, params, cv=5)
        abgs = GridSearchCV(ab, {'n_estimators':[10, 50, 100], 'learning_rate':[0.5, 1.0]}, cv=5)        
        return(lasso, ab, gs, abgs)

    def RIDGE(self):
        params = {'alpha': [0.25, 0.5, 0.75, 1.0], 'solver': ['auto', 'sag']}
        ridge = Ridge(random_state=33)
        ab = AdaBoostRegressor(base_estimator=ridge, n_estimators=50, learning_rate=1.0, random_state=33)
        gs = GridSearchCV(ridge, params, cv=5)
        abgs = GridSearchCV(ab, {'n_estimators':[10, 50, 100], 'learning_rate':[0.5, 1.0]}, cv=5)
        return(ridge, ab, gs, abgs)

    def FOREST(self):
        params = {'n_estimators': [10, 50, 10]}
        rf = RandomForestRegressor(bootstrap=True, max_features='auto', n_estimators=10, random_state=33)
        ab = AdaBoostRegressor(base_estimator=rf, n_estimators=50, learning_rate=1.0, random_state=33)
        gs = GridSearchCV(rf, params, cv=5)
        abgs = GridSearchCV(ab, {'n_estimators':[10, 50, 100], 'learning_rate':[0.5, 1.0]}, cv=5)
        return(rf, ab, gs, abgs)

    def DECISIONTREE(self):
        params = {'splitter': ['best']}
        dt = DecisionTreeRegressor()
        ab = AdaBoostRegressor(base_estimator=dt, n_estimators=50, learning_rate=1.0, random_state=33)
        gs = GridSearchCV(dt, params, cv=5)
        abgs = GridSearchCV(ab, {'n_estimators':[10, 50, 100], 'learning_rate':[0.5, 1.0]}, cv=5)
        return(dt, ab, gs, abgs)

    def DEEPNN(self):
        nn = Sequential()
        nn.add(Dense(15, input_dim=len(self.xtr[0]), activation='relu'))
        nn.add(Dense(8, activation='relu'))
        nn.add(Dense(1, activation='relu'))
        nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return(nn)
