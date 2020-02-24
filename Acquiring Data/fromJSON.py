import matplotlib.pyplot as plt
import json
import mainHardnessPred as pred
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import writeHardnessData as whd
from keras.models import Sequential
from keras.layers import Dense

np.set_printoptions(threshold=np.inf)

global ml; global spacegroup; global dataFile
global groups; global orders

groups = {'triclinic':[1,2], 'monoclinic':[3,15], 'orthorhombic':[16,74], 'tetragonal':[75,142], \
          'trigonal':[143,167], 'hexagonal':[168,194], 'cubic':[195,230], 'all':[0,230]}
orders = {'rAvg': 0, 'mAvg': 1, 'colAvg': 2, 'rowAvg': 3, 'atNumAvg': 4, 'sValFrac': 5,\
              'pValFrac': 6, 'dValFrac': 7, 'fValFrac': 8, 'sValAvg': 9, 'pValAvg': 10, \
              'dValAvg': 11, 'fValAvg': 12, 'elAffAvg': 13}

ml = 'FOREST'                                  # which machine learning technique to use?
spacegroup = 'all'                             # which spacegroup to use?
dataFile = 'TextFiles/ec.json'                 # 'TextFiles/materialsData.json' or 'TextFiles/ec.json',


def refineSpaceGroup(limits):
    File   = open(dataFile, 'r')
    data   = json.load(File)
    length = len(data); ind = 0
    xVals  = np.ndarray(shape=(length, 19), dtype=float)
    G_VRH  = np.zeros(length); K_VRH  = np.zeros(length)
    dataD = {}
    
    for elem in range(len(data)):
        curr = data[elem]
        eq = curr['formula']
        sp_group = curr['space_group']
        
        if limits[0] < sp_group and limits[1] > sp_group:
            dataD = whd.calcMatProps(eq, curr, dataD, False)
            G_VRH[ind] = data[elem]['G_VRH']
            K_VRH[ind] = data[elem]['K_VRH']

            for var in orders:
                xVals[ind][orders[var]] = dataD[var]
            ind += 1 
    
    inds = np.multiply(G_VRH, K_VRH)
    xVals = xVals[inds!=0]
    G_VRH = G_VRH[inds!=0]
    K_VRH = K_VRH[inds!=0]
    return(xVals, G_VRH, K_VRH)


def NEURALNETWORK(x_tr, y_tr, x_T):
    ml = MLPRegressor(solver='sgd', momentum=0.1, alpha=1e-6, hidden_layer_sizes=(10,2), random_state=33)
    ml.fit(x_tr, y_tr)
    pred = ml.predict(x_T)
    return(ml, pred)


def DEEPNN(x_tr, y_tr, x_T):
    nn = Sequential()
    nn.add(Dense(15, input_dim=len(x_tr[0]), activation='relu'))
    nn.add(Dense(8, activation='relu'))
    nn.add(Dense(1, activation='relu'))
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn.fit(x_tr, y_tr, epochs=50, batch_size=10)
    pred = nn.predict(x_T)
    return(nn, pred)


def getFunc(name, X_tr, Y_tr, X_T):
    if name=='LASSO' or name=='RIDGE':
        params = {'alpha' : [0.4, 0.8],\
                  'max_iter' : [100000000]} 
    if name=='FOREST' or name=='DECISIONTREE':
        params = {'base_estimator' : ['gini', 'entropy'],\
                  'base_estimator__splitter' : ['best', 'random'],\
                  'n_estimators' : [5, 15, 45, 135, 405]}
        
    if name == 'LASSO':
        return pred.LASSO(X_tr, Y_tr, X_T)
    elif name == 'RIDGE':
        return pred.RIDGE(X_tr, Y_tr, X_T)
    elif name == 'FOREST':
        return pred.FOREST(X_tr, Y_tr, X_T)
    elif name == 'TREE':
        return pred.DECISIONTREE(X_tr, Y_tr, X_T)
    elif name == 'NN':
        return NEURALNETWORK(X_tr, Y_tr, X_T)
    elif name == 'DEEP':
        return DEEPNN(X_tr, Y_tr, X_T)
    else:
        print('Enter a valid name: LASSO, RIDGE, FOREST, or TREE')


def getParams(name):
    if name == 'LASSO':
        return {'alpha': [0.001, 0.01, 0.1, 0.2]}
    elif name == 'RIDGE':
        return {'alpha': [0.25, 0.5, 0.75, 1.0], 'solver': ['auto', 'sag']}
    elif name == 'FOREST':
        return {'n_estimators': [10, 50, 100]}
    elif name == 'TREE':
        return {'splitter': ['best']}
        

def scaleData(train, test, validate):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    validate = scaler.transform(validate)
    return(train, test, validate)
    

def writePredictions(fname, names, method, predictions):
    wrFile = open("TextFiles/"+fname+"_values.txt", "w")
    wrFile.write("'Compound Name' '"+fname+" pred adaboost&gridsearch "+method+"'\n")
    for row in range(0, len(predictions)):
        wrFile.write(names[row]+" "+" "+str(predictions[row])+"\n")


def plot(method, modulus, predict, actual, extraMethods, num):
    plt.figure(num)
    plt.plot(predict, actual, linestyle='None', marker='o')
    plt.plot(actual, actual, linestyle='-', marker='None', color='black', alpha=0.5)
    plt.title(method+' '+extraMethods+' '+modulus)
    plt.xlabel('Predicted '+modulus)
    plt.ylabel('Actual '+modulus)


def getTrainX(filePath):
    testFile = open(filePath, 'r')
    lines = testFile.readlines()
    compNames = list(range(len(lines)))
    xVals = np.ndarray(shape=(len(lines), 19), dtype=float)
    data = {}; i = 0

    for i in range(len(lines)):
        name = lines[i].strip()
        data = whd.calcMatProps(name, '', data, False)
        compNames[i] = name
        for ind in orders:
            xVals[i][orders[ind]] = data[ind]
        i+=1
    return(compNames, xVals)




if __name__=='__main__':
    x_tr, G_vrh, K_vrh = refineSpaceGroup(groups[spacegroup])
    title = ['G$_{VRH}$', 'K$_{VRH}$']
    mods  = [G_vrh, K_vrh]
    
    iter = 0
    for num in range(1):
        x_tr, x_val, x_T, y_tr, y_val, y_T = pred.splitTrainTestValid(x_tr, G_vrh)
        x_tr, x_T, x_val = scaleData(x_tr, x_T, x_val)
        mlMethod, predict = getFunc(ml, x_tr, y_tr, x_T)

        #mlMethod, predict = getFunc(ml, x_tr, mods[num], x_T)
        #y_tr = mods[num]
        
        if ml!='NN' and ml!='DEEP':
            ab = AdaBoostRegressor(base_estimator=mlMethod, n_estimators=50, learning_rate=1.0, random_state=33)
            ab.fit(x_tr, y_tr)
            abPredict = ab.predict(x_T)

            params = getParams(ml)
            gs = GridSearchCV(mlMethod, params)
            gs.fit(x_tr, y_tr)
            gsPredict = gs.predict(x_T)
        
            abgs = GridSearchCV(ab, {'n_estimators':[10, 50, 100], 'learning_rate':[0.5, 1.0]})
            abgs.fit(x_tr, y_tr)
            abgsPredict = abgs.predict(x_T)

        plot(ml, title[num], predict, y_T, '', 1+iter)
        if ml!='NN' and ml!='DEEP':
            plot(ml, title[num], abPredict, y_T, 'adaboost', 2+iter)
            plot(ml, title[num], gsPredict, y_T, 'gridsearch', 3+iter)
            plot(ml, title[num], abgsPredict, y_T, 'adaboost and gridsearch', 4+iter)        
        iter+=4
        """
        fnames = ['G_vrh', 'K_vrh']
        plt.figure(num+1)
        plt.plot(range(1, len(abgsPredict)+1), abgsPredict, linestyle='None', marker='o')
        plt.title('Predicted values of '+fnames[num]+' for unsynthesized materials')
        plt.xlabel('compound index number')
        plt.ylabel('predicted '+fnames[num]+' GPa')
        writePredictions(fnames[num], names_T, ml, abgsPredict)
        """
    plt.show()
