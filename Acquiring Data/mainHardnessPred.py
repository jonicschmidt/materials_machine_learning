import matplotlib.pyplot as plt
import scipy.stats as scp
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import yaml
import fromJSON as frm

def prepArrs(filename, des_spgr):
    names    = ['formation_energy_per_atom', 'band_gap', 'energy_per_atom', 'rAvg', \
                'mAvg', 'colAvg', 'rowAvg', 'atNumAvg', 'sValFrac', 'pValFrac', \
                'dValFrac', 'fValFrac', 'sValAvg', 'pValAvg', 'dValAvg', 'fValAvg', \
                'elNegAvg', 'elAffAvg', 'G_VRH', 'K_VRH']
    groups   = {'triclinic':[1,2], 'monoclinic':[3,15], 'orthorhombic':[16,74], \
                'tetragonal':[75,142], 'trigonal':[143,167], 'hexagonal':[168,194], \
                'cubic':[195,230], 'all':[0,230]}

    thisFile = open(filename, 'r')
    lines = thisFile.readlines()
    length = len(lines); arrs = {}; dic = {}; num = 0

    for num in range(len(names)):
        arrs[names[num]] = np.zeros(length)

    for line in lines:
        dic = line
        dic = yaml.load(dic)
        limL, limH = groups[des_spgr]
        spaceGroup = float(dic['spacegroup_symbol'])
        
        if spaceGroup>limL and spaceGroup<limH:
            for name in range(len(names)):
                array = arrs[names[name]]
                try:
                    array[num]= float(dic[names[name]])
                except ValueError:
                    del arrs[names[name]]
            num += 1 

    K_VRH = arrs['K_VRH']
    for each in arrs:
        try:
            each = each[K_VRH!=0.]
        except TypeError:
            continue
            
    arr2D = [[(arrs[items])[i] for i in range(len(arrs['K_VRH']))] for items in arrs]
    return(arr2D, arrs['G_VRH'], arrs['K_VRH'])

            
def splitTrainTestValid(xVals, yVals):
    """ Inputs: xVals (2D array of the x-data (bandwidth, energy, etc.) with length corresponding to each material)
        *splits the data into train, test, and validation arrays
        Returns: training, testing, and validation data for each type (xVals, K_VRH, G_VRH)
    """
    x_train, x_T, y_train, y_T = train_test_split(xVals, yVals, test_size=0.20, shuffle=True)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train,  y_train, test_size=0.125, shuffle=False)
    return(x_tr, x_val, x_T, y_tr, y_val, y_T)
    
    
def LASSO(x_train, y_train, x_Test):
    """ *uses Lasso method to predict GVRH and KVRH to corresponding x_Test data
    """
    lasso = Lasso(random_state=33, alpha=0.5, max_iter=100000000)
    lasso.fit(x_train, y_train)
    pred = lasso.predict(x_Test)
    return(lasso, pred)


def RIDGE(x_train, y_train, x_Test):
    """ *uses Ridge method to predict GVRH and KVRH to corresponding x_Test data
    """ 
    ridge = Ridge(random_state=33)
    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_Test)
    return(ridge, pred) 
    

def FOREST(x_train, y_train, x_Test):
    """ *uses RandomForestClassifier method to predict GVRH and KVRH to corresponding x_Test data
    """ 
    rf = RandomForestRegressor(bootstrap=True, max_features='auto', n_estimators=10, random_state=33)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_Test)
    return(rf, pred) 


def DECISIONTREE(x_train, y_train, x_Test):
    """ *uses DecisionTreeClassifier method to predict GVRH and KVRH to corresponding x_Test data
    """ 
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    DecisionTreeRegressor(criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, random_state=None)
    pred = dt.predict(x_Test)
    return(dt , pred)
    


if __name__ == '__main__':
    spgr = 'cubic' #Enter your desired space group!!!
    ml = 'FOREST'

    myFile = open('TextFiles/materialsData.txt', 'r')
    xVals, G_VRH, K_VRH = prepArrs(myFile, spgr)
    
    xG_tr, xG_T, G_tr, G_T = train_test_split(xVals, G_VRH, test_size=0.20, shuffle=True)
    xK_tr, xK_T, K_tr, K_T = train_test_split(xVals, K_VRH, test_size=0.20, shuffle=True)

    gMethod, gPredict = frm.getFunc(ml, xG_tr, yG_tr, xG_T)
    kMethod, kPredict = frm.getFunc(ml, xK_tr, yK_tr, xK_T)

    plt.figure(1)
    plt.plot(gPredict, G_T, linestyle='None', marker='o')
    plt.title('$G_{VRH}$ Predicted vs. Actual')
    plt.xlabel('Predicted $G_{VRH}$')
    plt.ylabel('Actual $G_{VRH}$')

    plt.figure(2)
    plt.plot(kPredict, K_T, linestyle='None', marker='o')
    plt.title('K_{VRH}$ Predicted vs. Actual')
    plt.xlabel('Predicted $K_{VRH}$')
    plt.ylabel('Actual $K_{VRH}$')
    
    plt.show()
