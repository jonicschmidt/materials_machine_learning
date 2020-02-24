from Material import Material
from Set import Set
from MachineLearningTechniques import MachineLearningTechniques
from scipy.stats import pearsonr

import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import sklearn.preprocessing as prepro
import time

def vickersHardness(G, K):
    size = len(G)
    k = np.multiply(G, (1/K))
    hardness = 2*(np.multiply(np.multiply(k, k), G))**(0.585)-3
    plt.plot(np.linspace(1, size, size), hardness, linestyle='None', marker='o')
    plt.title("Vickers Hardness")
    plt.ylabel("$H_{V}$")
    plt.xlabel("individual material")
    return(hardness)


def readMaterials(filename):
    fopen = open(filename, "r")
    materials = np.array(fopen.read().split())
    fopen.close()
    return(materials)
    

def plotML(title, model, tsx, tsval, vx, vval):
    y = model.predict(tsx)
    p = pearsonr(y, tsval)
    c = cross_val_score(model, vx, vval, cv=3)
    plt.plot(y, tsval, linestyle='None', marker='o')
    plt.plot(tsval, tsval, color='k', alpha=0.5)
    plt.title(title+" R: "+str(p[0])+" val: "+str(c))
    return(y)


def writeData(materialsList, predG, predK, predHard, filename):
    calcHard = np.zeros(len(materialsList))
    fopen = open(filename, "w")
    fopen.write("material name     |     G vrh     |     K vrh     |     Vicker's Hardness Pred      |      Vicker's Hardness Calc\n")
    for i in range(len(materialsList)):
        fopen.write(materialsList[i]+"    "+str(predG[i])+"GPa   "+str(predK[i])+"GPa   "+\
                    str(predHard[i])+"GPa   "+str(2*((predG[i]/predK[i])**2*predG[i])**(0.585)-3)+"GPa\n")
        calcHard[i] = 2*((predG[i]/predK[i])**2*predG[i])**(0.585)-3
    fopen.close()


    


if __name__ == '__main__':
    dataFile = 'TextFiles/ec.json'            # 'TextFiles/materialsData.json' or 'TextFiles/ec.json'
    spacegroup = 'all'
    machinelearning = 'FOREST'
    
    groups = {'triclinic':[1,2], 'monoclinic':[3,15], 'orthorhombic':[16,74], 'tetragonal':[75,142],\
              'trigonal':[143,167], 'hexagonal':[168,194], 'cubic':[195,230], 'all':[0,230]}

    limits = groups[spacegroup]
    File   = open(dataFile, "r")
    data   = json.load(File)
    materials = []

    data = shuffle(data)
    # adding materials that are in the spacegroup to a list
    for elem in range(len(data)):
        curr = data[elem]
        material = Material(curr['formula'])
        material.getSpacegroup(curr)
        if limits[0]<material.spacegroup and limits[1]>material.spacegroup:
            material.separateElements()
            material.setData(curr)
            material.setValence()
            material.setAverages()
            materials.append(material)

    # shuffle, split, and scale the data
    size_70p = int(np.floor(len(materials)*0.7))                      # gets index at 70% of material vector length
    size_90p = int(np.floor(len(materials)*0.9))                      # gets index at 90% of material vector length

    # split material vector into train, test, and validate vectors
    train = Set(materials[0:size_70p])
    test  = Set(materials[size_70p+1:size_90p])
    valid = Set(materials[size_90p+1:len(materials)])

    # read in unsynthesized materials from file to predict values of
    newMatNames = np.concatenate((readMaterials("TextFiles/datsimple.txt"), readMaterials("TextFiles/datacomplex.txt")), axis=0)
    new_materials = []
    for i in range(len(newMatNames)):
        material = Material(newMatNames[i])
        material.separateElements()
        material.setValence()
        material.setAverages()
        new_materials.append(material)
    new_materials = shuffle(new_materials)
    new = Set(new_materials)
    
    # scale the train, text, and validate vectors to normalized training data
    scaler  = prepro.StandardScaler(copy=True, with_mean=True, with_std=True).fit(train.x)
    scaler.fit(np.transpose(train.x))
    train.x = scaler.transform(np.transpose(train.x))
    test.x  = scaler.transform(np.transpose(test.x))
    valid.x = scaler.transform(np.transpose(valid.x))
    new.x   = scaler.transform(np.transpose(new.x))

    """
    scalerSq  = prepro.StandardScaler(copy=True, with_mean=True, with_std=True).fit(train.xsq)
    scalerSq.fit(np.transpose(train.xsq))
    train.xsq = scalerSq.transform(np.transpose(train.xsq))
    test.xsq  = scalerSq.transform(np.transpose(test.xsq))
    valid.xsq = scalerSq.transform(np.transpose(valid.xsq))
    new.xsq = scalerSq.transform(np.transpose(new.xsq))
    """
    
    # setting the features matrices
    train.setFeatures()
    test.setFeatures()
    valid.setFeatures()
    new.setFeatures()
    
    # calling machine learning techniques on the data
    ml_GVRH              = MachineLearningTechniques(train.x, train.GVRH, test.x, test.GVRH, valid.x, valid.GVRH)
    ml_KVRH              = MachineLearningTechniques(train.x, train.KVRH, test.x, test.KVRH, valid.x, valid.KVRH)
    ml_vick              = MachineLearningTechniques(train.x, train.vickers, test.x, test.vickers, valid.x, valid.vickers)
    #mlsq_GVRH            = MachineLearningTechniques(train.xsq, train.GVRH, test.xsq, test.GVRH, valid.xsq, valid.GVRH)
    #mlsq_KVRH            = MachineLearningTechniques(train.xsq, train.KVRH, test.xsq, test.KVRH, valid.xsq, valid.KVRH)
    
    # Call forest methods on model
    fG1, fG2, fG3, fG4 = ml_GVRH.FOREST()
    fK1, fK2, fK3, fK4 = ml_KVRH.FOREST()
    fH1, fH2, fH3, fH4 = ml_vick.FOREST()
    #fG1sq, fG2sq, fG3sq, fG4sq = mlsq_GVRH.FOREST()
    #fK1sq, fK2sq, fK3sq, fK4sq = mlsq_KVRH.FOREST()

    train.x = np.transpose(train.x)
    test.x  = np.transpose(test.x)
    valid.x = np.transpose(valid.x)
    new.x   = np.transpose(new.x)
    
    fG3lin = fG3.fit(train.x, train.GVRH)
    fK3lin = fK3.fit(train.x, train.KVRH)
    fH3lin = fH3.fit(train.x, train.vickers)
    #fG3sq = fG3sq.fit(train.xsq, train.GVRH)
    #fK3sq = fK3sq.fit(train.xsq, train.KVRH)
    
    plt.figure(1)
    plotML("AB Forest G$_{VRH}$", fG3, test.x, test.GVRH, valid.x, valid.GVRH)
    plt.xlabel("predicted G$_{VRH}$ [GPa]")
    plt.ylabel("actual G$_{VRH}$ [GPa]")
    plt.savefig("Figures/curr/AB_Forest_Linear_test_GVRH.pdf")
    
    plt.figure(2)
    plotML("AB Forest K$_{VRH}$", fK3, test.x, test.KVRH, valid.x, valid.KVRH)
    plt.xlabel("predicted K$_{VRH}$ [GPa]")
    plt.ylabel("actual K$_{VRH}$ [GPa]")
    plt.savefig("Figures/curr/AB_Forest_Linear_test_KVRH.pdf")
    
    pred_fG3 = fG3.predict(new.x)
    pred_fK3 = fK3.predict(new.x)
    plt.figure(3)
    vhard = vickersHardness(pred_fG3, pred_fK3)

    size = len(pred_fG3)
    arr = np.linspace(1, size, size)

    plt.figure(4)
    plt.plot(arr, pred_fG3, linestyle='None', marker='o')
    plt.title("AB Forest Predicted $G_{VRH}$ for Unsynthesized")
    plt.ylabel("$G_{VRH}$ [GPa]")
    plt.xlabel("material index")
    plt.savefig("Figures/curr/AB_Forest_Linear_unsynth_GVRH.pdf")
    
    plt.figure(5)
    plt.plot(arr, pred_fK3, linestyle='None', marker='o')
    plt.title("AB Forest Predicted $K_{VRH}$ for Unsynthesized")
    plt.ylabel("$K_{VRH}$ [GPa]")
    plt.xlabel("material index") 
    plt.savefig("Figures/curr/AB_Forest_Linear_unsynth_KVRH.pdf")
    
    writeData(newMatNames, pred_fG3, pred_fK3, vhard, "Figures/curr/unsynthesizedPredictedValues.txt")
    
    plt.show()
