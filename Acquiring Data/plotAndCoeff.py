import matplotlib.pyplot as plt
import fromJSON as frm
import scipy.stats as scp
import numpy as np
import pymatgen as pmg
import json
import yaml

np.set_printoptions(threshold=np.inf)

names    = ['rAvg', 'mAvg', 'colAvg', 'rowAvg', 'atNumAvg', 'sValFrac', 'pValFrac', 'dValFrac', \
            'fValFrac', 'sValAvg', 'pValAvg', 'dValAvg', 'fValAvg', 'elAffAvg']

titles   = {'formation_energy_per_atom':'E$_{form}$ per atom','band_gap':'Band Gap',\
            'energy_per_atom':'E per atom', 'rAvg':'Average Radius', 'mAvg':'Average Mass',\
            'colAvg':'Average Column', 'rowAvg':'Average Row', 'atNumAvg':'Average Atomic Number',\
            'sValFrac':'S Valence $e^{-}$ Fraction', 'pValFrac':'P Valence $e^{-}$ Fraction',\
            'dValFrac':'D Valence $e^{-}$ Fraction', 'fValFrac':'F Valence $e^{-}$ Fraction',\
            'sValAvg':'Average S Valence $e^{-}s$', 'pValAvg':'Average P Valence $e^{-}s$',\
            'dValAvg':'Average D Valence $e^{-}s$', 'fValAvg':'Average F Valence $e^{-}s$',\
            'elAffAvg':'Average Electron Affinity', 'elNegAvg':'Average Electronegativity',\
            'G_VRH':'$G_{VRH}$', 'K_VRH':'$K_{VRH}$'}

orders   = {'rAvg': 0, 'mAvg': 1, 'colAvg': 2, 'rowAvg': 3, 'atNumAvg': 4, 'sValFrac': 5,\
           'pValFrac': 6, 'dValFrac': 7, 'fValFrac': 8, 'sValAvg': 9, 'pValAvg': 10,\
           'dValAvg': 11, 'fValAvg': 12, 'elAffAvg': 13}

groups   = {'triclinic':[1,2], 'monoclinic':[3,15], 'orthorhombic':[16,74], 'tetragonal':[75,142],\
            'trigonal':[143,167], 'hexagonal':[168,194], 'cubic':[195,230], 'all':[0,230]}



if __name__ == '__main__':
    """ main method
        * inputs data from filteredMaterialsData.json; reads through the
          data dictionary for each compound and creates arrays from the data
        * for both G_VRH and K_VRH, plots each array and the pearson value for
          how well each input correlates with the G_VRH and K_VRH data
        * writes the pearson values to file, identifying the modulus and data used
    """

    des_spgr = 'all' #Enter your desired space group!!!
    groups   = {'triclinic':[1,2], 'monoclinic':[3,15], 'orthorhombic':[16,74], 'tetragonal':[75,142], \
                'trigonal':[143,167], 'hexagonal':[168,194], 'cubic':[195,230], 'all':[0,230]}

    with open('TextFiles/materialsData.json') as jFile:
        data = json.load(jFile)
        
        wFile = open('TextFiles/pearsonVals.txt', 'w')
        stress   = ['$G_{VRH}$', '$K_{VRH}$']

        xs, Gvrh, Kvrh = frm.refineSpaceGroup(groups[des_spgr], orders)
        xs = np.transpose(xs)
        stressType = [Gvrh, Kvrh]
        for ind in range(len(stress)):
            plt.figure(ind+1)
            currStress = stress[ind]
            
            wFile.write('\n'+currStress+':\n')

            for k in range(1, len(names)):
                curr = stressType[ind]
                arr = xs[:][orders[names[k]]]
                
                a = plt.subplot(4,4,k+1)
                pearVal = scp.pearsonr(arr, curr)
                a.plot(arr, curr, marker='o', alpha=0.2, linestyle='None')
                string = titles[names[k]]+' '+str(pearVal)+'\n'
                a.set_title(currStress+' vs. '+titles[names[k]]+' '+str(round(pearVal[0], 5)))
                a.set_xlabel(titles[names[k]])
                a.set_ylabel(currStress)
                plt.xscale('linear')
                plt.yscale('linear')
                plt.subplots_adjust(hspace=0.5)
                wFile.write(string)

    plt.show()
    wFile.close()
