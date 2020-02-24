##### Fixing getData()
from pymatgen import MPRester
import pymatgen as pmg
import numpy as np
import json
import re

global key
key = 'YOUR API KEY HERE'

def calcMatProps(compound, elasticity, dataDict, getModuli):
    """ inputs: string of compound, dictionary for elasticity, input data dictionary
        * main function; calls the other functions to create a comprehensive dictionary
          of data about radii, row, column, mass, valence electrons, etc.
        returns: dictionary of data
    """
    els, nums = sepElements(compound)
    dataDict = getAvgs(els, nums, dataDict)
    
    if getModuli==True:
        dataDict["G_VRH"] = elasticity["G_VRH"]
        dataDict["K_VRH"] = elasticity["K_VRH"]
    """
    else:
        dataDict['band_gap'] = getBandGap(compound, 'TextFiles/bandgaps.txt')
    """

    #dataDict["elPoissonAvg"] = avgFromFile(els, nums, 'TextFiles/elementalPoisson.txt')
    dataDict["elAffAvg"] = avgFromFile(els, nums, 'TextFiles/electronAffinities.txt')
    formula = pmg.Composition(compound)
    dataDict["elNegAvg"] = formula.average_electroneg
    
    return(dataDict)


def getBandGap(comp, filename):
    file = open(filename, 'r')
    elAffDict = {}; elAff = 0
    for line in file:
        line = line.split('\t')
        if line[0] == comp:
            return(line[1])
        
    return(-1)


def avgFromFile(comp, nums, filename):
    file = open(filename, 'r')
    elemDict = {}; featureSum = 0
    for line in file:
        line = line.split('\t')
        elemDict[line[0]] = line[1]
        
    for el in range(len(comp)):
        featureSum += float(elemDict[comp[el]])*float(nums[el])

    return(featureSum/sum(nums))



def sepElements(comp):
    """ inputs: string of the compound
        * splits the elements and the number of each element in the compound
        returns: list of elements, list of number of each element
    """

    beg = 0; end = 0
    elems = re.findall('[A-Z][^A-Z]*', comp)
    numEl = np.zeros(len(elems))
    for elem in range(len(elems)):
        val = re.findall('\d+', elems[elem])
        if len(val) == 0:
            val = 1
        else:
            val = int(val[0])
        numEl[elem] = val
        elems[elem] = ''.join([i for i in elems[elem] if not i.isdigit()])

        for letter in elems[elem]:
            if letter == '(':
                word = elems[elem]
                beg = elem+1
                elems[elem].replace('(', '')
                elems[elem] = word[0:-1]
            elif letter == ')':
                word = elems[elem]
                end = elem
                elems[elem].replace(')', '')
                elems[elem] = word[0:-1]
                numEl[beg:end] = numEl[end]
    return(elems, numEl) 



def getValence(elem, elStruct):
    sValAvg = 0; pValAvg = 0; dValAvg = 0; fValAvg = 0
    if elem != 'H' and elem != 'He':
        last = elStruct[-1]
        secLast = elStruct[-2]
        orbNum = int(last[0])
        orbNum2 = int(secLast[0])
        orbNum = max(orbNum, orbNum2)
        
        for orb in range(len(elStruct)):
            currOrb = elStruct[orb]
            curr = int(currOrb[0])
            if curr==orbNum or curr==orbNum-1 or curr==orbNum-2:
                orbType = currOrb[1]

                if orbType=='s' and curr==orbNum:
                    sValAvg += currOrb[2]
                elif orbType=='p' and curr==orbNum:
                    pValAvg += currOrb[2]
                elif (orbType=='d' and curr==orbNum) or (orbType=='d' and curr==orbNum-1):
                    dValAvg += currOrb[2]
                elif (orbType=='f' and curr==orbNum) or (orbType=='f' and curr==orbNum-1) or (orbType=='f' and curr==orbNum-2):
                    fValAvg += currOrb[2]

    elif elem == 'H':
        sValAvg = 1
    elif elem == 'He':
        sValAvg = 0
    return(sValAvg, pValAvg, dValAvg, fValAvg)



def getAvgs(elmArr, numArr, outDict):
    """ inputs: array of the strings of elements, array of the corresponding
        number of each element, the current dictionary of data
        * calculates averages of row, collumn, mass, radius, valence electrons,
          and the fraction of each type of valence electron
        returns: dictionary of data edited to include new variables
    """

    rAvg = 0; mAvg = 0
    colAvg = 0; rowAvg = 0
    atNumAvg = 0
    sValAvg = 0; pValAvg = 0; dValAvg = 0; fValAvg = 0
    sValFrac = 0; pValFrac = 0; dValFrac = 0; fValFrac = 0
    for elInd in range(len(elmArr)):
        mult = numArr[elInd]
        element = pmg.Element(elmArr[elInd]) 

        try:
            rAvg += element.atomic_radius*mult
        except TypeError:
            rAvg += element.atomic_radius_calculated*mult
            
        mAvg += element.atomic_mass*mult
        atNumAvg += element.Z*mult
        rowAvg += element.row*mult
        colAvg += element.group*mult
        elStruc = element.full_electronic_structure
        sVal, pVal, dVal, fVal = getValence(elmArr[elInd], elStruc)
        
        sValAvg += sVal*mult
        pValAvg += pVal*mult
        dValAvg += dVal*mult
        fValAvg += fVal*mult
        
    atomNum = np.sum(numArr)
    outDict['rAvg'] = rAvg/atomNum
    outDict['mAvg'] = mAvg/atomNum
    outDict['colAvg'] = colAvg/atomNum
    outDict['rowAvg'] = rowAvg/atomNum
    outDict['atNumAvg'] = atNumAvg/atomNum
    
    valSum = sValAvg+pValAvg+dValAvg+fValAvg
    outDict['sValFrac'] = sValAvg/valSum
    outDict['pValFrac'] = pValAvg/valSum
    outDict['dValFrac'] = dValAvg/valSum
    outDict['fValFrac'] = fValAvg/valSum
        
    outDict['sValAvg'] = sValAvg/atomNum
    outDict['pValAvg'] = pValAvg/atomNum
    outDict['dValAvg'] = dValAvg/atomNum
    outDict['fValAvg'] = fValAvg/atomNum

    return(outDict)



def getCompounds(apiKey):
    mpr = MPRester(apiKey)
    data = mpr.query({'elasticity': {'$exists': True}}, ['material_id', 'full_formula', 'elasticity', 'formation_energy_per_atom', 'band_gap', 'spacegroup', 'energy_per_atom'])
    return(data)



def getData(compounds, maxG, maxK):
    bgFile = open('bandgaps.txt', 'w')
    dList = []
    for comp in range(len(compounds)):
        currInfo = compounds[comp]
        compName = currInfo['full_formula']
        elasticity = currInfo['elasticity']
        spacegroup = currInfo['spacegroup']
        G = elasticity['G_VRH']
        K = elasticity['K_VRH']
        
        if G>30. and K>50. and G<=maxG and K<=maxK:
            d={}
            d["formula"] = compName
            d["material_id"] = currInfo['material_id']
            d["space_group"] = spacegroup['number']
            d["G_VRH"] = G
            d["K_VRH"] = K
            d["formation_energy_per_atom"] = currInfo['formation_energy_per_atom']
            try:
                d["elastic_tensor"] = elasticity['elastic_tensor']
            except ValueError:
                d["elastsic_tensor_original"] = elasticity['elastic_tensor_original']
            bgFile.write(compName+'\t'+str(currInfo['band_gap'])+'\n')
            dList.append(d)

    return(dList)


if __name__ == '__main__':
    info = getCompounds(key)
    with open('TextFiles/materialsData.json', 'w') as wrFile:
        
        with MPRester(key) as m:
            currInfo    = m.query('mp-66', ['elasticity'])
            currInfo    = currInfo[0]
            elasticity  = currInfo['elasticity']
            diamondGVRH = elasticity['G_VRH']
            diamondKVRH = elasticity['K_VRH']

            dictList = getData(info, diamondGVRH, diamondKVRH)
            json.dump(dictList, wrFile)
