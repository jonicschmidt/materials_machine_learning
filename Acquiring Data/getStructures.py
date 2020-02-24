import pymatgen as pmg
from pymatgen import MPRester
import numpy as np
import json

def getCompounds(required):
    mpr = MPRester(key)
    data  = mpr.query({required: {'$exists': True}}, ['full_formula', 'material_id', 'spacegroup', required])
    return(data)


def getData(compounds):
    dList = []
    d = {}
    for comp in range(len(compounds)):
        info = compounds[comp]
        d["formula"] = info["full_formula"]
        d["material_id"] = info["material_id"]
        d["spacegroup"] = info["spacegroup"]
        d["cif"] = info["cif"]
        dList.append(d)
    return(dList) 


def main(requiredProp, fileName):
    info = getCompounds(requiredProp)

    with open(fileName, 'w') as wrFile:
        with MPRester(key) as m:
            dictList = getData(info)
            json.dump(dictList, wrFile)




if __name__ == '__main__':
    req = 'cif'
    outFile = 'structures.json'
    
    global key
    key = 'YOUR API KEY HERE'

    main(req, outFile)
