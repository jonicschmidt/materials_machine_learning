import numpy as np

np.set_printoptions(threshold=np.inf)

class Set:

    # Initializer
    def __init__(self, data):
        self.data = data
        self.setFeatures()

        
    def setFeatures(self):
        materials = self.data
        num = 0
        size = 20
        linfeatures = np.ndarray(shape=(size+1, len(materials)), dtype=float)
        sqfeatures = np.ndarray(shape=(size**2+size+1, len(materials)), dtype=float)
        G = np.zeros(len(materials)); K = np.zeros(len(materials)); H = np.zeros(len(materials))

        for each in materials:
            vals = np.array([each.sValence, each.pValence, each.dValence, each.fValence, each.sValenceAvg,\
                    each.pValenceAvg, each.dValenceAvg, each.fValenceAvg, each.sValenceFrac,\
                    each.pValenceFrac, each.dValenceFrac, each.fValenceFrac, \
                    each.row, each.column, each.avgMass, each.avgRadius, \
                    each.avgAtomicNumber, each.avgThermalConductivity, each.avgBoilingPoint, each.avgMeltingPoint])

            #self.avgElectronegativity
            #self.avgElectronAffinity 
            
            deg2 = np.multiply.outer(vals, vals)
            for curr in range(size):
                linfeatures[curr][num]  = vals[curr]
                sqfeatures[curr][num]  = vals[curr]
                
            for i in range(size):
                val = (i+1)*size
                for j in range(size):
                    sqfeatures[val+j][num] = deg2[i][j]

            try:
                G[num] = each.G_VRH
                K[num] = each.K_VRH
                H[num] = each.vickers
            except AttributeError:
                continue
            num += 1

        self.x = linfeatures
        self.sqx = sqfeatures
        self.GVRH = G
        self.KVRH = K
        self.vickers = H
