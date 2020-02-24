import re
import pymatgen as pmg


class Material:

    # Initializer
    def __init__(self, name):
        self.name = name


    def separateElements(self):
        incomp = self.name
        outElems = []; outNums = []
        split = re.findall('\([A-Za-z]+\)|[A-Za-z]+|\d+\.+\d+|\d+', incomp)
        for ind in range(len(split)):
            try:
                type(float(split[ind]))

            except ValueError:
                try:
                    val = float(split[ind+1])
                    curr = split[ind]
                    if(curr[0]!='('):
                        comp = re.findall('[A-Z][a-z]?', split[ind])
                        for i in range(len(comp)):
                            if(i!=len(comp)-1):
                                outElems.append(comp[i])
                                outNums.append(1.)
                            else:
                                outElems.append(comp[i])
                                outNums.append(val)
                    else:
                        comp = re.findall('[A-Z][a-z]?', split[ind])
                        for i in range(len(comp)):
                            outElems.append(comp[i])
                            outNums.append(val)
                        
                except:
                    comp = re.findall('[A-Z][a-z]?', split[ind])
                    for i in range(len(comp)):
                        outElems.append(comp[i])
                        outNums.append(1.)
        self.elems = outElems
        self.nums = outNums

        
    def getSpacegroup(self, compound):
        self.spacegroup = compound['space_group']

        
    def setData(self, compound):
        #self.formation_energy_per_atom = compound['formation_energy_per_atom']
        #self.bandgap = compound['band_gap']
        self.id      = compound['material_id']
        elasticity   = compound['elastic_tensor']
        self.G_VRH   = compound['G_VRH']
        self.K_VRH   = compound['K_VRH']
        self.vickers = 2*((self.G_VRH/self.K_VRH)**2*self.G_VRH)**(0.585)-3.0
        
        try:
            self.elastic_tensor = compound['elastic_tensor']
        except ValueError:
            self.elastsic_tensor_original = compound['elastic_tensor_original']

        
    def setAverages(self):
        self.row, self.column, self.avgMass, self.avgRadius, self.avgAtomicNumber = 0, 0, 0, 0, 0
        self.avgThermalConductivity, self.avgBoilingPoint, self.avgMeltingPoint = 0, 0, 0
        for ind in range(len(self.elems)):
            num = self.nums[ind]
            el = pmg.Element(self.elems[ind])
            self.row += el.row*num
            self.column += el.group*num
            self.avgMass += el.atomic_radius*num
            self.avgAtomicNumber += el.Z*num

            temp = str(el.thermal_conductivity).split()
            self.avgThermalConductivity += float(temp[0])*num
            temp = str(el.boiling_point).split()
            #self.avgBoilingPoint += float(temp[0])*num
            temp = str(el.melting_point).split()
            self.avgMeltingPoint += float(temp[0])*num

            try:
                self.avgRadius += el.atomic_radius*num
            except TypeError:
                self.avgRadius += el.atomic_radius_calculated*num

        formula = pmg.Composition(self.name)
        self.avgElectronegativity = formula.average_electroneg
        self.avgElectronAffinity = self.avgFromFile(self.elems, self.nums, 'TextFiles/electronAffinities.txt') 

        total = sum(self.nums)
        self.row                      = self.row/total
        self.column                   = self.column/total
        self.avgMass                  = self.avgMass/total
        self.avgAtomicNumber          = self.avgAtomicNumber/total
        self.avgRadius                = self.avgRadius/total

        self.avgThermalConductivity   = self.avgThermalConductivity/total
        self.avgBoilingPoint          = self.avgBoilingPoint/total
        self.avgMeltingPoint          = self.avgMeltingPoint/total

        
    def setValence(self):
        self.sValence, self.pValence, self.dValence, self.fValence, self.totalValence = 0, 0, 0, 0, 0
        for elem in self.elems:
            if elem != 'H' and elem != 'He':
                elem = pmg.Element(elem)
                elStruct = elem.full_electronic_structure
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
                            self.sValence += currOrb[2]
                        elif orbType=='p' and curr==orbNum:
                            self.pValence += currOrb[2]
                        elif (orbType=='d' and curr==orbNum) or (orbType=='d' and curr==orbNum-1):
                            self.dValence += currOrb[2]
                        elif (orbType=='f' and curr==orbNum) or (orbType=='f' and curr==orbNum-1) or (orbType=='f' and curr==orbNum-2):
                            self.fValence += currOrb[2]

            elif elem == 'H':
                self.sValence += 1
            elif elem == 'He':
                continue
        self.totalValence = self.sValence+self.pValence+self.dValence+self.fValence
        self.sValenceFrac = self.sValence/self.totalValence
        self.pValenceFrac = self.pValence/self.totalValence
        self.dValenceFrac = self.dValence/self.totalValence
        self.fValenceFrac = self.fValence/self.totalValence

        self.sValenceAvg = self.sValence/sum(self.nums)
        self.pValenceAvg = self.pValence/sum(self.nums)
        self.dValenceAvg = self.dValence/sum(self.nums)
        self.fValenceAvg = self.fValence/sum(self.nums)


    def avgFromFile(self, comp, nums, filename):
        file = open(filename, 'r')
        elemDict = {}; featureSum = 0
        for line in file:
            line = line.split('\t')
            elemDict[line[0]] = line[1]

        for el in range(len(comp)):
            featureSum += float(elemDict[comp[el]])*float(nums[el])

        return(featureSum/sum(nums))
