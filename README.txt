writeHardnessData.py:
  - the main file for collecting data from the MaterialsProject Database
    	* reads in data from database for all compounds
	* writes data to file for elements with specified G_vrh and K_vrh

mainHardnessPred.py:
  - file that contains the machine learning code used by fromJSON.py
    	* reads data from file 'materialsData.txt' in sub-folder TextFiles
	* plots the results of using this data to predict the shear and bulk
	  moduli of the materials in the file

fromJSON.py:
  - reads data for compounds from ec.json, the file of 'cleaned' data from Jong et. al
    	* uses ec.json and features calculated from writeHardness.py to predict
	  the shear and bulk moduli for materials
	* input the desired spagegroup to look at (can choose 'all') and the desired
	  machine learning technique ('LASSO', 'RIDGE', 'TREE', or 'FOREST')
	* creates plots of predicted moduli versus the measured moduli for the input
	  machine learning technique after adaptive boosting, grid search CV, both,
	  and neither

plotAndCoeff.py:
  - plots selected features against the bulk and shear moduli; calculates Pearson
    correlation coefficient for each feature
        * reads data from file 'materialsData.txt' in sub-folder TextFiles
	* creates two main plots that correspond to shear modulus and bulk modulus
	* for each plot, creates a subplot for each feature (up to 4) and plots
	  each feature against the modulus, showing the corresponding Pearson
	  correlation coefficient in the title of the subplot
	       ~ these values are very low, because they're for one feature,
	         many features are used for the machine learning algorithms

Material.py:
  - contains the Material class

Set.py:
  - contains the Set class to hold the Material objects

MachineLearningTechniques.py:
  - contains the different MachineLearningTechniques

materialsDriver.py:
  - the driver for the program








Data was aquired through the Materials Project using PyMatGen and the Materials API.

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323

M. de Jong, W. Chen, T. Angsten, A. Jain, R. Notestine, A. Gamst, M. Sluiter, C. K. Ande, S. van der Zwaag, J. J. Plata, C. Toher, S. Curtarolo, G. Ceder, K. A. Persson, M. Asta
Charting the complete elastic properties of inorganic crystalline compounds
Scientific Data 2: 150009 (2015).
doi:10.1038/sdata.2015.9

S. P. Ong, W. D. Richards, A. Jain, G. Hautier, M. Kocher, S. Cholia, D. Gunter, V. L. Chevrier, K. Persson, G. Ceder
Python Materials Genomics (pymatgen) : A Robust, Open-Source Python Library for Materials Analysis.
Computational Materials Science, 2013, 68, 314–319.
doi:10.1016/j.commatsci.2012.10.028

S. P. Ong, S. Cholia, A. Jain, M. Brafman, D. Gunter, G. Ceder, and K. A. Persson
The Materials Application Programming Interface (API): A simple, flexible and efficient API for materials data based on REpresentational State Transfer (REST) principles.
Computational Materials Science, 2015, 97, 209–215.
doi:10.1016/j.commatsci.2014.10.037