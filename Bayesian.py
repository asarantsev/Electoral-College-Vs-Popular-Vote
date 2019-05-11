# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:15:25 2019

@author: UNR Math Stat
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:27:12 2019

@author: UNR Math Stat
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:12:02 2019

@author: UNR Math Stat
"""

import numpy
from numpy import random
from numpy import linalg
import pandas
import scipy
from scipy import stats
import math
import matplotlib
from matplotlib import pyplot
import statsmodels
from statsmodels import api


dataframe = pandas.read_excel('updatedData.xlsx', sheet_name = 'Logarithms')
value = dataframe.values
States = [value[k][0] for k in range(50)]
data = [value[k][1:] for k in range(50)]
USALean = value[50][1:]

#Years of elections
Years = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 
         7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13]

#Old EC
EC = [9, 3, 11, 6, 55, 9, 7, 3, 29, 16, 4, 4, 20, 11, 6, 6, 
      8, 8, 4, 10, 11, 16, 10, 6, 10, 3, 5, 6, 4, 14, 5, 29, 
      15, 3, 18, 7, 7, 20, 4, 9, 3, 11, 38, 6, 3, 13, 12, 5, 10, 3]

#New EC
redistrict = [-1, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, -1, 
              0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 
              -2, 1, 0, -1, 0, 1, -1, -1, 0, 0, 0, 3, 0, 0, 0, 0, -1, 0, 0]

ECnew = EC + numpy.array(redistrict)

ShiftedYears = [item - 13 for item in Years]
Years = ShiftedYears

Alpha = []
AlphaSE = []
Beta = []
BetaSE = []
Gamma = []
GammaSE = []
Stderr = []
Nelect = []

#Inverse chi squared
def ichi2(degFreedom, scale):
    shape = degFreedom/2
    return ((shape*scale)/random.gamma(shape))

#fit linear regression for given state
def Regression(state):
    ClearLean = []
    ClearYear = []
    ClearNation = []
    for election in range(35):
        if (not numpy.isnan(data[state][election])):
            ClearLean.append(data[state][election])
            ClearYear.append(Years[election])
            ClearNation.append(USALean[election])
    ClearLean = numpy.array(ClearLean)
    ClearYear = numpy.array(ClearYear)
    n = numpy.size(ClearLean)
    Units = numpy.ones(n)
    Design = numpy.column_stack([Units, ClearNation, ClearYear])
    #print(Design)
    TDesign = numpy.transpose(Design)
    C = numpy.matmul(TDesign, Design)
    #print(C)
    I = numpy.linalg.inv(C)
    coeff = numpy.matmul(I, TDesign).dot(ClearLean)
    residuals = ClearLean - Design.dot(coeff)
    stderr = math.sqrt(1/(n-3)*numpy.dot(residuals, residuals))
    return (n, coeff, I, stderr)


AlphaM = []
AlphaP = []
BetaM = []
BetaP = []
GammaM = []
GammaP = []
    
NSIMS = 40000
T = 1
confLevel = 0.95
simAlpha = [[] for state in range(50)]
simBeta = [[] for state in range(50)]
simGamma = [[] for state in range(50)]
simVar = [[] for state in range(50)]

#fitting linear regression and computing standard errors
#for the whole regression and coefficients
for state in range(50):
    n, coeff, I, stderr = Regression(state)
    Nelect.append(n)
    Alpha.append(coeff[0])
    Beta.append(coeff[1])
    Gamma.append(coeff[2])
    AlphaSE.append(stderr*math.sqrt(I[0][0]))
    BetaSE.append(stderr*math.sqrt(I[1][1]))
    GammaSE.append(stderr*math.sqrt(I[2][2]))
    Stderr.append(stderr)
    for sim in range(NSIMS):
        simVar[state].append(ichi2(n-3, stderr**2))
        simMeanTemp = random.multivariate_normal(coeff, simVar[state][sim]*I) 
        simAlpha[state].append(simMeanTemp[0])
        simBeta[state].append(simMeanTemp[1])
        simGamma[state].append(simMeanTemp[2])
        
level = 0.95
for state in range(50):
    tstat = stats.t.ppf(level, Nelect[state] - 3)
    AlphaM.append(Alpha[state] - AlphaSE[state] * tstat)
    AlphaP.append(Alpha[state] + AlphaSE[state] * tstat)
    BetaM.append(Beta[state] - BetaSE[state] * tstat)
    BetaP.append(Beta[state] + BetaSE[state] * tstat)
    GammaM.append(Gamma[state] - GammaSE[state] * tstat)
    GammaP.append(Gamma[state] + GammaSE[state] * tstat)
    
Coefficients = pandas.DataFrame({'state': States, 'quantity': Nelect, 
                    'alpha': Alpha, 'alphaM': AlphaM, 'alphaP': AlphaP, 
                    'beta': Beta, 'betaM': BetaM, 'betaP': BetaP,
                    'gamma': Gamma, 'gammaM': GammaM, 'gammaP': GammaP,
                    'stderr': Stderr, 'EC': EC, 'ECnew': ECnew})
    
print(Coefficients)

export_csv = Coefficients.to_csv (r'coeff&Intervals.csv', index = None, header=True)

#Simulating EC using Bayesian 
#nationwide = log(D/R)
#Time = year
#College = which EC we use?
def BayesEV(nationwide, Time, college):
    outcome = []
    StateProb = numpy.zeros(51)
    for sim in range(NSIMS):
        electoralvote = 3
        for state in range(50):
            a = simAlpha[state][sim]
            b = simBeta[state][sim]
            c = simGamma[state][sim]
            s = math.sqrt(simVar[state][sim])
            z = a + nationwide * b + Time * c + s * random.normal()
            if (z > 0):
                electoralvote = electoralvote + college[state]
                StateProb[state] = StateProb[state] + 1
        if (electoralvote > 269):
            StateProb[50] = StateProb[50] + 1
        outcome.append(electoralvote)
    StateProb = StateProb / NSIMS
    pyplot.axvline(x = 269, color = 'r')
    pyplot.hist(outcome, bins = 50)
    pyplot.show()
    print('nationwide = ', nationwide)
    print('Time = ', Time)
    return (StateProb)

output = pandas.DataFrame({
            'States': numpy.append(States, 'USA'),
            '2020 even': BayesEV(0, 1, EC),
            '2020 as 2016': BayesEV(math.log(48.2/46.1), 1, EC),
            '2020 as 2008': BayesEV(math.log(52.9/45.7), 1, EC),
            '2020 as 2004': BayesEV(math.log(48.3/50.7), 1, EC),
            '2024 even': BayesEV(0, 3, ECnew),
            '2024 as 2016': BayesEV(math.log(48.2/46.1), 3, ECnew)
            })
    
export_csv = output.to_csv (r'BayesianScenarios.csv', index = None, header=True)


nationwideRange = numpy.arange(-0.2, 0.2, 0.01)
nationwideNumber = numpy.size(nationwideRange)
NationWin = numpy.zeros(nationwideNumber)

#Plot data for individual state together with linear trend
def plotElections(state):
    ClearLean = []
    ClearYear = []
    for election in range(35):
        if (not numpy.isnan(data[state][election])):
            ClearLean.append(data[state][election])
            ClearYear.append(Years[election]*2+2018)
    s = stats.linregress(ClearYear, ClearLean)
    b = s.intercept
    m = s.slope
    ClearYear = numpy.array(ClearYear)
    pyplot.plot(ClearYear, ClearLean, 'go')
    pyplot.plot(ClearYear, b + m * ClearYear, 'r-')
    pyplot.show()
    return(States[state])

#Cook Partisan Voting Index 
CPVI = numpy.array([-14, -9, -5, -15, 12, 1, 6, 6, -2, -5, 18, -19, 
                    7, -9, -3, -13, -15, -11, 3, 12, 12, 1, 1, -9,
                    -9, -11, -14, 1, 0, 7, 3, 12, -3, -17, -3, -20,
                    5, 0, 10, -8, -14, -14, -8, -20, 15, 1, 7, -19, 0, -25])


#print('2024 even')
#print(EV(0, 3, ECnew)) 

    
    