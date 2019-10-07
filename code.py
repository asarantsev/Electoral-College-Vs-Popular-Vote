# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:53:15 2019

@author: UNR Math Stat
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:00:42 2019

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


dataframe = pandas.read_excel('data.xlsx', sheet_name = 'Sums')
value = dataframe.values
States = [value[k][0] for k in range(50)]
data = [value[k][1:] for k in range(50)]
USALean = value[50][1:]

dataframeEC = pandas.read_excel('data.xlsx', sheet_name = 'EC')
valueEC = dataframeEC.values
EC = valueEC[:, 1]
ECnew = valueEC[:, 2]
CPVI = valueEC[:, 3]
pop90 = valueEC[:, 4]
pop00 = valueEC[:, 5]
pop10 = valueEC[:, 6] 
logpop00 = [math.log(item) for item in pop00]
logpop10 = [math.log(item) for item in pop10]

Alpha = []
AlphaSE = []
Beta = []
BetaSE = []
Gamma = []
GammaSE = []
Stderr = []
Nelect = []
pValues = []
allResiduals = numpy.array([])

NSIMS = 10000
T = 1
simAlpha = [[] for state in range(50)]
simBeta = [[] for state in range(50)]
simGamma = [[] for state in range(50)]
simVar = [[] for state in range(50)]

#Inverse chi squared
def ichi2(degFreedom, scale):
    shape = degFreedom/2
    return ((shape*scale)/random.gamma(shape))


#fit linear regression for given state
def Regression(state):
    ClearLean = []
    ClearYear = []
    ClearNation = []
    for year in range(14):
        if (not numpy.isnan(data[state][year])):
            ClearLean.append(data[state][year])
            ClearYear.append(year-14)
            ClearNation.append(USALean[year])
    ClearLean = numpy.array(ClearLean)
    ClearYear = numpy.array(ClearYear)
    n = numpy.size(ClearLean)
    Units = numpy.ones(n)
    Design = numpy.column_stack([Units, ClearNation, ClearYear])
    TDesign = numpy.transpose(Design)
    C = numpy.matmul(TDesign, Design)
    I = numpy.linalg.inv(C)
    coeff = numpy.matmul(I, TDesign).dot(ClearLean)
    residuals = ClearLean - Design.dot(coeff)
    p = stats.shapiro(residuals)[1]
    stderr = math.sqrt(1/(n-3) * numpy.dot(residuals, residuals))
    return (n, coeff, I, stderr, p, residuals)



#fitting linear regression and computing standard errors
#for the whole regression and coefficients
for state in range(50):
    n, coeff, I, stderr, p, residuals = Regression(state)
    allResiduals = numpy.append(allResiduals, residuals)
    pValues.append(p)
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
        
Coefficients = pandas.DataFrame({'state': States, 'quantity': Nelect, 
                    'alpha': Alpha, 'alphaSE': AlphaSE, 'beta': Beta, 
                    'betaSE': BetaSE, 'gamma': Gamma, 'gammaSE': GammaSE,
                    'stderr': Stderr, 'P': pValues})
    
print(Coefficients)

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
    print('Win = ', StateProb[50])
    return (StateProb)

PV2016 = math.log(48.2/46.1)
PV2012 = math.log(51.1/47.2)
PV2008 = math.log(52.9/45.7)
PV2004 = math.log(48.3/50.7)

output = pandas.DataFrame({
            'States': numpy.append(States, 'USA'),
            '2020 even': BayesEV(0, 0, EC),
            '2020 as 2016': BayesEV(PV2016, 0, EC),
            '2020 as 2008': BayesEV(PV2008, 0, EC),
            '2020 as 2004': BayesEV(PV2004, 0, EC),
            '2012 even': BayesEV(0, -4, EC),
            '2012 as 2012': BayesEV(PV2012, -4, EC),
            '2016 even': BayesEV(0, -2, EC),
            '2016 as 2016': BayesEV(PV2016, -2, EC)
            })
    
output.to_csv(r'stateProbUpdated.csv', index = None, header=True)

Coefficients.to_csv(r'updated.csv', index = None, header=True)

def importance(PV, Time, college):
    output = []
    for state in range(50):
        avg = Alpha[state] + Beta[state] * PV + Gamma[state] * Time
        stdev = Stderr[state]
        output.append(stats.norm.pdf(0, avg, stdev) * college[state])
    return (output)

output = pandas.DataFrame({
            'States': States,
            '2020 even': importance(0, 0, EC),
            '2020 as 2016': importance(PV2016, 0, EC),
            '2020 as 2008': importance(PV2008, 0, EC),
            '2020 as 2004': importance(PV2004, 0, EC),
           })
    
output.to_csv(r'importance.csv', index = None, header=True)
