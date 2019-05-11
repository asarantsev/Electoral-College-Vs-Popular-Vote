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
UStates = numpy.append(States, 'USA')
data = [value[k][1:] for k in range(50)]
usa = value[50][1:]

#Election Years
Years = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 
         7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13]

#Old EC
EC = [9, 3, 11, 6, 55, 9, 7, 3, 29, 16, 4, 4, 20, 11, 6, 6, 
      8, 8, 4, 10, 11, 16, 10, 6, 10, 3, 5, 6, 4, 14, 5, 29, 
      15, 3, 18, 7, 7, 20, 4, 9, 3, 11, 38, 6, 3, 13, 12, 5, 10, 3]

ShiftedYears = [item - 13 for item in Years]
Years = ShiftedYears

Alpha = []
Beta = []
Gamma = []
Stderr = []
Nelect = []

#Fit regression for each state
for state in range(50):
    ClearLean = []
    ClearYear = []
    ClearNation = []
    for election in range(35):
        if (not numpy.isnan(data[state][election])):
            ClearLean.append(data[state][election])
            ClearYear.append(Years[election])
            ClearNation.append(usa[election])
    X = pandas.DataFrame({'USA': ClearNation, 'Year': ClearYear})
    X = api.add_constant(X)
    R = api.OLS(ClearLean, X).fit()
    residuals = ClearLean - R.predict(X)
    n = numpy.size(ClearLean)
    Nelect.append(n)
    stderr = math.sqrt(1/(n-3)*numpy.dot(residuals, residuals))
    Alpha.append(R.params[0])
    Beta.append(R.params[1])
    Gamma.append(R.params[2])
    Stderr.append(stderr)
    
Coefficients = pandas.DataFrame({'state': States, 'alpha': Alpha, 
                    'beta': Beta, 'gamma': Gamma, 'stderr': Stderr, 'EC': EC,
                    'quantity': Nelect})
    
print(Coefficients)#all coefficients

export_csv = Coefficients.to_csv (r'coefficients.csv', index = None, header=True)

TimeNow = 1

#simulate EC for multiple values of nationwide ln(D/R)
nationwideRange = numpy.arange(-0.2, 0.2, 0.01)
nationwideNumber = numpy.size(nationwideRange)
StateProb = numpy.empty([50, nationwideNumber])

for item in range(nationwideNumber):
    nationwide = nationwideRange[item]
    for state in range(50):
        z = Alpha[state] + nationwide * Beta[state] + TimeNow * Gamma[state]
        P = stats.norm.cdf(z/Stderr[state])
        StateProb[state][item] = P#probability of Dem winning 'state'
    
NationWin = numpy.zeros(nationwideNumber)
NSIMS = 40000
for sim in range(NSIMS):
    u = random.uniform(0, 1, 50)
    electoralVote = [3 for item in nationwideRange] 
    for item in range(nationwideNumber):
        for state in range(50):
            if (u[state] < StateProb[state][item]):
                electoralVote[item] = electoralVote[item] + EC[state]
        if (electoralVote[item] > 269):
            NationWin[item] = NationWin[item] + 1


print(NationWin/NSIMS)

pyplot.axvline(x = 0.00, color = 'r')
pyplot.axhline(y = 0.5, color = 'g')
pyplot.plot(nationwideRange, NationWin/NSIMS)
pyplot.show()

            
#simulate EC for 'nationwide' = ln(D/R) for USA
#Time  = election year
#college = which EC do we use?
def EV(nationwide, Time, college):
    outcome = []
    StateWin = []
    for state in range(50):
        z = Alpha[state] + nationwide * Beta[state] + Time * Gamma[state]
        P = stats.norm.cdf(z/Stderr[state])
        StateWin.append(P)
    for sim in range(NSIMS):
        electoralvote = 3
        for state in range(50):
            if (random.uniform(0,1) < StateWin[state]):
                electoralvote = electoralvote + college[state]
        outcome.append(electoralvote)
    print('time = ', Time)
    print('nationwide = ', nationwide)
    pyplot.axvline(x=269, color = 'r')
    pyplot.hist(outcome, bins = 30)
    pyplot.show()
    prob = numpy.count_nonzero([item > 269 for item in outcome])/NSIMS
    StateWin.append(prob)
    return (StateWin)

#plot data for chosen 'state'
def plotElections(state):
    ClearLean = []
    ClearYear = []
    for election in range(35):
        if (not numpy.isnan(data[state][election])):
            ClearLean.append(data[state][election])
            ClearYear.append(Years[election]*2+2018)
    pyplot.plot(ClearYear, ClearLean, 'go')
    pyplot.show()
    return(States[state])

#new EC
redistrict = [-1, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, -1, 
              0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 
              -2, 1, 0, -1, 0, 1, -1, -1, 0, 0, 0, 3, 0, 0, 0, 0, -1, 0, 0]

ECnew = EC + numpy.array(redistrict)

#scenarios six
scenarios = pandas.DataFrame({
            'States' : UStates, 
            '2020 even' : EV(0, 1, EC),
            '2020 as 2016': EV(math.log(48.2/46.1), 1, EC),
            '2020 as 2008': EV(math.log(52.9/45.7), 1, EC),
            '2020 as 2004': EV(math.log(48.3/50.7), 1, EC),
            '2024 even': EV(0, 3, ECnew),
            '2024 as 2016': EV(math.log(48.2/46.1), 3, ECnew)
            })

export_csv = scenarios.to_csv (r'scenarios.csv', index = None, header=True)

   
    