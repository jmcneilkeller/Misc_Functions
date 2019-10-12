### CHI SQUARED TEST ###

import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Needs 2x2 np.array.
chi2 = np.array([factor_level_1,factor_level_2])
chi2_stat, p_val, dof, ex = chi2_contingency(chi2)
print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)

#interpret test-statistics
prob = .99
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, chi2_stat))
if abs(chi2_stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
    
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p_val))
if p_val <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
    
######################################################

### CALCULATE NUM SAMPLES NEEDED AT A GIVEN ALPHA AND POWER ###

from statsmodels.stats.power import TTestIndPower, TTestPower
import matplotlib.pyplot as plt

# Add Cohen's D for effect size calculation. 
sample_power = TTestPower() 
sample_power.solve_power(effect_size=effect_size, alpha=alpha, power=power) 

# Plot power curves for alternative formulations. "Nobs" = number of observations. Adjust params to suit. 
sample_power.plot_power(dep_var="nobs",
                          nobs = np.array(range(5,1500)),
                          effect_size=np.array([.005, .01, .02, .03]),
                          alpha=.01)
plt.show()

######################################################

### CREATE SAMPLE DISTRIBUTION OF SAMPLE MEANS ###

import scipy.stats as st
import pandas as pd
import numpy as np

def create_sample_distribution(data, dist_size=100, n=30):
    sample_dist = []
    while len(sample_dist) != dist_size:
        # With replacement. 
        sample = data.sample(n, replace=True)
        sample_dist.append(sum(sample) / len(sample))
    return sample_dist




