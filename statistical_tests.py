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


# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

import plotly.plotly as py
import plotly.graph_objs as go

@np.vectorize
def power_grid(x,y):
    power = TTestIndPower().solve_power(effect_size = x,
                                        nobs1 = y,
                                        alpha = 0.05)
    return power

X,Y = np.meshgrid(np.linspace(0.01, 1, 51),
                  np.linspace(10, 1000, 100))
X = X.T
Y = Y.T

Z = power_grid(X, Y) # power

data = [go.Surface(x = effect_size, y= Y, z = Z)]

layout = go.Layout(
    title='Power Analysis',
    scene = dict(
                    xaxis = dict(
                        title='effect size'),
                    yaxis = dict(
                        title='number of observations'),
                    zaxis = dict(
                        title='power'),)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='power_analysis')
