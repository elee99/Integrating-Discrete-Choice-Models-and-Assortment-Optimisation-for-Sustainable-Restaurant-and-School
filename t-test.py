# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:03:13 2023

@author: elfre
"""
import numpy as np
import math 

# t = x bar diff - 0 / s x bar
# calc differences between pairs
# calc mean of differences
# for sd : subtract mean from each diff and square, get mean of squared differences, take square root 

group1 = [0.171, 0.162, 0.164, 0.171, 0.172, 0.159]
group2 = [0.171, 0.162, 0.167, 0.170, 0.171, 0.159]

# 2.776 for n = 5, 2.57 for n = 6
def t_test(group1,group2,n,t_crit=2.02):
    diff = [group2[i] - group1[i] for i in range(len(group1))]
    
    x_bar = np.mean(diff)
    
    diff_mean = [diff[i] - x_bar for i in range(len(diff))]
    
    x_diff_mean = np.mean(diff_mean)
    
    s_diff = math.sqrt(np.mean(np.square(diff_mean)))
    
    s_diff = s_diff / math.sqrt(n)
    
    print("Mean of differences:", x_bar)
    print("Mean of centered differences:", x_diff_mean)
    print("Standard deviation of centered differences:", s_diff)
    
    t = x_bar / s_diff
    
    print("Test statistic:", t)
    
    if t < t_crit:
        print('Do not Reject H_0: The means are not significantly different')
    else:
        print('Reject H_0: Means are significantly different')
    
# A1
t_test(group1,group2,6)

# A2
t_test([0.207,0.205,0.205,0.206,0.177],[0.207,0.205,0.205,0.206,0.177],5,2.132)

# M1
t_test([0.166,0.166,0.169,0.167,0.167,0.165],[0.166,0.165,0.169,0.167,0.166,0.166],6,2.02)

# M2
t_test([0.185,0.213,0.227,0.206,0.169],[0.206,0.212,0.205,0.209,0.167],5,2.132)

# D1
t_test([0.463,0.0235,0.488,0.00000176,0.0256],[0.462,0.023,0.463,0.026,0.025],5,2.132)

# D2
t_test([0.3160,0.023,0.321,0.313,0.026],[0.316,0.023,0.322,0.312,0.026],5,2.132)

# S
t_test([0.215,0.211,0.0191,0.111,0.217,0.208,0.0199],[0.196,0.194,0.017,0.188,0.198,0.189,0.018],7,2.447)

# not significantly different for any category 

from scipy.stats import t

alpha = 0.05  # Example significance level
degrees_of_freedom = 7 - 1  # n is the number of paired observations

# Calculate the t-critical value
t_critical = t.ppf(1 - alpha/2, degrees_of_freedom)

# Print the t-critical value
print("T-Critical Value:", t_critical)

# testing results of MNL and MV-MNL optimisations

optimization1 =  [[624.9843622858813, 215.87900523910918, 308.2493117560954],
 [896.4573056252912, 380.7533820468844, 507.5841305979144],
 [652.6258718784964, 72.62279143952199, 557.6458590234961]]

optimization2 = [[409.83710801393727, 182.0097194205025, 219.05813313772236],
 [1096.0102954464555, 298.2634648558364, 664.0024095725746],
 [458.31932773109247, 67.52268907563027, 539.8588235294118]]

optimization3 = [[493.83552034711727, 345.39241615313017, 187.38468629004635],
 [996.2338005358732, 339.5084234513604, 585.7932700852444],
 [956.2787499793291, 85.764051029663, 778.1407526226545]]
###
optimization1_mvmnl = [[624.9843622858813, 215.87900523910918, 308.2493117560954],
 [896.4573056252912, 380.7533820468844, 507.5841305979144],
 [557.7165386261744, 63.787180931353774, 620.214603374862]]

# opt 2
optimization2_mvmnl = [[409.83710801393727, 182.0097194205025, 219.05813313772236],
 [1096.0102954464555, 298.2634648558364, 664.0024095725746],
 [458.31932773109247, 67.52268907563027, 539.8588235294118]]

# opt3
optimization3_mvmnl = [[624.9843622858813, 215.87900523910918, 308.2493117560954],
 [996.2338005358733, 339.5084234513604, 585.7932700852446],
 [538.608634293412, 75.05374302392713, 887.1098338572071]]

t_test([624.98,896.46,652.62,409.8,1096.0,458.32,493.84,996.23,956.27],[624.98,896.46,557.72,409.84,1096,458.32,624.98,996.23,538.61], 9, t.ppf(1 - 0.05/2, 9-1)) # cals, not signif diff

t_test([215.88,380.75,72.62,181,298.26,67.5,345.4,339.5,85.76],[215.87,380.75,85.76,182,298.26,67.5,215.87,339.51,75.1],9, t.ppf(1 - 0.05/2, 9-1))

t_test([308.25,507.58,557.64,219,664,539,187,585,778],[308,507,620,219,664,539,308,585,887],9, t.ppf(1 - 0.05/2, 9-1))

t.ppf(1 - 0.05/2, 9-1)
