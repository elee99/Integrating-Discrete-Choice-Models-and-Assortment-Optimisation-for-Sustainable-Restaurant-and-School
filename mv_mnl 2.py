# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:20:45 2023

@author: elfre
"""
# Following methodology in 'A Multivariate Model for Multinomial Choices' by Bel and Paap
# setting beta_k1 to zero and also alpha_21, 31
# in the paper P(Yi = 1 | ...) is proportional to 1? 'NPA'

# 15/08: have updated calculation of initial probabilities but need to update basket probs and think about optimisation
# as basket probs are technically what we need but if we followed the fact all probs of items in the offer set should be 1 
# then all probs would just equal 1? Unless we offered 2 baskets e.g. 2 s/2 m/2 d then each basket prob would be ~ 1/2? 
# TODO: just use conditional probs as they are?? Normalise so for each offer set they sum to 1

# change calc probs 2 to include separate cals/price/discount etc. params
import pandas as pd
import numpy as np
import biogeme.database as db
from biogeme.expressions import Variable
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
import itertools
from itertools import product
import glob
from scipy.special import softmax
from scipy.stats import gumbel_r
from itertools import combinations_with_replacement
from scipy.optimize import minimize

import pandas as pd 
import pulp
from scipy.optimize import linprog
import pulp
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpMaximize
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import biogeme.database as db
from biogeme.expressions import Variable
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
import itertools
from itertools import product
import glob
from scipy.special import softmax
from scipy.stats import gumbel_r
from itertools import combinations_with_replacement
from scipy.optimize import minimize
import math
from pulp import LpConstraintLE, LpConstraintGE, LpConstraintEQ, LpConstraint
from pulp import value 

# Get a list of all CSV files in the directory
csv_files = glob.glob('menu*.csv')

# Create an empty list to store dataframes
dfs = []

# Iterate over each CSV file and read it into a dataframe
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)
df = combined_df

df[['Appetizer 1', 'A1 Price', 'A1 Rating', 'A1 Calories', 'A1 Discount']] = df['Appertizer1'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Appetizer 2', 'A2 Price', 'A2 Rating', 'A2 Calories', 'A2 Discount']] = df['Appetizer2'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Main Course 1', 'MC1 Price', 'MC1 Rating', 'MC1 Calories', 'MC1 Discount']] = df['MainCourse1'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Main Course 2', 'MC2 Price', 'MC2 Rating', 'MC2 Calories', 'MC2 Discount']] = df['MainCourse2'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Soup', 'S Price', 'S Rating', 'S Calories', 'S Discount']] = df['Soup'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Dessert 1', 'D1 Price', 'D1 Rating', 'D1 Calories', 'D1 Discount']] = df['Dessert1'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Dessert 2', 'D2 Price', 'D2 Rating', 'D2 Calories', 'D2 Discount']] = df['Dessert2'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')

# Drop the original column
df.drop(columns=['Appertizer1'], inplace=True)
df.drop(columns=['Appetizer2'], inplace=True)
df.drop(columns=['MainCourse1'], inplace=True)
df.drop(columns=['MainCourse2'], inplace=True)
#df.drop(columns=['Soup'], inplace=True)
df.drop(columns=['Dessert1'], inplace=True)
df.drop(columns=['Dessert2'], inplace=True)

df['Gender'] = df['Gender'].replace({
    'Female':0,
    'Male':1})

df['Payment'] = df['Payment'].replace({
    'Cash':0,
    'Cashless':1})

df['MC1 Choice'] = df['Main Course 1'].replace({
    'PadKrapow': 1,
    'PadThai': 2,
    'HainaneseChickenRice': 3,
    'GreenChickenCurry': 4,
    'ShrimpStickyRice': 5,
    'ShrimpFriedRice':6
    # Add more mappings for other main course choices
})

df['A1 Choice'] = df['Appetizer 1'].replace({
    'FriedCalamari': 1,
    'FriedCalamari ':1,
    'EggRolls': 2,
    'ShrimpCake': 3,
    'FishCake': 4,
    'FishCake ':4,
    'FriedShrimpBall': 5,
    'HerbChicken':6
})

df['D1 Choice'] = df['Dessert 1'].replace({
    'CoconutCustard': 1,
    'KhanomMawKaeng': 2,
    'MangoStickyRice': 3,
    'LodChongThai': 4,
    'KanomKrok': 5
    })

#%%

# just using A1, MC1, D1 for now 

# basically want to get choice probs such as P(Yi = (1,2,3)) or P(Yi = (1,1,1))

# P(Yik = j | yil for l != k, Xi) = exp(Zik,j) / sum_l(exp(Zik,l)). Prob that individual i chooses j from category k, conditional prob
# Zik,j = alpha_k,j + Xi Beta_k,j + sum_l!=k psi_kl,jyil
# To get P(Yi | Xi) need to sum probabilities over all possible combinations of choices

# getting conditonal probs, e.g. P(Yi1 = 1 | yi2,yi3,Xi) denotes prob of inidividual i choosing item 1 from category 1, given the other choices
# I think (?) Xi is just attributes of item 1 in category 1 but not sure 
# then we have additional association params. For above example we would have psi_12,2yi2 and psi_12,2yi3 ? yi2 and yi3 realised values?
# one asc for each product in each category (5+5+6)
# association param x2 for each prob (?) --> 12 for A1. Assume A1 is category '1'

# need to consider setting some params to 0 / probs to 1 like in paper, but do this later

# Making function more efficient
def calc_probs_a1(df, initial_coeffs, name, length, cat, cat_oth, cat_oth1, cat_oth_length,cat_oth1_length):
    
    coeffs = {
        'asc_11': 0, # one asc for each product in A1 (6)
        'asc_12': initial_coeffs[1],
        'asc_13': initial_coeffs[2],
        'asc_14': initial_coeffs[3],
        'asc_15': initial_coeffs[4],
        'asc_16': initial_coeffs[5],
        'asc_21': 0, # set to 0
        'asc_22': initial_coeffs[7],
        'asc_23': initial_coeffs[8],
        'asc_24': initial_coeffs[9],
        'asc_25': initial_coeffs[10],
        'asc_26': initial_coeffs[11],
        'asc_31': 0, # set to zero 
        'asc_32': initial_coeffs[13],
        'asc_33': initial_coeffs[14],
        'asc_34': initial_coeffs[15],
        'asc_35': initial_coeffs[16],
        
        # changing to have separate beta param for each category AND alternative - paper specifies 'Beta_k,j'
        
        'b_price_11': 0,
        'b_calories_11': 0,
        'b_rating_11': 0,
        'b_discount_11': 0,
        'b_price_22': initial_coeffs[25],
        'b_calories_22': initial_coeffs[26],
        'b_rating_22': initial_coeffs[27],
        'b_discount_22': initial_coeffs[28],
        
        'b_price_12': initial_coeffs[29],
        'b_calories_12': initial_coeffs[30],
        'b_rating_12': initial_coeffs[31],
        'b_discount_12': initial_coeffs[32],
        'b_price_21': 0,
        'b_calories_21': 0,
        'b_rating_21': 0,
        'b_discount_21': 0,
        
        'b_price_13': initial_coeffs[33], # set to 0?
        'b_calories_13': initial_coeffs[34],
        'b_rating_13': initial_coeffs[109],
        'b_discount_13': initial_coeffs[110],
        'b_price_23': initial_coeffs[111],
        'b_calories_23': initial_coeffs[112],
        'b_rating_23': initial_coeffs[113],
        'b_discount_23': initial_coeffs[114],
        
        'b_price_14': initial_coeffs[115], # set to 0?
        'b_calories_14': initial_coeffs[116],
        'b_rating_14': initial_coeffs[117],
        'b_discount_14': initial_coeffs[118],
        'b_price_24': initial_coeffs[119],
        'b_calories_24': initial_coeffs[120],
        'b_rating_24': initial_coeffs[121],
        'b_discount_24': initial_coeffs[122],
        
        'b_price_15': initial_coeffs[123], # set to 0?
        'b_calories_15': initial_coeffs[124],
        'b_rating_15': initial_coeffs[125],
        'b_discount_15': initial_coeffs[126],
        'b_price_25': initial_coeffs[127],
        'b_calories_25': initial_coeffs[128],
        'b_rating_25': initial_coeffs[129],
        'b_discount_25': initial_coeffs[130],
        
        'b_price_16': initial_coeffs[131], # set to 0?
        'b_calories_16': initial_coeffs[132],
        'b_rating_16': initial_coeffs[133],
        'b_discount_16': initial_coeffs[134],
        'b_price_26': initial_coeffs[135],
        'b_calories_26': initial_coeffs[136],
        'b_rating_26': initial_coeffs[137],
        'b_discount_26': initial_coeffs[138],
        
        'b_price_31':0,
        'b_calories_31': 0,
        'b_rating_31': 0,
        'b_discount_31': 0,
       
        'b_price_32': initial_coeffs[139],
        'b_calories_32': initial_coeffs[140],
        'b_rating_32': initial_coeffs[141],
        'b_discount_32': initial_coeffs[142],
        'b_price_33': initial_coeffs[143], # set to 0?
        'b_calories_33': initial_coeffs[144],
        'b_rating_33': initial_coeffs[145],
        'b_discount_33': initial_coeffs[146],
        'b_price_34': initial_coeffs[147],
        'b_calories_34': initial_coeffs[148],
        'b_rating_34': initial_coeffs[149],
        'b_discount_34': initial_coeffs[150],
        'b_price_35': initial_coeffs[151],
        'b_calories_35': initial_coeffs[156],
        'b_rating_35': initial_coeffs[157],
        'b_discount_35': initial_coeffs[158],
 
        'assoc_12_11': 0,
        'assoc_12_12': 0, # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_12_13': 0, 
        'assoc_12_14': 0,
        'assoc_12_15': 0,
        'assoc_12_16': 0,
        
        'assoc_13_11': 0,
        'assoc_13_12': 0,
        'assoc_13_13': 0,
        'assoc_13_14': 0,
        'assoc_13_15': 0,
        'assoc_31_51': 0,
        'assoc_13_16': 0,
        
        'assoc_21_21': 0,
        'assoc_21_41': 0,
        'assoc_21_51': 0,
        'assoc_21_61': 0,
        
        'assoc_12_21': 0,
        'assoc_21_12': 0,
        'assoc_12_22': initial_coeffs[35],
        'assoc_21_22': initial_coeffs[35],
        'assoc_12_23': initial_coeffs[36], 
        'assoc_21_32': initial_coeffs[36],
        'assoc_12_24': initial_coeffs[37],
        'assoc_21_42': initial_coeffs[37],
        'assoc_12_25': initial_coeffs[38],
        'assoc_21_52': initial_coeffs[38],
        'assoc_12_26': initial_coeffs[39],
        'assoc_21_62': initial_coeffs[39],
        
        'assoc_13_21': 0,
        'assoc_13_22': initial_coeffs[40],
        'assoc_31_22': initial_coeffs[40],
        'assoc_13_23': initial_coeffs[41],
        'assoc_31_32': initial_coeffs[41],
        'assoc_13_24': initial_coeffs[42],
        'assoc_31_42': initial_coeffs[42],
        'assoc_13_25': initial_coeffs[43],
        'assoc_31_52': initial_coeffs[43],
        
        'assoc_12_31': 0,
        'assoc_21_31': 0,
        'assoc_12_32': initial_coeffs[44],
        'assoc_21_23': initial_coeffs[44],
        'assoc_12_33': initial_coeffs[45], 
        'assoc_21_33': initial_coeffs[45], 
        'assoc_12_34': initial_coeffs[46],
        'assoc_21_43': initial_coeffs[46],
        'assoc_12_35': initial_coeffs[47],
        'assoc_21_53': initial_coeffs[47],
        'assoc_12_36': initial_coeffs[48],
        'assoc_21_63': initial_coeffs[48],
                
        'assoc_13_31': 0,
        'assoc_13_32': initial_coeffs[49],
        'assoc_13_33': initial_coeffs[50],
        'assoc_13_34': initial_coeffs[51],
        'assoc_13_35': initial_coeffs[52],
        'assoc_31_13': 0,
        'assoc_31_23': initial_coeffs[49],
        'assoc_31_33': initial_coeffs[50],
        'assoc_31_43': initial_coeffs[51],
        'assoc_31_53': initial_coeffs[52],
        
        'assoc_12_41': 0,
        'assoc_12_42': initial_coeffs[53],
        'assoc_12_43': initial_coeffs[54], 
        'assoc_12_44': initial_coeffs[55],
        'assoc_12_45': initial_coeffs[56],
        'assoc_12_46': initial_coeffs[57],
        'assoc_21_14': 0,
        'assoc_21_24': initial_coeffs[53],
        'assoc_21_34': initial_coeffs[54], 
        'assoc_21_44': initial_coeffs[55],
        'assoc_21_54': initial_coeffs[56],
        'assoc_21_64': initial_coeffs[57],
        
        'assoc_13_41': 0,
        'assoc_13_42': initial_coeffs[56],
        'assoc_13_43': initial_coeffs[57],
        'assoc_13_44': initial_coeffs[58],
        'assoc_13_45': initial_coeffs[59],
        'assoc_31_14': 0,
        'assoc_31_24': initial_coeffs[56],
        'assoc_31_34': initial_coeffs[57],
        'assoc_31_44': initial_coeffs[58],
        'assoc_31_54': initial_coeffs[59],
        
        'assoc_12_51': 0,
        'assoc_12_52': initial_coeffs[60], 
        'assoc_12_53': initial_coeffs[61], 
        'assoc_12_54': initial_coeffs[62],
        'assoc_12_55': initial_coeffs[63],
        'assoc_12_56': initial_coeffs[64],
        'assoc_21_15': 0,
        'assoc_21_25': initial_coeffs[60], 
        'assoc_21_35': initial_coeffs[61], 
        'assoc_21_45': initial_coeffs[62],
        'assoc_21_55': initial_coeffs[63],
        'assoc_21_65': initial_coeffs[64],
        
        'assoc_13_51': 0,
        'assoc_13_52': initial_coeffs[65],
        'assoc_13_53': initial_coeffs[66],
        'assoc_13_54': initial_coeffs[67],
        'assoc_13_55': initial_coeffs[68],
        'assoc_31_15': 0,
        'assoc_31_25': initial_coeffs[65],
        'assoc_31_35': initial_coeffs[66],
        'assoc_31_45': initial_coeffs[67],
        'assoc_31_55': initial_coeffs[68],
        
        'assoc_12_61': 0,
        'assoc_12_62': initial_coeffs[69], # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_12_63': initial_coeffs[70], 
        'assoc_12_64': initial_coeffs[71],
        'assoc_12_65': initial_coeffs[72],
        'assoc_12_66': initial_coeffs[73],
        'assoc_21_16': 0,
        'assoc_21_26': initial_coeffs[69], # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_21_36': initial_coeffs[70], 
        'assoc_21_46': initial_coeffs[71],
        'assoc_21_56': initial_coeffs[72],
        'assoc_21_66': initial_coeffs[73],
        
        'assoc_13_61': 0,
        'assoc_13_62': initial_coeffs[74],
        'assoc_13_63': initial_coeffs[75],
        'assoc_13_64': initial_coeffs[76],
        'assoc_13_65': initial_coeffs[77],
        'assoc_31_16': 0,
        'assoc_31_26': initial_coeffs[74],
        'assoc_31_36': initial_coeffs[75],
        'assoc_31_46': initial_coeffs[76],
        'assoc_31_56': initial_coeffs[77],
        
        'assoc_21_11': 0,
        'assoc_21_12': 0, # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_21_13': 0, 
        'assoc_21_14': 0,
        'assoc_21_15': 0,
        'assoc_21_16': 0,
        
        'assoc_23_11': 0,
        'assoc_23_12': 0,
        'assoc_23_13': 0,
        'assoc_23_14': 0,
        'assoc_23_15': 0,
        'assoc_23_16': 0,
        
        'assoc_23_21': 0,
        'assoc_23_22': initial_coeffs[78],
        'assoc_23_23': initial_coeffs[79],
        'assoc_23_24': initial_coeffs[80],
        'assoc_23_25': initial_coeffs[81],
        'assoc_32_12': 0,
        'assoc_32_22': initial_coeffs[78],
        'assoc_32_32': initial_coeffs[79],
        'assoc_32_42': initial_coeffs[80],
        'assoc_32_52': initial_coeffs[81],
        
        'assoc_23_31': 0,
        'assoc_23_32': initial_coeffs[82],
        'assoc_23_33': initial_coeffs[83],
        'assoc_23_34': initial_coeffs[84],
        'assoc_23_35': initial_coeffs[85],
        'assoc_32_13': 0,
        'assoc_32_23': initial_coeffs[82],
        'assoc_32_33': initial_coeffs[83],
        'assoc_32_43': initial_coeffs[84],
        'assoc_32_53': initial_coeffs[85],
        
        'assoc_23_41': 0,
        'assoc_23_42': initial_coeffs[86],
        'assoc_23_43': initial_coeffs[87],
        'assoc_23_44': initial_coeffs[88],
        'assoc_23_45': initial_coeffs[89],
        'assoc_32_14': 0,
        'assoc_32_24': initial_coeffs[86],
        'assoc_32_34': initial_coeffs[87],
        'assoc_32_44': initial_coeffs[88],
        'assoc_32_54': initial_coeffs[89],
        
        'assoc_23_51': 0,
        'assoc_23_52': initial_coeffs[90],
        'assoc_23_53': initial_coeffs[91],
        'assoc_23_54': initial_coeffs[92],
        'assoc_23_55': initial_coeffs[93],
        'assoc_32_15': 0,
        'assoc_32_25': initial_coeffs[90],
        'assoc_32_35': initial_coeffs[91],
        'assoc_32_45': initial_coeffs[92],
        'assoc_32_55': initial_coeffs[93],
        
        'assoc_23_61': 0,
        'assoc_23_62': initial_coeffs[94],
        'assoc_23_63': initial_coeffs[95],
        'assoc_23_64': initial_coeffs[96],
        'assoc_23_65': initial_coeffs[97],
        'assoc_32_16': 0,
        'assoc_32_26': initial_coeffs[94],
        'assoc_32_36': initial_coeffs[95],
        'assoc_32_46': initial_coeffs[96],
        'assoc_32_56': initial_coeffs[97],
        
        'assoc_32_21': 0,
        'assoc_31_21': 0,
        'assoc_32_31': 0,
        'assoc_31_31': 0,
        'assoc_32_41': 0,
        'assoc_31_41': 0,
        'assoc_32_51': 0,
        'assoc_31_51': 0,
        
        'assoc_32_11': 0,
        'assoc_32_12': 0, # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_32_13': 0, 
        'assoc_32_14': 0,
        'assoc_32_15': 0,
        'assoc_32_16': 0,
        
        'assoc_31_11': 0,
        'assoc_31_12': 0,
        'assoc_31_13': 0,
        'assoc_31_14': 0,
        'assoc_31_15': 0,
        'assoc_31_16': 0,
        
        'assoc_32_61': initial_coeffs[98],
        'assoc_32_62': initial_coeffs[99], # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_32_63': initial_coeffs[100], 
        'assoc_32_64': initial_coeffs[101],
        'assoc_32_65': initial_coeffs[102],
        'assoc_32_66': initial_coeffs[103],
        
        'assoc_31_61': initial_coeffs[104],
        'assoc_31_62': initial_coeffs[105],
        'assoc_31_63': initial_coeffs[106],
        'assoc_31_64': initial_coeffs[107],
        'assoc_31_65': initial_coeffs[108],
        
        'assoc_13_36': 0,
        'assoc_13_46': 0,
        'assoc_13_56': 0,
        'assoc_13_66': 0,
        

    }
    
    utilities_dict = {k: [] for k in range(1, length + 1)}
    utilities = []
    
    df[f'{name} Calories'] = df[f'{name} Calories'].astype(float)
    df[f'{name} Price'] = df[f'{name} Price'].astype(float)
    df[f'{name} Rating'] = df[f'{name} Rating'].astype(float)
    df[f'{name} Discount'] = df[f'{name} Discount'].astype(float)

    for k in range(1,length+1):
        choice_df = df.loc[df[f'{name} Choice']==k]
        utilities.append(coeffs[f'asc_{cat}{k}'] + coeffs[f'b_price_{cat}{k}'] * choice_df[f'{name} Price'].iloc[0] + coeffs[f'b_calories_{cat}{k}'] * choice_df[f'{name} Calories'].iloc[0] + coeffs[f'b_discount_{cat}{k}'] * choice_df[f'{name} Discount'].iloc[0] + coeffs[f'b_rating_{cat}{k}'] * choice_df[f'{name} Rating'].iloc[0])
        choice_probs = np.exp(utilities)
        
        for j in range(1, cat_oth_length + 1):
            for i in range(1, cat_oth1_length + 1):
            # need to get separate probs for each k value
                utilities_dict[k].append(choice_probs + np.exp(coeffs[f'assoc_{cat}{cat_oth1}_{k}{i}'] + coeffs[f'assoc_{cat}{cat_oth}_{k}{j}']))

    tot_utility = 0
    for i in range(1,length+1):
        tot_utility += np.sum(np.sum(utilities_dict[i][j] for j in range(len(utilities_dict[1]))))

    probabilities = []
    for k in range(1,length+1):
        probabilities.append([np.sum(utilities_dict[k][j], axis=0) / tot_utility for j in range(len(utilities_dict[1]))])

    return probabilities
                
initial_coeffs = [0.01]*170
# should they all have 180? think so as its 5*(6*6) or 6*(5*6)
a1_cond_probs = calc_probs_a1(df, initial_coeffs, 'A1', 6, 1,2,3, 6,5) # 180 total probs 
m1_cond_probs = calc_probs_a1(df, initial_coeffs, 'MC1', 6, 2,1,3, 6,5) # 180 total probs
d1_cond_probs = calc_probs_a1(df, initial_coeffs, 'D1',5 ,3, 1,2, 6,6) # 180 total probs

sum(m1_cond_probs[0]) + sum(m1_cond_probs[1]) + sum(m1_cond_probs[2]) + sum(m1_cond_probs[3]) + sum(m1_cond_probs[4]) + sum(m1_cond_probs[5])

# 180 possible baskets 5x6x6
# e.g. P(Yi = (1,2,1)), P(Yi = (3,3,2)) etc...

def define_coeffs(initial_coeffs):
    coeffs = {
        'asc_11': 0, # one asc for each product in A1 (6)
        'asc_12': initial_coeffs[1],
        'asc_13': initial_coeffs[2],
        'asc_14': initial_coeffs[3],
        'asc_15': initial_coeffs[4],
        'asc_16': initial_coeffs[5],
        'asc_21': 0, # set to 0
        'asc_22': initial_coeffs[7],
        'asc_23': initial_coeffs[8],
        'asc_24': initial_coeffs[9],
        'asc_25': initial_coeffs[10],
        'asc_26': initial_coeffs[11],
        'asc_31': 0, # set to zero 
        'asc_32': initial_coeffs[13],
        'asc_33': initial_coeffs[14],
        'asc_34': initial_coeffs[15],
        'asc_35': initial_coeffs[16],
        
        # changing to have separate beta param for each category AND alternative - paper specifies 'Beta_k,j'
        
        'b_price_11': 0,
        'b_calories_11': 0,
        'b_rating_11': 0,
        'b_discount_11': 0,
        'b_price_22': initial_coeffs[25],
        'b_calories_22': initial_coeffs[26],
        'b_rating_22': initial_coeffs[27],
        'b_discount_22': initial_coeffs[28],
        
        'b_price_12': initial_coeffs[29],
        'b_calories_12': initial_coeffs[30],
        'b_rating_12': initial_coeffs[31],
        'b_discount_12': initial_coeffs[32],
        'b_price_21': 0,
        'b_calories_21': 0,
        'b_rating_21': 0,
        'b_discount_21': 0,
        
        'b_price_13': initial_coeffs[33], # set to 0?
        'b_calories_13': initial_coeffs[34],
        'b_rating_13': initial_coeffs[109],
        'b_discount_13': initial_coeffs[110],
        'b_price_23': initial_coeffs[111],
        'b_calories_23': initial_coeffs[112],
        'b_rating_23': initial_coeffs[113],
        'b_discount_23': initial_coeffs[114],
        
        'b_price_14': initial_coeffs[115], # set to 0?
        'b_calories_14': initial_coeffs[116],
        'b_rating_14': initial_coeffs[117],
        'b_discount_14': initial_coeffs[118],
        'b_price_24': initial_coeffs[119],
        'b_calories_24': initial_coeffs[120],
        'b_rating_24': initial_coeffs[121],
        'b_discount_24': initial_coeffs[122],
        
        'b_price_15': initial_coeffs[123], # set to 0?
        'b_calories_15': initial_coeffs[124],
        'b_rating_15': initial_coeffs[125],
        'b_discount_15': initial_coeffs[126],
        'b_price_25': initial_coeffs[127],
        'b_calories_25': initial_coeffs[128],
        'b_rating_25': initial_coeffs[129],
        'b_discount_25': initial_coeffs[130],
        
        'b_price_16': initial_coeffs[131], # set to 0?
        'b_calories_16': initial_coeffs[132],
        'b_rating_16': initial_coeffs[133],
        'b_discount_16': initial_coeffs[134],
        'b_price_26': initial_coeffs[135],
        'b_calories_26': initial_coeffs[136],
        'b_rating_26': initial_coeffs[137],
        'b_discount_26': initial_coeffs[138],
        
        'b_price_31':0,
        'b_calories_31': 0,
        'b_rating_31': 0,
        'b_discount_31': 0,
       
        'b_price_32': initial_coeffs[139],
        'b_calories_32': initial_coeffs[140],
        'b_rating_32': initial_coeffs[141],
        'b_discount_32': initial_coeffs[142],
        'b_price_33': initial_coeffs[143], # set to 0?
        'b_calories_33': initial_coeffs[144],
        'b_rating_33': initial_coeffs[145],
        'b_discount_33': initial_coeffs[146],
        'b_price_34': initial_coeffs[147],
        'b_calories_34': initial_coeffs[148],
        'b_rating_34': initial_coeffs[149],
        'b_discount_34': initial_coeffs[150],
        'b_price_35': initial_coeffs[151],
        'b_calories_35': initial_coeffs[156],
        'b_rating_35': initial_coeffs[157],
        'b_discount_35': initial_coeffs[158],
 
        'assoc_12_11': 0,
        'assoc_12_12': 0, # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_12_13': 0, 
        'assoc_12_14': 0,
        'assoc_12_15': 0,
        'assoc_12_16': 0,
        
        'assoc_13_11': 0,
        'assoc_13_12': 0,
        'assoc_13_13': 0,
        'assoc_13_14': 0,
        'assoc_13_15': 0,
        'assoc_31_51': 0,
        'assoc_13_16': 0,
        
        'assoc_21_21': 0,
        'assoc_21_41': 0,
        'assoc_21_51': 0,
        'assoc_21_61': 0,
        
        'assoc_12_21': 0,
        'assoc_21_12': 0,
        'assoc_12_22': initial_coeffs[35],
        'assoc_21_22': initial_coeffs[35],
        'assoc_12_23': initial_coeffs[36], 
        'assoc_21_32': initial_coeffs[36],
        'assoc_12_24': initial_coeffs[37],
        'assoc_21_42': initial_coeffs[37],
        'assoc_12_25': initial_coeffs[38],
        'assoc_21_52': initial_coeffs[38],
        'assoc_12_26': initial_coeffs[39],
        'assoc_21_62': initial_coeffs[39],
        
        'assoc_13_21': 0,
        'assoc_13_22': initial_coeffs[40],
        'assoc_31_22': initial_coeffs[40],
        'assoc_13_23': initial_coeffs[41],
        'assoc_31_32': initial_coeffs[41],
        'assoc_13_24': initial_coeffs[42],
        'assoc_31_42': initial_coeffs[42],
        'assoc_13_25': initial_coeffs[43],
        'assoc_31_52': initial_coeffs[43],
        
        'assoc_12_31': 0,
        'assoc_21_31': 0,
        'assoc_12_32': initial_coeffs[44],
        'assoc_21_23': initial_coeffs[44],
        'assoc_12_33': initial_coeffs[45], 
        'assoc_21_33': initial_coeffs[45], 
        'assoc_12_34': initial_coeffs[46],
        'assoc_21_43': initial_coeffs[46],
        'assoc_12_35': initial_coeffs[47],
        'assoc_21_53': initial_coeffs[47],
        'assoc_12_36': initial_coeffs[48],
        'assoc_21_63': initial_coeffs[48],
                
        'assoc_13_31': 0,
        'assoc_13_32': initial_coeffs[49],
        'assoc_13_33': initial_coeffs[50],
        'assoc_13_34': initial_coeffs[51],
        'assoc_13_35': initial_coeffs[52],
        'assoc_31_13': 0,
        'assoc_31_23': initial_coeffs[49],
        'assoc_31_33': initial_coeffs[50],
        'assoc_31_43': initial_coeffs[51],
        'assoc_31_53': initial_coeffs[52],
        
        'assoc_12_41': 0,
        'assoc_12_42': initial_coeffs[53],
        'assoc_12_43': initial_coeffs[54], 
        'assoc_12_44': initial_coeffs[55],
        'assoc_12_45': initial_coeffs[56],
        'assoc_12_46': initial_coeffs[57],
        'assoc_21_14': 0,
        'assoc_21_24': initial_coeffs[53],
        'assoc_21_34': initial_coeffs[54], 
        'assoc_21_44': initial_coeffs[55],
        'assoc_21_54': initial_coeffs[56],
        'assoc_21_64': initial_coeffs[57],
        
        'assoc_13_41': 0,
        'assoc_13_42': initial_coeffs[56],
        'assoc_13_43': initial_coeffs[57],
        'assoc_13_44': initial_coeffs[58],
        'assoc_13_45': initial_coeffs[59],
        'assoc_31_14': 0,
        'assoc_31_24': initial_coeffs[56],
        'assoc_31_34': initial_coeffs[57],
        'assoc_31_44': initial_coeffs[58],
        'assoc_31_54': initial_coeffs[59],
        
        'assoc_12_51': 0,
        'assoc_12_52': initial_coeffs[60], 
        'assoc_12_53': initial_coeffs[61], 
        'assoc_12_54': initial_coeffs[62],
        'assoc_12_55': initial_coeffs[63],
        'assoc_12_56': initial_coeffs[64],
        'assoc_21_15': 0,
        'assoc_21_25': initial_coeffs[60], 
        'assoc_21_35': initial_coeffs[61], 
        'assoc_21_45': initial_coeffs[62],
        'assoc_21_55': initial_coeffs[63],
        'assoc_21_65': initial_coeffs[64],
        
        'assoc_13_51': 0,
        'assoc_13_52': initial_coeffs[65],
        'assoc_13_53': initial_coeffs[66],
        'assoc_13_54': initial_coeffs[67],
        'assoc_13_55': initial_coeffs[68],
        'assoc_31_15': 0,
        'assoc_31_25': initial_coeffs[65],
        'assoc_31_35': initial_coeffs[66],
        'assoc_31_45': initial_coeffs[67],
        'assoc_31_55': initial_coeffs[68],
        
        'assoc_12_61': 0,
        'assoc_12_62': initial_coeffs[69], # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_12_63': initial_coeffs[70], 
        'assoc_12_64': initial_coeffs[71],
        'assoc_12_65': initial_coeffs[72],
        'assoc_12_66': initial_coeffs[73],
        'assoc_21_16': 0,
        'assoc_21_26': initial_coeffs[69], # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_21_36': initial_coeffs[70], 
        'assoc_21_46': initial_coeffs[71],
        'assoc_21_56': initial_coeffs[72],
        'assoc_21_66': initial_coeffs[73],
        
        'assoc_13_61': 0,
        'assoc_13_62': initial_coeffs[74],
        'assoc_13_63': initial_coeffs[75],
        'assoc_13_64': initial_coeffs[76],
        'assoc_13_65': initial_coeffs[77],
        'assoc_31_16': 0,
        'assoc_31_26': initial_coeffs[74],
        'assoc_31_36': initial_coeffs[75],
        'assoc_31_46': initial_coeffs[76],
        'assoc_31_56': initial_coeffs[77],
        
        'assoc_21_11': 0,
        'assoc_21_12': 0, # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_21_13': 0, 
        'assoc_21_14': 0,
        'assoc_21_15': 0,
        'assoc_21_16': 0,
        
        'assoc_23_11': 0,
        'assoc_23_12': 0,
        'assoc_23_13': 0,
        'assoc_23_14': 0,
        'assoc_23_15': 0,
        'assoc_23_16': 0,
        
        'assoc_23_21': 0,
        'assoc_23_22': initial_coeffs[78],
        'assoc_23_23': initial_coeffs[79],
        'assoc_23_24': initial_coeffs[80],
        'assoc_23_25': initial_coeffs[81],
        'assoc_32_12': 0,
        'assoc_32_22': initial_coeffs[78],
        'assoc_32_32': initial_coeffs[79],
        'assoc_32_42': initial_coeffs[80],
        'assoc_32_52': initial_coeffs[81],
        
        'assoc_23_31': 0,
        'assoc_23_32': initial_coeffs[82],
        'assoc_23_33': initial_coeffs[83],
        'assoc_23_34': initial_coeffs[84],
        'assoc_23_35': initial_coeffs[85],
        'assoc_32_13': 0,
        'assoc_32_23': initial_coeffs[82],
        'assoc_32_33': initial_coeffs[83],
        'assoc_32_43': initial_coeffs[84],
        'assoc_32_53': initial_coeffs[85],
        
        'assoc_23_41': 0,
        'assoc_23_42': initial_coeffs[86],
        'assoc_23_43': initial_coeffs[87],
        'assoc_23_44': initial_coeffs[88],
        'assoc_23_45': initial_coeffs[89],
        'assoc_32_14': 0,
        'assoc_32_24': initial_coeffs[86],
        'assoc_32_34': initial_coeffs[87],
        'assoc_32_44': initial_coeffs[88],
        'assoc_32_54': initial_coeffs[89],
        
        'assoc_23_51': 0,
        'assoc_23_52': initial_coeffs[90],
        'assoc_23_53': initial_coeffs[91],
        'assoc_23_54': initial_coeffs[92],
        'assoc_23_55': initial_coeffs[93],
        'assoc_32_15': 0,
        'assoc_32_25': initial_coeffs[90],
        'assoc_32_35': initial_coeffs[91],
        'assoc_32_45': initial_coeffs[92],
        'assoc_32_55': initial_coeffs[93],
        
        'assoc_23_61': 0,
        'assoc_23_62': initial_coeffs[94],
        'assoc_23_63': initial_coeffs[95],
        'assoc_23_64': initial_coeffs[96],
        'assoc_23_65': initial_coeffs[97],
        'assoc_32_16': 0,
        'assoc_32_26': initial_coeffs[94],
        'assoc_32_36': initial_coeffs[95],
        'assoc_32_46': initial_coeffs[96],
        'assoc_32_56': initial_coeffs[97],
        
        'assoc_32_21': 0,
        'assoc_31_21': 0,
        'assoc_32_31': 0,
        'assoc_31_31': 0,
        'assoc_32_41': 0,
        'assoc_31_41': 0,
        'assoc_32_51': 0,
        'assoc_31_51': 0,
        
        'assoc_32_11': 0,
        'assoc_32_12': 0, # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_32_13': 0, 
        'assoc_32_14': 0,
        'assoc_32_15': 0,
        'assoc_32_16': 0,
        
        'assoc_31_11': 0,
        'assoc_31_12': 0,
        'assoc_31_13': 0,
        'assoc_31_14': 0,
        'assoc_31_15': 0,
        'assoc_31_16': 0,
        
        'assoc_32_61': initial_coeffs[98],
        'assoc_32_62': initial_coeffs[99], # choosing 1 for cat 1, need all other combinations of what could be chosen for cat 2 and 3
        'assoc_32_63': initial_coeffs[100], 
        'assoc_32_64': initial_coeffs[101],
        'assoc_32_65': initial_coeffs[102],
        'assoc_32_66': initial_coeffs[103],
        
        'assoc_31_61': initial_coeffs[104],
        'assoc_31_62': initial_coeffs[105],
        'assoc_31_63': initial_coeffs[106],
        'assoc_31_64': initial_coeffs[107],
        'assoc_31_65': initial_coeffs[108],
        
        'assoc_13_36': 0,
        'assoc_13_46': 0,
        'assoc_13_56': 0,
        'assoc_13_66': 0,
        

    }
    return coeffs

# log likelihood is calculated with P(Yi = yi | Xi)
# but calc of joint probs can become burdensome if dimensions of logit are large
# as we have to sum many terms in the denominator 
# so can use composite likelihood, using conditional probs (avoids summation over complete outcome space)
# l^c(theta,y) = sum_i sum_k sum_j I[Yik = j] log P[Yik = j | yil for l != k, Xi]

# could sum up probs across each k and then times by number of people choosing that k?

def calc_log_lik(coeffs):
    cond_probs_a1 = calc_probs_a1(df, coeffs, 'A1', 6, 1, 2, 3, 6, 5)
    cond_probs_m1 = calc_probs_a1(df, coeffs, 'MC1', 6, 2, 1, 3, 6, 5)
    cond_probs_d1 = calc_probs_a1(df, coeffs, 'D1', 5, 3, 2, 1, 6, 6)
    
    all_probs = [cond_probs_a1, cond_probs_m1, cond_probs_d1]

    categories = ['A1','MC1','D1']
    lengths = [6,6,5]
    total_log_lik = 0
    
    log_lik = 0 
    for i in range(1,lengths[0]+1):
        log_lik += len(df.loc[df[f'{categories[0]} Choice'] == i]) * np.log(np.sum(all_probs[0][i-1]))
        log_lik += len(df.loc[df[f'{categories[1]} Choice'] == i]) * np.log(np.sum(all_probs[1][i-1]))
    for i in range(1,lengths[2]):
        log_lik += len(df.loc[df[f'{categories[2]} Choice'] == i]) * np.log(np.sum(all_probs[2][i-1]))
    total_log_lik += log_lik 
    
    return total_log_lik
        
calc_log_lik(initial_coeffs)

# Example usage:
initial_coeffs = [0.01] * 170
log_likelihood = calc_log_lik(initial_coeffs)
    
def objective(coeffs,df):
    # Call calc_probs with the dataframe and the current set of coefficients
    log_lik = calc_log_lik(coeffs)
    print(log_lik)
    # Negate the log-likelihood since we want to maximize it
    return -log_lik

objective(initial_coeffs, df)

###
import random

result = minimize(lambda theta: objective(theta,df), [0.05]*170, method='L-BFGS-B', options={'disp': True, 'maxiter': 15})
updated_coeffs = result.x 
final_log_lik = result.fun # 64973.67
res1_coeffs = define_coeffs(updated_coeffs)

cond_probs_a1 = calc_probs_a1(df, updated_coeffs, 'A1', 6, 1, 2, 3, 6, 5)
cond_probs_m1 = calc_probs_a1(df, updated_coeffs, 'MC1', 6, 2, 1, 3, 6, 5)
cond_probs_d1 = calc_probs_a1(df, updated_coeffs, 'D1', 5, 3, 2, 1, 6, 6)

result2 = minimize(lambda theta: objective(theta,df), [0.01]*170, method='L-BFGS-B', options={'disp': True, 'maxiter': 15})
final_log_lik_2 = result2.fun # 63829.388
res2_coeffs = define_coeffs(result2.x)

result3 = minimize(lambda theta: objective(theta,df), [0.015]*170, method='TNC', options={'disp': True, 'maxiter': 15})
final_log_lik_3 = result3.fun # 126449.004
res3_coeffs = define_coeffs(result3.x)

result4 = minimize(lambda theta: objective(theta,df), [0.005]*170, method='L-BFGS-B', options={'disp': True, 'maxiter': 15})
final_log_lik_4 = result4.fun # 61234.56

final_mvmnl_coeffs = define_coeffs(result4.x)  
coeffs_df = pd.DataFrame(list(final_mvmnl_coeffs.items()), columns=['Coeff', 'Value'])
coeffs_df.to_csv('mvmnl_coeffs.csv',index=True)

#%%
# calculating choice probs (dont actually need these)

# say the basket was (2,1,3), the choice prob would be exp(alpha_12 + alpha_21 + alpha_33 + beta... + assoc_12,21 + assoc_13,23 +...)
def calc_choice_probs2(coeffs,df):
    baskets = []
    choice_probs = []
    
    name1 = 'A1'
    name2 = 'MC1'
    name3 = 'D1'
    # Calculate the denominator (sum of all joint probabilities)
    total_prob = 0.0

    # Iterate through all combinations of choices from each category
    for i in range(6):  # Category 1 has 5 items
        for j in range(6):  # Category 2 has 5 items
            for k in range(5):  # Category 3 has 6 items
                choice1_df = df.loc[df[f'{name1} Choice'] == i+1]
                choice2_df = df.loc[df[f'{name2} Choice'] == j+1]
                choice3_df = df.loc[df[f'{name3} Choice'] == k+1]
                
                basket = (i+1, j+1, k+1)  # Adding 1 to indices to get basket values (1 to 5/6)

                basket_utility = []
                basket_utility.append(coeffs[f'asc_1{i+1}'] + coeffs[f'asc_2{j+1}'] + coeffs[f'asc_3{k+1}'] + \

                coeffs[f'b_price_1{i+1}'] * float(choice1_df[f'{name1} Price'].values.max()) + coeffs[f'b_calories_1{i+1}'] * float(choice1_df[f'{name1} Calories'].values.max()) + \
                coeffs[f'b_rating_1{i+1}'] * float(choice1_df[f'{name1} Rating'].values.max()) + coeffs[f'b_discount_1{i+1}'] * float(choice1_df[f'{name1} Discount'].values.max()) + \
                
                coeffs[f'b_price_2{j+1}'] * float(choice2_df[f'{name2} Price'].values.max()) + \
                coeffs[f'b_calories_2{j+1}'] * float(choice2_df[f'{name2} Calories'].values.max()) + \
                coeffs[f'b_rating_2{j+1}'] * float(choice2_df[f'{name2} Rating'].values.max()) + \
                coeffs[f'b_discount_2{j+1}'] * float(choice2_df[f'{name2} Discount'].values.max()) + \
                
                coeffs[f'b_price_3{k+1}'] * float(choice3_df[f'{name3} Price'].values.max()) + \
                coeffs[f'b_calories_3{k+1}'] * float(choice3_df[f'{name3} Calories'].values.max()) + \
                coeffs[f'b_rating_3{k+1}'] * float(choice3_df[f'{name3} Rating'].values.max()) + \
                coeffs[f'b_discount_3{k+1}'] * float(choice3_df[f'{name3} Discount'].values.max()) + \
                
                coeffs[f'assoc_12_{i+1}{j+1}'] + coeffs[f'assoc_13_{i+1}{k+1}'] + coeffs[f'assoc_21_{j+1}{i+1}'] + \
                coeffs[f'assoc_23_{j+1}{k+1}'] + \
                coeffs[f'assoc_32_{k+1}{j+1}'] + \
                coeffs[f'assoc_31_{k+1}{i+1}'] )
                    
                print(basket_utility)
                joint_prob = np.exp(basket_utility)
                total_prob += joint_prob
                baskets.append(basket)
                choice_probs.append(joint_prob)
    
    # Normalize the choice probabilities
    for i in range(len(choice_probs)):
        choice_probs[i] /= total_prob
            
    return choice_probs, baskets

# first utility comes out as 1 because all asc's / betas are set to 0 so exp(utility) = 1, but then other probs are so small
choice_probs, baskets = calc_choice_probs2(final_mvmnl_coeffs, df)

#%% 
# To be used in optimisation 

# cond probs a1 first list represents choosing item 1 from a1 then cond_probs_a1[0][0] is choosing (1 |1,1) then (1| 1,2) etc.
# need to get probs for each basket for example p(1|S=(1,1,1)) and then p(1|S)+... sums to 1

# a1: (app,main,dessert), m1: (main,app,dessert), d1: (dessert,app,main)
# a1: 1,1 1,2 1,3 ... 1,5 ..... 6,5

def get_norm_probs(a1_cond_probs, m1_cond_probs, d1_cond_probs):
                
    a1_cond_probs_tot = [] 
    m1_cond_probs_tot = []
    d1_cond_probs_tot = []   
    keys = [(i,j) for i in range(1,7) for j in range(1,6)]    
    keys2 = [(i,j) for i in range(1,7) for j in range(1,7)]                    
    for i in range(6):  # Category 1 has 5 items
        a1_cond_probs_dict = dict(zip(keys, a1_cond_probs[i]))
        m1_cond_probs_dict = dict(zip(keys, m1_cond_probs[i]))
        a1_cond_probs_tot.append(a1_cond_probs_dict)
        m1_cond_probs_tot.append(m1_cond_probs_dict)
        
    for i in range(5):
        d1_cond_probs_dict = dict(zip(keys2, d1_cond_probs[i]))
        d1_cond_probs_tot.append(d1_cond_probs_dict)


    basket_probabilities = {}
    for i in range(6):  # Category 1 has 5 items
        for j in range(6):  # Category 2 has 5 items
            for k in range(5):  # Category 3 has 6 items
                basket = (i+1, j+1, k+1) 
                m_prob = m1_cond_probs_tot[basket[1]-1][(basket[0],basket[2])]
                a_prob = a1_cond_probs_tot[basket[0]-1][(basket[1],basket[2])]
                d_prob = d1_cond_probs_tot[basket[2]-1][(basket[0],basket[1])]
                tot = m_prob + a_prob + d_prob
                
                m_prob_norm = m_prob / tot
                a_prob_norm = a_prob / tot
                d_prob_norm = d_prob / tot
                
                basket_probabilities[basket] = {
                    'm_prob': m_prob_norm,
                    'a_prob': a_prob_norm,
                    'd_prob': d_prob_norm
                }
            # so m_prob is p(m=j | a=k, d=i) etc.
    return basket_probabilities

# using result4 (starting values of 0.005) as this gave rise to lowest LL value (least negative)
a1_cond_probs = calc_probs_a1(df, result4.x, 'A1', 6, 1,2,3, 6,5) # 180 total probs 
m1_cond_probs = calc_probs_a1(df, result4.x, 'MC1', 6, 2,1,3, 6,5) # 180 total probs
d1_cond_probs = calc_probs_a1(df, result4.x, 'D1',5 ,3, 1,2, 6,6) # 180 total probs

basket_probs3 = get_norm_probs(a1_cond_probs, m1_cond_probs, d1_cond_probs)

#%%
# optimisation

soup_dishes = {
    'TomJuedMooSap': 1,
    'KaiPalo': 2,
    'OnionSoup':3,
    'TomYumKung':4,
    'TomKhaGai':5,
    'LengZab':6,
    'GlassNoodlesSoup':7}

dessert_1_dishes = {
    'CoconutCustard': 1,
    'KhanomMawKaeng': 2,
    'MangoStickyRice': 3,
    'LodChongThai': 4,
    'KanomKrok':5 }

dessert2_dishes = {
    'Pudding': 1,
    'VanillaIceCream': 2,
    'ApplePie': 3,
    'ChocolateCake': 4,
    'ChocolateIcecream': 5}

dessert_dishes = {
    'KhanomMawKaeng': 1,
    'MangoStickyRice': 2,
    'LodChongThai': 3,
    'KanomKrok':4,
    'Pudding': 5,
    'VanillaIceCream': 6,
    'ApplePie': 7,
    'ChocolateCake': 8,
    'ChocolateIcecream': 9}

appetizer1_dishes = {
    'FriedCalamari': 1,
    'FriedCalamari ':1,
    'EggRolls': 2,
    'ShrimpCake': 3,
    'FishCake': 4,
    'FishCake ':4,
    'FriedShrimpBall': 5,
    'HerbChicken':6
}

appetizer2_dishes = {
    'GrilledShrimp': 1,
    'CrispyCrab': 2,
    'MiangKham': 3,
    'ShrimpDumpling': 4,
    'CrispyCrab ': 2,
    'SaladRoll': 5}

main1_dishes = {
    'PadKrapow': 1,
    'PadThai': 2,
    'HainaneseChickenRice': 3,
    'GreenChickenCurry': 4,
    'ShrimpStickyRice': 5,
    'ShrimpFriedRice':6
    # Add more mappings for other main course choices
}

main2_dishes = {
    'ChickenPorridge':1,
    'KanomJeenNamYa':2,
    'ShrimpGlassNoodles':3,
    'AmericanFriedRice':4,
    'SausageFriedRice':5
    }

main_dishes = {
    'PadKrapow': 1,
    'PadThai': 2,
    'HainaneseChickenRice': 3,
    'GreenChickenCurry': 4,
    'ShrimpStickyRice': 5,
    'ShrimpFriedRice':6,
    'ChickenPorridge':7,
    'KanomJeenNamYa':8,
    'ShrimpGlassNoodles':9,
    'AmericanFriedRice':10,
    'SausageFriedRice':11
    }

def create_df(name,full_name,dishes_dict,):
    specific_df = df[[f'{name} Price', f'{name} Rating', f'{name} Calories', f'{name} Discount', f'{full_name}',
                   'Payment','Gender']]

    specific_df = specific_df.dropna()

    # add new columns with choice1 calories/price/rating etc?
    specific_df['Choice'] = specific_df[f'{full_name}'].replace(dishes_dict)

    specific_df['Gender'] = specific_df['Gender'].replace({
        'Female':0,
        'Male':1})

    specific_df['Payment'] = specific_df['Payment'].replace({
        'Cash':0,
        'Cashless':1})

    specific_df['Choice'] = specific_df['Choice'].astype(int)
 
    #specific_df.drop(columns=[f'{name} Price', f'{name} Rating', f'{name} Calories', f'{name} Discount', f'{full_name}'], inplace=True)
    
    return specific_df

dessert1_df = create_df('D1','Dessert 1',dessert_1_dishes)
dessert2_df = create_df('D2','Dessert 2',dessert2_dishes)
appetizer1_df = create_df('A1','Appetizer 1',appetizer1_dishes)
appetizer2_df = create_df('A2','Appetizer 2',appetizer2_dishes)
main1_df = create_df('MC1','Main Course 1',main1_dishes)
main2_df = create_df('MC2','Main Course 2',main2_dishes)
soup_df = create_df('S', 'Soup', soup_dishes)

final_scores = pd.read_csv('final_scores.csv')
all_ingredients_df = pd.read_csv('all_ingr.csv')
unique_ingr = all_ingredients_df['normalised ingredient'].unique()
most_common = pd.Series(all_ingredients_df['normalised ingredient']).value_counts().head(10)
nutrients = pd.read_csv('nutrients.csv')

soup_calories = [80,180,180,229,357,300]
soup_price = [40,50,55,90,60,80]
soup_score = [202.171,150.3,1087.99,3527.75,998.606,631.573]
soup_names = ['TomJuedMooSap','KaiPalo','OnionSoup','TomYumKung','TomKhaGai','GlassNoodlesSoup']
soup_fat = [float(nutrients.loc[nutrients['Dish names']==soup_names[i]]['Fat']) for i in range(len(soup_names))]
soup_protein = [float(nutrients.loc[nutrients['Dish names']==soup_names[i]]['Protein']) for i in range(len(soup_names))]
soup_carbs = [float(nutrients.loc[nutrients['Dish names']==soup_names[i]]['Carbs']) for i in range(len(soup_names))]


dessert1_calories = [540,244,270,215,240]
dessert1_price = [40,30,20,20]
dessert1_score = [183.244,69.12,34.4928,156.586]
dessert1_score = [110.86,183.244, 69.12, 34.4928, 156.586]
dessert1_names = ['Coconut Custard','KhanomMawKaeng','MangoStickyRice','LodChongThai','KanomKrok']
dessert1_fat = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Fat']) for i in range(len(dessert1_names))]
sum(dessert1_fat)/4
dessert1_fat = [7.375,8.872727272727273, 7.199999999999999, 6.37037037037037, 7.0588235294117645]
dessert1_protein = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Protein']) for i in range(len(dessert1_names))]
sum(dessert1_protein)/4
dessert1_protein = [2.962,4.4363636363636365,3.0857142857142854,0.7962962962962963,3.5294117647058822]
dessert1_carbs = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Carbs']) for i in range(len(dessert1_names))]
sum(dessert1_carbs)/4
dessert1_carbs = [42.917,4.4363636363636365,3.0857142857142854,0.7962962962962963,3.5294117647058822]


dessert2_calories = [120,330,296,424,335]
dessert2_price = [25,25,25,45,25]
dessert2_score = [196.258,380.859,390.752,191.838,346.64]
dessert2_names = ['Pudding','VanillaIceCream','ApplePie','ChocolateCake','ChocolateIcecream']
dessert2_fat = [float(nutrients.loc[nutrients['Dish names']==dessert2_names[i]]['Fat']) for i in range(len(dessert2_names))]
dessert2_protein = [float(nutrients.loc[nutrients['Dish names']==dessert2_names[i]]['Protein']) for i in range(len(dessert2_names))]
dessert2_carbs = [float(nutrients.loc[nutrients['Dish names']==dessert2_names[i]]['Carbs']) for i in range(len(dessert2_names))]


appetizer1_calories = [187,480,990,147,468,376]
appetizer1_price = [80,40,120,100,30,60]
appetizer1_score = [514.585,644.45,11627.6,175.107,1825.49,593.445]
appetizer1_names = ['FriedCalamari','EggRolls','ShrimpCake','FishCake','FriedShrimpBall','HerbChicken']
appetizer1_fat = [float(nutrients.loc[nutrients['Dish names']==appetizer1_names[i]]['Fat']) for i in range(len(appetizer1_names))]
appetizer1_protein = [float(nutrients.loc[nutrients['Dish names']==appetizer1_names[i]]['Protein']) for i in range(len(appetizer1_names))]
appetizer1_carbs = [float(nutrients.loc[nutrients['Dish names']==appetizer1_names[i]]['Carbs']) for i in range(len(appetizer1_names))]


appetizer2_calories = [125,544,280,300,182]
appetizer2_price = [80,30,90,60,60]
appetizer2_score = [2180.74,1440.48,283.801,512.902,1429.69]
appetizer2_names = ['GrilledShrimp','CrispyCrab','MiangKham','ShrimpDumpling','SaladRoll']
appetizer2_fat = [float(nutrients.loc[nutrients['Dish names']==appetizer2_names[i]]['Fat']) for i in range(len(appetizer2_names))]
appetizer2_protein = [float(nutrients.loc[nutrients['Dish names']==appetizer2_names[i]]['Protein']) for i in range(len(appetizer2_names))]
appetizer2_carbs = [float(nutrients.loc[nutrients['Dish names']==appetizer2_names[i]]['Carbs']) for i in range(len(appetizer2_names))]


main1_calories = [372,486,597,240,477,289]
main1_price = [35,90,80,60,60,80]
main1_score = [815.865,295.153,1116.75,337.024,3956.84,1098.2]
main1_names = ['PadKrapow','PadThai','HainaneseChickenRice','GreenChickenCurry','ShrimpStickyRice','ShrimpFriedRice']
main1_fat = [float(nutrients.loc[nutrients['Dish names']==main1_names[i]]['Fat']) for i in range(len(main1_names))]
main1_protein = [float(nutrients.loc[nutrients['Dish names']==main1_names[i]]['Protein']) for i in range(len(main1_names))]
main1_carbs = [float(nutrients.loc[nutrients['Dish names']==main1_names[i]]['Carbs']) for i in range(len(main1_names))]

# American fried rice has no info on fat/protein/carbs, impute average value?
main2_calories = [228,81,300,790,610]
main2_price = [50,45,90,100,70]
main2_score = [887.141,101.133,1171.55,1636.32,967.658]
main2_names = ['ChickenPorridge','KanomJeenNamYa','ShrimpGlassNoodles','AmericanFriedRice','SausageFriedRice']
main2_fat = [float(nutrients.loc[nutrients['Dish names']==main2_names[i]]['Fat']) for i in range(len(main2_names))]
main2_protein = [float(nutrients.loc[nutrients['Dish names']==main2_names[i]]['Protein']) for i in range(len(main2_names))]
main2_carbs = [float(nutrients.loc[nutrients['Dish names']==main2_names[i]]['Carbs']) for i in range(len(main2_names))]
def impute_nan_with_average(data_list):
    # Step 1: Create a new list without 'nan' values
    non_nan_values = [val for val in data_list if not np.isnan(val)]
    
    # Step 2: Calculate the average of non-'nan' values
    average = np.mean(non_nan_values)
    
    # Step 3: Replace 'nan' with the average
    imputed_list = [val if not np.isnan(val) else average for val in data_list]
    
    return imputed_list

main2_fat = impute_nan_with_average(main2_fat)
main2_protein = impute_nan_with_average(main2_protein)
main2_carbs = impute_nan_with_average(main2_carbs)

def get_cholesterol(names):
    chol_list = [float(nutrients.loc[nutrients['Dish names']==names[i]]['Chol (mg)']) for i in range(len(names))]
    return chol_list

appetizer1_chol = get_cholesterol(appetizer1_names)
appetizer2_chol = get_cholesterol(appetizer2_names)
main1_chol = get_cholesterol(main1_names)
main2_chol = get_cholesterol(main2_names)
dessert1_chol = get_cholesterol(dessert1_names)
dessert2_chol = get_cholesterol(dessert2_names)

def get_salt(names):
    salt_list = [float(nutrients.loc[nutrients['Dish names']==names[i]]['Salt (mg)']) for i in range(len(names))]
    return salt_list

appetizer1_salt = get_salt(appetizer1_names)
appetizer2_salt = get_salt(appetizer2_names)
main1_salt = get_salt(main1_names)
main2_salt = get_salt(main2_names)
dessert1_salt = get_salt(dessert1_names)
dessert2_salt = get_salt(dessert2_names)
dessert1_salt = [79.52, 54.345454545454544, 57.08571428571428, 104.3148148148148, 102.35294117647058]
soup_salt = get_salt(soup_names)
main2_salt = impute_nan_with_average(main2_salt)

def get_sat_fat(names):
    fat_list = [float(nutrients.loc[nutrients['Dish names']==names[i]]['Sat_fat']) for i in range(len(names))]
    return fat_list

appetizer1_sat = get_sat_fat(appetizer1_names)
appetizer2_sat = get_sat_fat(appetizer2_names)
main1_sat = get_sat_fat(main1_names)
main2_sat = get_sat_fat(main2_names)
dessert1_sat = get_sat_fat(dessert1_names)
dessert2_sat = get_sat_fat(dessert2_names)
main2_sat = impute_nan_with_average(main2_sat)

total_a_scores = appetizer1_score + appetizer2_score
total_a_calories = appetizer1_calories + appetizer2_calories
total_a_carbs = appetizer1_carbs + appetizer2_carbs
total_a_protein = appetizer1_protein + appetizer2_protein
total_a_fat = appetizer1_fat + appetizer2_fat
total_a_salt = appetizer1_salt + appetizer2_salt

# mains
# need to remove one with nan value
total_m_scores = main1_score + main2_score
total_m_calories = main1_calories + main2_calories
total_m_carbs = main1_carbs + main2_carbs
total_m_protein = main1_protein + main2_protein
total_m_fat = main1_fat + main2_fat
total_m_salt = main1_salt + main2_salt

# desserts
total_d_scores = dessert1_score + dessert2_score
total_d_calories = dessert1_calories + dessert2_calories
total_d_carbs = dessert1_carbs + dessert2_carbs
total_d_protein = dessert1_protein + dessert2_protein
total_d_fat = dessert1_fat + dessert2_fat
total_d_salt = dessert1_salt + dessert2_salt

total_s_scores = impute_nan_with_average(soup_score)
total_s_calories = impute_nan_with_average(soup_calories)
total_s_carbs = impute_nan_with_average(soup_carbs)
total_s_protein = impute_nan_with_average(soup_protein)
total_s_fat = impute_nan_with_average(soup_fat)
total_s_salt = impute_nan_with_average(soup_salt)

def get_dict(names,nutrient):
    _dict = {}
    for name in names:
        value = float(nutrients.loc[nutrients['Dish names'] == name][nutrient])
        _dict[name] = value
    return _dict

def update_dict(nutrient):
    protein_dict = {}
    protein_dict.update(get_dict(appetizer1_names,nutrient))
    protein_dict.update(get_dict(appetizer2_names,nutrient))
    protein_dict.update(get_dict(main1_names,nutrient))
    protein_dict.update(get_dict(main2_names,nutrient))
    protein_dict.update(get_dict(dessert1_names,nutrient))
    protein_dict.update(get_dict(dessert2_names,nutrient))
    protein_dict.update(get_dict(soup_names,nutrient))
    return protein_dict

fat_dict = update_dict('Fat')
carbs_dict = update_dict('Carbs')
salt_dict = update_dict('Salt (mg)')
protein_dict = update_dict('Protein')
sat_dict = update_dict('Sat_fat')
calories_dict = update_dict('Calories')

scores_dict= dict(zip(final_scores['dish'], final_scores['total_score']))


#%%

def opt_mvmnl(basket_probs3,max_fat = 200,max_carbs = 1000,max_protein = 400,max_salt = 2000,max_cals = 2000,min_fat = 0,min_carbs = 0,min_protein = 0):
    problem = LpProblem("Basket_Selection",LpMinimize)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    categories = {
        "App": 6,
        "Main": 6,
        "Dessert": 5
    }
    
    # Create variables with the desired names
    item_vars = LpVariable.dicts(
        "Item",
        ((day, cat, item) for day in days for cat, num_items in categories.items() for item in range(1, num_items + 1)),
        cat='Binary'
    )
    
    arrival_rate = 130
    
    objective_values = []
    fat_constraints = []
    protein_constraints = []
    carb_constraints = []
    salt_constraints = []
    calorie_constraints = []
        
    for basket in basket_probs3.keys():
        for day in days:
            # Initialize the objective value for this offer set and day to zero
            objective_value = 0
            total_fat = 0
            total_protein = 0
            total_carbs = 0
            total_salt = 0
            total_calories = 0
            # Loop through each item in the offer set
            
            objective_value += arrival_rate * item_vars[(day,'App',basket[0])] * appetizer1_score[basket[0]-1] * basket_probs3[basket]['a_prob']
            objective_value += arrival_rate * item_vars[(day,'Main',basket[1])] * main1_score[basket[1]-1] * basket_probs3[basket]['m_prob']
            objective_value += arrival_rate * item_vars[(day,'Dessert',basket[2])] * dessert1_score[basket[2]-1] * basket_probs3[basket]['d_prob']
            
            total_fat += item_vars[(day,'App',basket[0])] * appetizer1_fat[basket[0]-1]
            total_fat += item_vars[(day,'Main',basket[1])] * main1_fat[basket[1]-1]
            total_fat += item_vars[(day,'Dessert',basket[2])] * dessert1_fat[basket[2]-1]
            
            total_carbs += item_vars[(day,'App',basket[0])] * appetizer1_carbs[basket[0]-1]
            total_carbs += item_vars[(day,'Main',basket[1])] * main1_carbs[basket[1]-1]
            total_carbs += item_vars[(day,'Dessert',basket[2])] * dessert1_carbs[basket[2]-1]
            
            total_protein += item_vars[(day,'App',basket[0])] * appetizer1_protein[basket[0]-1]
            total_protein += item_vars[(day,'Main',basket[1])] * main1_protein[basket[1]-1]
            total_protein += item_vars[(day,'Dessert',basket[2])] * dessert1_protein[basket[2]-1]
            
            total_salt += item_vars[(day,'App',basket[0])] * appetizer1_salt[basket[0]-1]
            total_salt += item_vars[(day,'Main',basket[1])] * main1_salt[basket[1]-1]
            total_salt += item_vars[(day,'Dessert',basket[2])] * dessert1_salt[basket[2]-1]
            
            total_calories += item_vars[(day,'App',basket[0])] * appetizer1_calories[basket[0]-1]
            total_calories += item_vars[(day,'Main',basket[1])] * main1_calories[basket[1]-1]
            total_calories += item_vars[(day,'Dessert',basket[2])] * dessert1_calories[basket[2]-1]
            
            objective_values.append(objective_value)
    
            fat_constraints.append(LpConstraint(total_fat, sense=LpConstraintLE, rhs=max_fat))
            fat_constraints.append(LpConstraint(total_fat,sense=LpConstraintGE,rhs=min_fat))
            
            protein_constraints.append(LpConstraint(total_protein,sense=LpConstraintLE,rhs=max_protein))
            protein_constraints.append(LpConstraint(total_protein,sense=LpConstraintGE,rhs=min_protein))
            
            carb_constraints.append(LpConstraint(total_carbs, sense=LpConstraintLE, rhs=max_carbs))
            carb_constraints.append(LpConstraint(total_carbs, sense=LpConstraintGE, rhs=min_carbs))
    
            salt_constraints.append(LpConstraint(total_salt, sense=LpConstraintLE, rhs=max_salt))
    
            calorie_constraints.append(LpConstraint(total_calories, sense=LpConstraintLE, rhs=max_cals))
            
    problem += lpSum(objective_values)
    
    # Add constraints for each day to select only 1 item from each category
    for day in days:
        for cat, num_items in categories.items():
            problem += lpSum(item_vars[(day, cat, item)] for item in range(1, num_items + 1)) == 1
    
    #item_occurrences = LpVariable.dicts("Item_Occurrence", [(day, cat, item) for day in days for cat,num_items in categories.items() for item in range(1,num_items+1)], cat='Binary')
    max_item_occurrences = 2
    
    # Constraint to ensure that each item is selected at most 'max_item_occurrences' times
    for cat, items in categories.items():
        for item in range(1,items+1):
            problem += lpSum(item_vars[(day, cat, item)] for day in days) <= max_item_occurrences

    # convsecutive days
    for item1 in range(1, 6+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'App', item1)] + item_vars[(days[i + 1],'App', item1)] <= 1        
    for item2 in range(1, 6+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'Main', item2)] + item_vars[(days[i + 1],'Main', item2)] <= 1        
    for item3 in range(1, 5+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'Dessert', item3)] + item_vars[(days[i + 1],'Dessert', item3)] <= 1        
    
    for fat_constraint in fat_constraints:
        problem += fat_constraint
        
    for p_constraint in protein_constraints:
        problem += p_constraint
        
    for c_constraint in carb_constraints:
        problem += c_constraint
    
    for s_constraint in salt_constraints:
        problem += s_constraint
        
    for cal_constraint in calorie_constraints:
        problem += cal_constraint
        
    problem.solve()
    """
    variable_values = {}
    for var in problem.variables():
        variable_values[var.name] = var.varValue
    
    # Printing the variable values
    for var_name, var_value in variable_values.items():
        print(f"{var_name}: {var_value}")
    """ 
    return [item for item in item_vars if item_vars[item].value() == 1],item_vars,problem

# choosing one basket per day (one main, starter, dessert)
# item can occur max twice over period
# not on consecutive days
# according to nutritional constraints

# lowest values that give feasible solution
# is it max cals etc for each item or whole basket? Cals need to be at least 950 yet protein can be 50...
items,item_vars,probem_ = opt_mvmnl(basket_probs3,max_fat=60,max_carbs=90,max_protein=50,max_salt=1300,max_cals=950)

dishes_dict = {
    'Pudding': ('Dessert', 1),
    'VanillaIceCream': ('Dessert', 2),
    'ApplePie': ('Dessert', 3),
    'ChocolateCake': ('Dessert', 4),
    'ChocolateIcecream': ('Dessert', 5),
    'FriedCalamari': ('App', 1),
    'EggRolls': ('App', 2),
    'ShrimpCake': ('App', 3),
    'FishCake': ('App', 4),
    'FriedShrimpBall': ('App', 5),
    'HerbChicken': ('App', 6),
    'PadKrapow': ('Main', 1),
    'PadThai': ('Main', 2),
    'HainaneseChickenRice': ('Main', 3),
    'GreenChickenCurry': ('Main', 4),
    'ShrimpStickyRice': ('Main', 5),
    'ShrimpFriedRice': ('Main', 6)
    # Add more mappings for other main course choices
}

# need to compare to MNL but that is by individual categories so need to find a way of calculating score/calories etc. 
# so that they are over a basket 

scores_opt_s1 = [15988.18,9882.72,1940.71,7934.20] # scores for each category over whole week (10 starters, mains etc)
(scores_opt_s1[0] + scores_opt_s1[1] + scores_opt_s1[2])/30 # avg score of 745 per item
11175/15 # avg score of 745 per item (mv mnl)

# Create dictionaries to store the total nutritional values per category
total_calories_per_cat = {cat: 0 for cat in categories.keys()}
total_fat_per_cat = {cat: 0 for cat in categories.keys()}
total_protein_per_cat = {cat: 0 for cat in categories.keys()}
total_salt_per_cat = {cat: 0 for cat in categories.keys()}
total_carbs_per_cat = {cat: 0 for cat in categories.keys()}
total_price_per_cat = {cat: 0 for cat in categories.keys()}
total_score_per_cat = {cat: 0 for cat in categories.keys()}

# Iterate over the days and categories
for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
    for cat in categories.keys():
        print(f"Dishes selected for {cat} {day}:")
        for dish_name, (dish_cat, dish_item) in dishes_dict.items():
            if cat == dish_cat:
                if value(item_vars[(day, dish_cat, dish_item)]) == 1:
                    print(f"{dish_name}")
                    
                    # Retrieve the nutritional information for the dish from the dictionaries
                    dish_calories = calories_dict[dish_name]
                    dish_fat = fat_dict[dish_name]
                    dish_carbs = carbs_dict[dish_name]
                    dish_protein = protein_dict[dish_name]
                    dish_salt = salt_dict[dish_name]
                    dish_score = scores_dict[dish_name]
                    
                    # Update the total nutritional values for the category
                    total_calories_per_cat[cat] += dish_calories
                    total_fat_per_cat[cat] += dish_fat
                    total_protein_per_cat[cat] += dish_protein
                    total_salt_per_cat[cat] += dish_salt
                    total_carbs_per_cat[cat] += dish_carbs
                    total_score_per_cat[cat] += dish_score

# Print the total nutritional values per category for the entire week
print("Total Nutritional Values per Category for the Week:")
for cat in categories.keys():
    print(f"{cat}:")
    print(f"  Total Calories: {total_calories_per_cat[cat]}")
    print(f"  Total Fat: {total_fat_per_cat[cat]}g")
    print(f"  Total Protein: {total_protein_per_cat[cat]}g")
    print(f"  Total Salt: {total_salt_per_cat[cat]}g")
    print(f"  Total Carbs: {total_carbs_per_cat[cat]}g")
    print(f"  Total Score: {total_score_per_cat[cat]}")
    print()

def get_menu(dishes_dict, problem, item_vars, items):
    
    # Rest of the code remains the same
    total_calories_week = 0
    total_fat_week = 0
    total_protein_week = 0
    total_salt_week = 0
    total_carbs_week = 0
    total_price_week = 0
    total_score_week = 0
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        print(f"Day: {day}")
        
        # Initialize variables to store the total calories for the day
        total_calories_1 = 0
        total_fat_1 = 0
        total_protein_1 = 0
        total_salt_1 = 0
        total_carbs_1 = 0
        total_price_1 = 0
        total_score_1 = 0
        
        # Retrieve the dish name and quantity for each category in the given day
        for variable in problem.variables():
            # Check if the variable is selected (has a value greater than 0) and belongs to the given day
            if variable.varValue > 0 and day in variable.name:
                dish_name = variable_to_dish[variable.name]
        
                # Retrieve the nutritional information for the dish from the dictionaries
                dish_calories = calories_dict[dish_name]
                dish_fat = fat_dict[dish_name]
                dish_carbs = carbs_dict[dish_name]
                dish_protein = protein_dict[dish_name]
                dish_salt = salt_dict[dish_name]
                dish_score = scores_dict[dish_name]
                
                if not math.isnan(dish_calories):
                    total_calories_1 += dish_calories
                if not math.isnan(dish_fat):
                    total_fat_1 += dish_fat 
                if not math.isnan(dish_protein):
                    total_protein_1 += dish_protein 
                if not math.isnan(dish_carbs):
                    total_carbs_1 += dish_carbs 
                if not math.isnan(dish_salt):
                    total_salt_1 += dish_salt 
        
                if not math.isnan(dish_score):
                    total_score_1 += dish_score
        
                # Print the dish name, quantity, and nutritional information
                print(f"{dish_name}")
                print(f"  Calories: {dish_calories}")
                print(f"  Fat: {dish_fat}g")
                print(f"  Carbs: {dish_carbs}g")
                print(f"  Protein: {dish_protein}g")
                print(f"  Salt: {dish_salt}g")
        
        # Add the daily totals to the weekly totals
        total_calories_week += total_calories_1
        total_fat_week += total_fat_1
        total_protein_week += total_protein_1
        total_salt_week += total_salt_1
        total_carbs_week += total_carbs_1
        total_price_week += total_price_1
        total_score_week += total_score_1
        
        # Print the total calories for the day
        print(f"Total Calories for {day}: {total_calories_1}")
        print(f"Total Fat for {day}: {total_fat_1}")
        print(f"Total Protein for {day}: {total_protein_1}")
        print(f"Total Carbs for {day}: {total_carbs_1}")
        print(f"Total Salt for {day}: {total_salt_1}")
        print()

    return total_calories_week, total_fat_week, total_protein_week, total_salt_week, total_carbs_week, total_price_week, total_score_week

get_menu(dishes_dict, probem_, item_vars, items)

#%%

plt.rcParams['font.sans-serif'] = "Georgia"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

total_fat_appetizers = 69.44*9
total_protein_appetizers = 53.97*4
total_carbs_appetizers = 77.1*4

total_fat_mains = 110.69*9
total_protein_mains = 84.88*4
total_carbs_mains = 146.45*4

total_fat_desserts = 88.15*9
total_protein_desserts = 16.03*4
total_carbs_desserts = 251.86*4


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Proportions of Total Calories by Fat, Protein, and Carbs')
labels = ['Fat', 'Protein', 'Carbs']
colors = ['purple', 'green', 'blue']
color_palette = sns.color_palette("Set3")  # Choose a different palette

# Appetizers
axs[0, 0].pie([total_fat_appetizers, total_protein_appetizers, total_carbs_appetizers], labels=labels, colors=color_palette,
        autopct='%1.1f%%', shadow=True, startangle=140)
axs[0, 0].set_title('Appetizers')

# Mains
axs[0, 1].pie([total_fat_mains, total_protein_mains, total_carbs_mains], labels=labels, colors=color_palette,
        autopct='%1.1f%%', shadow=True, startangle=140)
axs[0, 1].set_title('Mains')

# Desserts
axs[1, 0].pie([total_fat_desserts, total_protein_desserts, total_carbs_desserts], labels=labels, colors=color_palette,
        autopct='%1.1f%%', shadow=True, startangle=140)
axs[1, 0].set_title('Desserts')

axs[1, 1].axis('off')

plt.tight_layout()
plt.savefig("pie_chart_mvmnl.pdf", bbox_inches='tight', format='pdf')
plt.show()

###
# comparing avg scores with mnl
avg_score_a = [2023.83/5]
avg_score_m = [2080.22/5]
avg_score_d = [1511.82]
scores_mvmnl = [2023.83/5,2080.22/5,1511.82/5]

scores_mnl = [5832.16/10,4872.63/10,485.9/5]


# Bar plot comparing total S score when optimising for price vs S score
fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width
bar_width = 0.4

categories = ['Appetizer','Main','Dessert']
# Calculate the positions for the bars
bar_positions_price = np.arange(len(categories))
bar_positions_new = bar_positions_price + bar_width

# Create the bar plots for both cases
bar1 = ax.barh(bar_positions_price, scores_mvmnl, bar_width, label='MVMNL Optimization', color='skyblue')
bar2 = ax.barh(bar_positions_new, scores_mnl, bar_width, label='MNL Optimization', color='lightgreen')

# Set labels and title
ax.set_xlabel('Average Menu Carbon Footprint Score')
ax.set_ylabel('Categories')
ax.set_title('Average Score by Category')
ax.set_yticks(bar_positions_price + bar_width / 2)
ax.set_yticklabels(categories)
ax.legend()

plt.savefig("score_comparison_mvmnl.pdf", bbox_inches='tight', format='pdf')
plt.tight_layout()
plt.show()