# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:35:19 2023

@author: elfre
"""
# TODO: potentially change values of max fat etc and also add constraints regarding saturated fat/chol
# use correct MLE coeff estimates for each category
# exclude one dish from mains with NA values 

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


#%%
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


dessert1_calories = [244,270,215,240]
dessert1_price = [40,30,20,20]
dessert1_score = [183.244,69.12,34.4928,156.586]
dessert1_names = ['KhanomMawKaeng','MangoStickyRice','LodChongThai','KanomKrok']
dessert1_fat = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Fat']) for i in range(len(dessert1_names))]
dessert1_protein = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Protein']) for i in range(len(dessert1_names))]
dessert1_carbs = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Carbs']) for i in range(len(dessert1_names))]

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

#%%

# coeffs from biogeme (FINAL)

results_mc1 = np.load('results_mc1.npy',allow_pickle=True)
coeffs_m1 = {'asc_0':0,
             'asc_1':results_mc1[0][0],
             'asc_2': results_mc1[1][0],
             'asc_3': results_mc1[2][0],
             'asc_4': results_mc1[3][0],
             'asc_5': results_mc1[4][0],
             'asc_6': 0,
             'b_calories': results_mc1[5][0],
             'b_discount': results_mc1[6][0],
             'b_price': results_mc1[7][0],
             'b_rating': results_mc1[8][0]}

results_mc2 = np.load('results_mc2.npy',allow_pickle=True)
coeffs_m2 = {'asc_0':0,
             'asc_1':results_mc2[0][0],
             'asc_2': results_mc2[1][0],
             'asc_3': results_mc2[2][0],
             'asc_4': results_mc2[3][0],
             'asc_5': 0,
             'asc_6': 0,
             'b_calories': results_mc2[4][0],
             'b_discount': results_mc2[5][0],
             'b_price': results_mc2[6][0],
             'b_rating': results_mc2[7][0]}

results_a1 = np.load('a1_results.npy',allow_pickle=True)
coeffs_a1 = {'asc_0':0,
             'asc_1':results_a1[0][0],
             'asc_2': results_a1[1][0],
             'asc_3': results_a1[2][0],
             'asc_4': results_a1[3][0],
             'asc_5': results_a1[4][0],
             'asc_6': 0,
             'b_calories': results_a1[5][0],
             'b_discount': results_a1[6][0],
             'b_price': results_a1[7][0],
             'b_rating': results_a1[8][0]}

results_a2 = np.load('a2_results.npy',allow_pickle=True)
coeffs_a2 = {'asc_0':0,
             'asc_1':results_a2[0][0],
             'asc_2': results_a2[1][0],
             'asc_3': results_a2[2][0],
             'asc_4': results_a2[3][0],
             'asc_5': 0,
             'asc_6': 0,
             'b_calories': results_a2[4][0],
             'b_discount': results_a2[5][0],
             'b_price': results_a2[6][0],
             'b_rating': results_a2[7][0]}

results_d1 = np.load('d1_results.npy',allow_pickle=True)
coeffs_d1 = {'asc_0':0,
             'asc_1':results_d1[0][0],
             'asc_2': results_d1[1][0],
             'asc_3': results_d1[2][0],
             'asc_4': results_d1[3][0],
             'asc_5': 0,
             'asc_6': 0,
             'b_calories': results_d1[4][0],
             'b_discount': results_d1[5][0],
             'b_price': results_d1[6][0],
             'b_rating': results_d1[7][0]}

results_d2 = np.load('results_d2.npy',allow_pickle=True)
coeffs_d2 = {'asc_0':0,
             'asc_1':results_d2[0][0],
             'asc_2': results_d2[1][0],
             'asc_3': results_d2[2][0],
             'asc_4': results_d2[3][0],
             'asc_5': 0,
             'asc_6': 0,
             'b_calories': results_d2[4][0],
             'b_discount': results_d2[5][0],
             'b_price': results_d2[6][0],
             'b_rating': results_d2[7][0]}

results_s = np.load('s_results.npy',allow_pickle=True)
coeffs_s = {'asc_0':0,
             'asc_1':results_s[0][0],
             'asc_2': results_s[1][0],
             'asc_3': results_s[2][0],
             'asc_4': results_s[3][0],
             'asc_5': results_s[4][0],
             'asc_6': results_s[5][0],
             'b_calories': results_s[6][0],
             'b_discount': results_s[7][0],
             'b_price': results_s[8][0],
             'b_rating': results_s[9][0]}

#%% 
# old way of calculting probs 
"""
def calc_probs(df,initial_coeffs,name,length):
    # calculating overall MNL probabilities
    
    coeffs = initial_coeffs
    
    # getting utilities for each person and each alternative 
    probs_choice = []
    for k in range(1,length+1):
        choice_df = df.loc[df['Choice']==k]
        utilities = []
        for i in range(0,len(choice_df)):
            if k == 1:
                # could add gender and payment also 
                #+ coeffs['b_payment'] * choice_df['Payment'].iloc[i] + coeffs['b_gender'] * choice_df['Gender'].iloc[i]
                utilities.append(coeffs['b_price'] * float(choice_df[f'{name} Price'].iloc[i]) + coeffs['b_calories'] * float(choice_df[f'{name} Calories'].iloc[i]) + coeffs['b_rating'] * float(choice_df[f'{name} Rating'].iloc[i]) + coeffs['b_discount'] * float(choice_df[f'{name} Discount'].iloc[i]) )
            else:
                utilities.append(coeffs[f'asc_{k-1}'] + coeffs['b_price'] * float(choice_df[f'{name} Price'].iloc[i]) + coeffs['b_calories'] * float(choice_df[f'{name} Calories'].iloc[i]) + coeffs['b_rating'] * float(choice_df[f'{name} Rating'].iloc[i]) + coeffs['b_discount'] * float(choice_df[f'{name} Discount'].iloc[i]) )


        choice_probs = [np.exp(utility) for utility in utilities]
        probs_choice.append(choice_probs)
        
        ###
        utilities = np.zeros(len(choice_df))

        choice_df[f'{name} Price'] = choice_df[f'{name} Price'].astype(float)
        choice_df[f'{name} Calories'] = choice_df[f'{name} Calories'].astype(
            float)
        choice_df[f'{name} Rating'] = choice_df[f'{name} Rating'].astype(
            float)
        choice_df[f'{name} Discount'] = choice_df[f'{name} Discount'].astype(
            float)

        utilities += coeffs[f'asc_{k-1}']
       # utilities1 += coeffs['b_gender'] * choice_df1['Gender'].values
        #utilities1 += coeffs['b_payment'] * choice_df1['Payment'].values
        utilities += coeffs['b_price'] * choice_df[f'{name} Price'].values
        utilities += coeffs['b_calories'] * choice_df[f'{name} Calories'].values
        utilities += coeffs['b_rating'] * choice_df[f'{name} Rating'].values
        utilities += coeffs['b_discount'] * choice_df[f'{name} Discount'].values
        
        choice_probs = np.exp(utilities)
        probs_choice.append(choice_probs)
    ###
    
    probs_choice = [sum(probs_choice[j]) for j in range(0,length)]
    
    return  probs_choice
"""

def calc_probs(df,coeffs,name,length):
    
    utilities = []
    
    df[f'{name} Calories'] = df[f'{name} Calories'].astype(float)
    df[f'{name} Price'] = df[f'{name} Price'].astype(float)
    df[f'{name} Rating'] = df[f'{name} Rating'].astype(float)
    df[f'{name} Discount'] = df[f'{name} Discount'].astype(float)
    
    for k in range(1,length+1):
        choice_df = df.loc[df['Choice']==k]
        utilities.append(coeffs[f'asc_{k-1}'] + coeffs['b_price'] * choice_df[f'{name} Price'].iloc[0] + coeffs['b_calories'] * choice_df[f'{name} Calories'].iloc[0] + coeffs['b_discount'] * choice_df[f'{name} Discount'].iloc[0] + coeffs['b_rating'] * choice_df[f'{name} Rating'].iloc[0])
    
    tot = 0
    for utility in utilities:
        tot += np.exp(utility)
    
    probs = []
    for utility in utilities:
        probs.append(np.exp(utility)/tot)
    print(probs)
    
    return probs
    
# changing to get probs for one category as a whole (e.g. A1 and A2) as we select two starters per day, therefore one from each
# of A1 and A2, if we did it per sub category then we would only be selecting one item and all conditional probs would be 1
# 2x starter, 2x main, 2x dessert per day over 5 days --> 10 per category
# combined categories either have 10 or 11 total items. Then adding in fact that each item can occur max twice we have 22 or 20 'items'
# from which we want 10. 22!/(10! 12!) is 646,646 combinations...

math.factorial(11)/(math.factorial(2)*math.factorial(9)) # 55
math.factorial(11)/(math.factorial(2)*math.factorial(9)) # 55
math.factorial(10)/(math.factorial(2)*math.factorial(8)) # 55
math.factorial(7)/(math.factorial(2)*math.factorial(5)) # 55

def get_cond_probs(df,coeffs,name,length,df2,coeffs2,name2,length2):
    utility_values0 = calc_probs(df, coeffs, name,length)
    utility_values1 = calc_probs(df2,coeffs2,name2,length2)
    
    utility_values = utility_values0 + utility_values1
    
    num_items = len(utility_values)
    
    # Dictionary to store conditional probabilities for each set
    conditional_probs_dict = {}
    
    # Calculate conditional probabilities for each possible set of 2 items
    for set_S in itertools.combinations(range(1, num_items + 1), 2):
        # Step 1: Calculate the sum of exponential utility values for the alternatives in set S
        sum_utility_set_S = np.sum([utility_values[i - 1] for i in set_S])
    
        # Step 2: Calculate the conditional probabilities P(i|set S) for each alternative i in set S
        conditional_probs = {i: utility_values[i - 1] / sum_utility_set_S for i in set_S}
    
        # Add the conditional probabilities to the dictionary
        conditional_probs_dict[set_S] = conditional_probs
        
    return conditional_probs_dict

conditional_probs_dict = get_cond_probs(appetizer1_df, coeffs_a1, 'A1', 6, appetizer2_df, coeffs_a1, 'A2', 5)

# combining lists for A1 and A2
total_a_scores = appetizer1_score + appetizer2_score
total_a_calories = appetizer1_calories + appetizer2_calories
total_a_carbs = appetizer1_carbs + appetizer2_carbs
total_a_protein = appetizer1_protein + appetizer2_protein
total_a_fat = appetizer1_fat + appetizer2_fat
total_a_salt = appetizer1_salt + appetizer2_salt

# mains
# need to remove one with nan value
cond_probs_m = get_cond_probs(main1_df, coeffs_m1, 'MC1', 6, main2_df, coeffs_m2, 'MC2', 5)
total_m_scores = main1_score + main2_score
total_m_calories = main1_calories + main2_calories
total_m_carbs = main1_carbs + main2_carbs
total_m_protein = main1_protein + main2_protein
total_m_fat = main1_fat + main2_fat
total_m_salt = main1_salt + main2_salt

# desserts
cond_probs_d = get_cond_probs(dessert1_df, coeffs_d1, 'D1', 4, dessert2_df, coeffs_d2, 'D2', 5)
total_d_scores = dessert1_score + dessert2_score
total_d_calories = dessert1_calories + dessert2_calories
total_d_carbs = dessert1_carbs + dessert2_carbs
total_d_protein = dessert1_protein + dessert2_protein
total_d_fat = dessert1_fat + dessert2_fat
total_d_salt = dessert1_salt + dessert2_salt

# soup
utility_values0 = calc_probs(soup_df, coeffs_s, 'S',6)

utility_values = utility_values0 

num_items = len(utility_values)

# Dictionary to store conditional probabilities for each set
cond_probs_s = {}

# Calculate conditional probabilities for each possible set of 2 items
for set_S in itertools.combinations(range(1, num_items + 1), 2):
    # Step 1: Calculate the sum of exponential utility values for the alternatives in set S
    sum_utility_set_S = np.sum([utility_values[i - 1] for i in set_S])

    # Step 2: Calculate the conditional probabilities P(i|set S) for each alternative i in set S
    conditional_probs = {i: utility_values[i - 1] / sum_utility_set_S for i in set_S}

    # Add the conditional probabilities to the dictionary
    cond_probs_s[set_S] = conditional_probs
    
total_s_scores = impute_nan_with_average(soup_score)
total_s_calories = impute_nan_with_average(soup_calories)
total_s_carbs = impute_nan_with_average(soup_carbs)
total_s_protein = impute_nan_with_average(soup_protein)
total_s_fat = impute_nan_with_average(soup_fat)
total_s_salt = impute_nan_with_average(soup_salt)

#%% 

# over multiple days

def opt_menu(total_a_calories,total_a_scores,total_a_carbs,total_a_protein,total_a_salt,total_a_fat,conditional_probs_dict,max_cals,min_protein,max_protein,min_carbs,max_carbs,min_fat,max_fat,max_salt):
    problem = LpProblem("Offer_Set_Selection", LpMinimize)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    item_vars = LpVariable.dicts("Item", ((day, item) for day in days for item in range(1, len(total_a_scores) + 1)), cat='Binary')
    arrival_rate = 130
    
    objective_values = []
    fat_constraints = []
    protein_constraints = []
    carb_constraints = []
    salt_constraints = []
    calorie_constraints = []
    
    for set_S in conditional_probs_dict.keys():
        for day in days:
            # Initialize the objective value for this offer set and day to zero
            objective_value = 0
            total_fat = 0
            total_protein = 0
            total_carbs = 0
            total_salt = 0
            total_calories = 0
            # Loop through each item in the offer set
            for item in set_S:
                # Add the contribution of each item in the offer set to the objective value
                objective_value += arrival_rate * item_vars[(day, item)] * conditional_probs_dict[set_S][item] * total_a_scores[item - 1]
                total_fat += item_vars[(day, item)] * total_a_fat[item - 1]
                total_carbs += item_vars[(day, item)] * total_a_carbs[item - 1]
                total_protein += item_vars[(day, item)] * total_a_protein[item - 1]
                total_salt += item_vars[(day, item)] * total_a_salt[item - 1]
                total_calories += item_vars[(day, item)] * total_a_calories[item - 1]
            # Append the objective value for this offer set and day to the list
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
    
    # Add constraints for each day to select only 2 items
    for day in days:
        problem += lpSum(item_vars[(day, item)] for item in range(1, len(total_a_scores) + 1)) == 2
    
    # Add constraints for each item to ensure it is selected at most twice over the 5 days
    #for item in range(1, len(total_a_scores) + 1):
     #   problem += lpSum(item_vars[(day, item)] for day in days) <= 2
    
    item_occurrences = LpVariable.dicts("Item_Occurrence", [(day, item) for day in days for item in range(1, len(total_a_scores) + 1)], cat='Binary')
    max_item_occurrences = 2
    
    # Constraint to ensure that each item is selected at most 'max_item_occurrences' times
    for item in range(1, len(total_a_scores) + 1):
        problem += lpSum(item_occurrences[(day, item)] for day in days) <= max_item_occurrences
    
    # Constraint to relate the occurrence variables to the decision variables
    for day in days:
        for set_S in conditional_probs_dict.keys():
            for item in range(1, len(total_a_scores) + 1):
                if item in set_S:
                    problem += item_vars[(day, item)] <= item_occurrences[(day, item)]
    
    # convsecutive days
    for item in range(1, len(total_a_scores) + 1):
        for i in range(len(days) - 1):
            problem += item_occurrences[(days[i], item)] + item_occurrences[(days[i + 1], item)] <= 1        

    for fat_constraint in fat_constraints:
        problem += fat_constraint
        
    for p_constraint in protein_constraints:
        problem += p_constraint
        
    for c_constraint in carb_constraints:
        problem += c_constraint
        
    if len(total_a_calories) != 6: # for some reason this causes soup to have errors and infeasible solution even tho all items have salt < 2000 mg
    # so just removed constraint for that
        for s_constraint in salt_constraints:
            problem += s_constraint
        
    for cal_constraint in calorie_constraints:
        problem += cal_constraint
        
    problem.solve()
    
    return [item for item in item_vars if item_vars[item].value() == 1],item_vars,problem

#%%
# testing

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


total_a_price = appetizer1_price + appetizer2_price
total_m_price = main1_price + main2_price
total_d_price = dessert1_price + dessert2_price
total_s_price = soup_price

price_dict = {
    'FriedCalamari': total_a_price[0],
    'EggRolls': total_a_price[1],
    'ShrimpCake': total_a_price[2],
    'FishCake':  total_a_price[3],
    'FriedShrimpBall': total_a_price[4],
    'HerbChicken': total_a_price[5],
    'GrilledShrimp': total_a_price[6],
    'CrispyCrab': total_a_price[7],
    'MiangKham': total_a_price[8],
    'ShrimpDumpling': total_a_price[9],
    'SaladRoll': total_a_price[10],
    'PadKrapow': total_m_price[0],
     'PadThai': total_m_price[1],
     'HainaneseChickenRice': total_m_price[2],
     'GreenChickenCurry': total_m_price[3],
     'ShrimpStickyRice': total_m_price[4],
     'ShrimpFriedRice': total_m_price[5],
     'ChickenPorridge': total_m_price[6],
     'KanomJeenNamYa': total_m_price[7],
     'ShrimpGlassNoodles': total_m_price[8],
     'AmericanFriedRice': total_m_price[9],
     'SausageFriedRice': total_m_price[10],
     'KhanomMawKaeng': total_d_price[0],
      'MangoStickyRice': total_d_price[1],
      'LodChongThai': total_d_price[2],
      'KanomKrok': total_d_price[3],
      'Pudding': total_d_price[4],
      'VanillaIceCream': total_d_price[5],
      'ApplePie': total_d_price[6],
      'ChocolateCake': total_d_price[7],
      'ChocolateIcecream': total_d_price[8],
      'TomJuedMooSap': total_s_price[0],
       'KaiPalo': total_s_price[1],
       'OnionSoup': total_s_price[2],
       'TomYumKung': total_s_price[3],
       'TomKhaGai': total_s_price[4],
       'GlassNoodlesSoup': total_s_price[5],
       }

# scenario 1
test,item_vars,problem = opt_menu(total_a_calories,total_a_scores, total_a_carbs, total_a_protein, total_a_salt, total_a_fat, conditional_probs_dict,405,0,35,0,50,0,20,1300) # original: 450, 75, 38.8, 2000
main_items,main_item_vars,main_problem = opt_menu(total_m_calories,total_m_scores, total_m_carbs, total_m_protein, total_m_salt, total_m_fat, cond_probs_m,550,0,35,0,60,0,35,1800) 
dessert_items,dessert_item_vars,dessert_problem = opt_menu(total_d_calories,total_d_scores, total_d_carbs, total_d_protein, total_d_salt, total_d_fat, cond_probs_d, 500,0,35,0,90,0,20,300) 
soup_items,soup_item_vars,soup_problem = opt_menu(total_s_calories, total_s_scores, total_s_carbs, total_s_protein, total_s_salt, total_s_fat, cond_probs_s, 500,0,40,0,50,0,30,500)

# relaxing constraints
test,item_vars,problem = opt_menu(total_a_calories,total_a_scores, total_a_carbs, total_a_protein, total_a_salt, total_a_fat, conditional_probs_dict,600,0,50,0,70,0,40,2000) # original: 450, 75, 38.8, 2000
main_items,main_item_vars,main_problem = opt_menu(total_m_calories,total_m_scores, total_m_carbs, total_m_protein, total_m_salt, total_m_fat, cond_probs_m,650,0,50,0,75,0,45,2000) 
dessert_items,dessert_item_vars,dessert_problem = opt_menu(total_d_calories,total_d_scores, total_d_carbs, total_d_protein, total_d_salt, total_d_fat, cond_probs_d, 600,0,70,0,90,0,40,1000) 
soup_items,soup_item_vars,soup_problem = opt_menu(total_s_calories, total_s_scores, total_s_carbs, total_s_protein, total_s_salt, total_s_fat, cond_probs_s, 600,0,60,0,70,0,40,1000)

appetizer_dishes = {
    'FriedCalamari': 1,
    'EggRolls': 2,
    'ShrimpCake': 3,
    'FishCake':  4,
    'FriedShrimpBall': 5,
    'HerbChicken': 6,
    'GrilledShrimp':7,
    'CrispyCrab': 8,
    'MiangKham': 9,
    'ShrimpDumpling': 10,
    'SaladRoll': 11 
    }


def get_menu(dishes_dict,problem,item_vars,items):
    
    total_calories_week = 0
    total_fat_week = 0
    total_protein_week = 0
    total_salt_week = 0
    total_carbs_week = 0
    total_price_week = 0
    total_score_week = 0
    
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        print(f"Dishes selected for {day}:")
        for j, dish_name in enumerate(dishes_dict.keys()):
                if value(item_vars[(day,j+1)]) == 1:
                    print(f"{dish_name}")
                    
    variable_to_dish = {}

    # Iterate over the days
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        # Iterate over the categories
            # Get the dish dictionary for the category
        dish_dict = dishes_dict

            # Iterate over the items in the dish dictionary
        for j, dish_name in enumerate(dish_dict.keys()):
            # Map the variable x[day][category][j] to the dish name
            variable_to_dish[item_vars[(day,j+1)].name] = dish_name

    # Print the mapping
    #for variable, dish_name in variable_to_dish.items():
     #   print(variable, '->', dish_name)
                
    # Print the menu for each day
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
            # Check if the variable is selected (has a value greater than 0) and belongs to the given day and category
            if variable.varValue > 0 and variable.name in variable_to_dish and day in variable.name:
                dish_name = variable_to_dish[variable.name]

                # Retrieve the nutritional information for the dish from the dictionaries
                dish_calories = calories_dict[dish_name]
                dish_fat = fat_dict[dish_name]
                dish_carbs = carbs_dict[dish_name]
                dish_protein = protein_dict[dish_name]
                dish_salt = salt_dict[dish_name]
               # dish_sat = sat_dict[dish_name]
                dish_price = price_dict[dish_name]
                dish_score = scores_dict[dish_name]

                # Calculate the total calories for the day
                if math.isnan(dish_calories) == False:
                    total_calories_1 += dish_calories
                if math.isnan(dish_fat) == False:
                    total_fat_1 += dish_fat 
                if math.isnan(dish_protein) == False:
                    total_protein_1 += dish_protein 
                if math.isnan(dish_carbs) == False:
                    total_carbs_1 += dish_carbs 
                if math.isnan(dish_salt) == False:
                    total_salt_1 += dish_salt 
                if math.isnan(dish_price) == False:
                    total_price_1 += dish_price
                if math.isnan(dish_score) == False:
                    total_score_1 += dish_score

                # Print the dish name, quantity, and nutritional information
                print(f"{dish_name}")
                print(f"  Calories: {dish_calories}")
                print(f"  Fat: {dish_fat}g")
                print(f"  Carbs: {dish_carbs}g")
                print(f"  Protein: {dish_protein}g")
                print(f"  Salt: {dish_salt}g")
               # print(f"  Saturated Fat: {dish_sat}g")
               
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
    
soup_dishes = {
    'TomJuedMooSap': 1,
    'KaiPalo': 2,
    'OnionSoup':3,
    'TomYumKung':4,
    'TomKhaGai':5,
    'GlassNoodlesSoup':6}
    
get_menu(appetizer_dishes,problem,item_vars,test)   
get_menu(main_dishes, main_problem, main_item_vars, main_items)
get_menu(dessert_dishes, dessert_problem, dessert_item_vars, dessert_items) 
get_menu(soup_dishes, soup_problem, soup_item_vars, soup_items)

prices_opt_s1 = [820,540,270,570]

prices_opt_rel = [780,560,240,570]

scores_opt_s1 = [9167.84,6478.42,1279.4,6141.27]

scores_opt_rel = [5832.16,4872.63,1975.39,6141.27]
#%%
plt.rcParams['font.sans-serif'] = "Georgia"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# results from constraints 1
#objective_values_new = [3936900.46,3111509.65,534859.42,1721923.97]
appetizers = [1842, 82.72*9,107.68*4,179.34*4,4395.55]
mains = [2420, 127.85*9, 135.13*4, 196.82*4, 5818.85]
desserts = [2178,67.1*9,28.74*4,385.69*4,650.32]
soup = [2194, 77.7*9, 117.36*4, 152.3*4, 7902.83]

categories = ['Appetizers', 'Mains', 'Desserts', 'Soup']

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(categories, objective_values_new, color='skyblue')
plt.xlabel('Objective Value')
plt.ylabel('Categories')
plt.title('Objective Values by Category')
plt.show()

####
objective_values = [6257912.48,4102681.88,673641.49,1721923.97]
categories = ['Appetizers', 'Mains', 'Desserts', 'Soup']

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(categories, objective_values, color='skyblue')
plt.xlabel('Objective Value')
plt.ylabel('Categories')
plt.title('Objective Values by Category')
plt.show()

####

# Define the nutrient categories and their corresponding values - relaxed constraints
nutrients = ['Calories (kcal)', 'Fat (g)', 'Protein (g)', 'Carbs (g)', 'Salt (g)']
appetizers = [2192, 102.5*9, 92.72*4, 236.5*4, 6536.98]
mains = [2814, 159.17*9, 141.71*4, 229.7*4, 6337.1]
desserts = [2780,131.33*9,31.73*4,392.82*4,860.87]
soup = [2194, 77.7*9, 117.36*4, 152.3*4, 7902.83]


# Create a bar chart

color_palette = sns.color_palette("Set2")  # You can choose a different palette

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.15
x = range(len(nutrients))

# Specify colors for each category using the color palette
ax.bar([val - width*2 for val in x], appetizers, width, label='Appetizers', color=color_palette[0])
ax.bar([val - width for val in x], mains, width, label='Mains', color=color_palette[1])
ax.bar(x, desserts, width, label='Desserts', color=color_palette[2])
ax.bar([val + width for val in x], soup, width, label='Soup', color=color_palette[3])

ax.set_xticks(x)
ax.set_xticklabels(nutrients)
ax.set_ylabel('Amount')
ax.set_title('Nutrient Amounts by Category')
ax.legend()

plt.tight_layout()
plt.show()
#####

# Calculate the total values of fat, protein, and carbs for each category
total_fat_appetizers = appetizers[1]
total_protein_appetizers = appetizers[2]
total_carbs_appetizers = appetizers[3]

total_fat_mains = mains[1]
total_protein_mains = mains[2]
total_carbs_mains = mains[3]

total_fat_desserts = desserts[1]
total_protein_desserts = desserts[2]
total_carbs_desserts = desserts[3]

total_fat_soup = soup[1]
total_protein_soup = soup[2]
total_carbs_soup = soup[3]

# Create subplots for each category
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Proportions of Total Calories by Fat, Protein, and Carbs')
labels = ['Fat', 'Protein', 'Carbs']
colors = ['purple', 'green', 'blue']
color_palette = sns.color_palette("Set2")  # Choose a different palette

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

# Soup
axs[1, 1].pie([total_fat_soup, total_protein_soup, total_carbs_soup], labels=labels, colors=color_palette,
        autopct='%1.1f%%', shadow=True, startangle=140)
axs[1, 1].set_title('Soup')

plt.tight_layout()
plt.savefig("pie_chart_mnl.pdf", bbox_inches='tight', format='pdf')
plt.show()


#%%

# optimising for price instead 

def opt_menu_price(total_a_price,total_a_calories,total_a_scores,total_a_carbs,total_a_protein,total_a_salt,total_a_fat,conditional_probs_dict,max_cals,min_protein,max_protein,min_carbs,max_carbs,min_fat,max_fat,max_salt):
    problem = LpProblem("Offer_Set_Selection", LpMaximize)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    item_vars = LpVariable.dicts("Item", ((day, item) for day in days for item in range(1, len(total_a_scores) + 1)), cat='Binary')
    arrival_rate = 130
    
    objective_values = []
    fat_constraints = []
    protein_constraints = []
    carb_constraints = []
    salt_constraints = []
    calorie_constraints = []
    
    for set_S in conditional_probs_dict.keys():
        for day in days:
            # Initialize the objective value for this offer set and day to zero
            objective_value = 0
            total_fat = 0
            total_protein = 0
            total_carbs = 0
            total_salt = 0
            total_calories = 0
            # Loop through each item in the offer set
            for item in set_S:
                # Add the contribution of each item in the offer set to the objective value
                objective_value += arrival_rate * item_vars[(day, item)] * conditional_probs_dict[set_S][item] * total_a_price[item - 1]
                total_fat += item_vars[(day, item)] * total_a_fat[item - 1]
                total_carbs += item_vars[(day, item)] * total_a_carbs[item - 1]
                total_protein += item_vars[(day, item)] * total_a_protein[item - 1]
                total_salt += item_vars[(day, item)] * total_a_salt[item - 1]
                total_calories += item_vars[(day, item)] * total_a_calories[item - 1]
            # Append the objective value for this offer set and day to the list
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
    
    # Add constraints for each day to select only 2 items
    for day in days:
        problem += lpSum(item_vars[(day, item)] for item in range(1, len(total_a_scores) + 1)) == 2
    
    # Add constraints for each item to ensure it is selected at most twice over the 5 days
    #for item in range(1, len(total_a_scores) + 1):
     #   problem += lpSum(item_vars[(day, item)] for day in days) <= 2
    
    item_occurrences = LpVariable.dicts("Item_Occurrence", [(day, item) for day in days for item in range(1, len(total_a_scores) + 1)], cat='Binary')
    max_item_occurrences = 2
    
    # Constraint to ensure that each item is selected at most 'max_item_occurrences' times
    for item in range(1, len(total_a_scores) + 1):
        problem += lpSum(item_occurrences[(day, item)] for day in days) <= max_item_occurrences
    
    # Constraint to relate the occurrence variables to the decision variables
    for day in days:
        for set_S in conditional_probs_dict.keys():
            for item in range(1, len(total_a_scores) + 1):
                if item in set_S:
                    problem += item_vars[(day, item)] <= item_occurrences[(day, item)]
    
    # convsecutive days
    for item in range(1, len(total_a_scores) + 1):
        for i in range(len(days) - 1):
            problem += item_occurrences[(days[i], item)] + item_occurrences[(days[i + 1], item)] <= 1        

    for fat_constraint in fat_constraints:
        problem += fat_constraint
        
    for p_constraint in protein_constraints:
        problem += p_constraint
        
    for c_constraint in carb_constraints:
        problem += c_constraint
        
    if len(total_a_calories) != 6: # for some reason this causes soup to have errors and infeasible solution even tho all items have salt < 2000 mg
    # so just removed constraint for that
        for s_constraint in salt_constraints:
            problem += s_constraint
        
    for cal_constraint in calorie_constraints:
        problem += cal_constraint
        
    problem.solve()
    
    return [item for item in item_vars if item_vars[item].value() == 1],item_vars,problem

# With relaxed constraints 
app_price_items, app_price_vars, app_price_prob = opt_menu_price(total_a_price, total_a_calories, total_a_scores, total_a_carbs, total_a_protein, total_a_salt, total_a_fat, conditional_probs_dict,600,0,50,0,70,0,40,2000)

app_menu = get_menu(appetizer_dishes, app_price_prob, app_price_vars, app_price_items)

m_price_items,m_price_vars,m_price_prob = opt_menu_price(total_m_price, total_m_calories, total_m_scores, total_m_carbs, total_m_protein, total_m_salt, total_m_fat, cond_probs_m, 650,0,50,0,75,0,45,2000)

main_menu = get_menu(main_dishes,m_price_prob,m_price_vars,m_price_items)

d_price_items,d_price_vars,d_price_prob = opt_menu_price(total_d_price, total_d_calories, total_d_scores, total_d_carbs, total_d_protein, total_d_salt, total_d_fat, cond_probs_d, 600,0,70,0,90,0,40,1000) 

dessert_menu = get_menu(dessert_dishes,d_price_prob,d_price_vars,d_price_items)

s_price_items,s_price_vars,s_price_prob = opt_menu_price(total_s_price, total_s_calories, total_s_scores, total_s_carbs, total_s_protein, total_s_salt, total_s_fat, cond_probs_s, 600,0,60,0,70,0,40,1000)  

soup_menu = get_menu(soup_dishes,s_price_prob,s_price_vars,s_price_items)

#obj_vals_price = [541401.63,415066.53,186611.33,235711.46]
tot_price = [app_menu[5],main_menu[5],dessert_menu[5],soup_menu[5]]

# Bar plot comparing obj values for optimising price vs sustainability score

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(categories, obj_vals_price, color='skyblue')
plt.xlabel('Objective Value')
plt.ylabel('Categories')
plt.title('Objective Values by Category')
plt.show()

categories = ['Appetizers', 'Mains', 'Desserts', 'Soup']
#obj_vals_price = [541401.63, 415066.53, 186611.33, 235711.46]
#objective_values_new = [3936900.46, 3111509.65, 534859.42, 1721923.97]
scores_price = [app_menu[6],main_menu[6],dessert_menu[6],soup_menu[6]]

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width
bar_width = 0.4

# Calculate the positions for the bars
bar_positions_price = np.arange(len(categories))
bar_positions_new = bar_positions_price + bar_width

# Create the bar plots for both cases
bar1 = ax.barh(bar_positions_price, obj_vals_price, bar_width, label='Price Optimization', color='skyblue')
bar2 = ax.barh(bar_positions_new, objective_values_new, bar_width, label='Sustainability Score Optimization', color='lightgreen')

# Set labels and title
ax.set_xlabel('Objective Value')
ax.set_ylabel('Categories')
ax.set_title('Objective Values by Category')
ax.set_yticks(bar_positions_price + bar_width / 2)
ax.set_yticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()

# Bar plot comparing total price when optimising for price (maximising) vs minimising S score
plt.figure(figsize=(10, 6))
plt.barh(categories, obj_vals_price, color='skyblue')
plt.xlabel('Objective Value')
plt.ylabel('Categories')
plt.title('Objective Values by Category')
plt.show()

categories = ['Appetizers', 'Mains', 'Desserts', 'Soup']
obj_vals_price = [541401.63, 415066.53, 186611.33, 235711.46]
objective_values_new = [3936900.46, 3111509.65, 534859.42, 1721923.97]

# Create a figure and axes

fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width
bar_width = 0.4

# Calculate the positions for the bars
categories = ['Appetizer','Main','Dessert','Soup']
bar_positions_price = np.arange(len(categories))
bar_positions_new = bar_positions_price + bar_width

# Create the bar plots for both cases
bar1 = ax.barh(bar_positions_price, tot_price, bar_width, label='Price Optimization', color='skyblue')
bar2 = ax.barh(bar_positions_new, prices_opt_rel, bar_width, label='Sustainability Score Optimization', color='lightgreen')

# Set labels and title
ax.set_xlabel('Total Menu Price')
ax.set_ylabel('Categories')
ax.set_title('Total Price by Category')
ax.set_yticks(bar_positions_price + bar_width / 2)
ax.set_yticklabels(categories)
ax.legend()

plt.savefig("price_comparison.pdf", bbox_inches='tight', format='pdf')
plt.tight_layout()
plt.show()

# Bar plot comparing total S score when optimising for price vs S score
fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width
bar_width = 0.4

# Calculate the positions for the bars
bar_positions_price = np.arange(4)
bar_positions_new = bar_positions_price + bar_width

# Create the bar plots for both cases
bar1 = ax.barh(bar_positions_price, scores_price, bar_width, label='Price Optimization', color='skyblue')
bar2 = ax.barh(bar_positions_new, scores_opt_rel, bar_width, label='Sustainability Score Optimization', color='lightgreen')

# Set labels and title
ax.set_xlabel('Total Menu Carbon Footprint Score')
ax.set_ylabel('Categories')
ax.set_title('Total Score by Category')
ax.set_yticks(bar_positions_price + bar_width / 2)
ax.set_yticklabels(categories)
ax.legend()

plt.savefig("score_comparison.pdf", bbox_inches='tight', format='pdf')
plt.tight_layout()
plt.show()
