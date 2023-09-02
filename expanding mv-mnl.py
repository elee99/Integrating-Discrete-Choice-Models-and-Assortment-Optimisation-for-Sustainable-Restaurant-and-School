# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:04:12 2023

@author: elfre
"""
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

csv_files = glob.glob('menu*.csv')

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

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

df['A2 Choice'] = df['Appetizer 2'].replace({
    'GrilledShrimp': 1,
    'CrispyCrab': 2,
    'MiangKham': 3,
    'ShrimpDumpling': 4,
    'CrispyCrab ': 2,
    'SaladRoll': 5})

df['MC2 Choice'] = df['Main Course 2'].replace({
    'ChickenPorridge':1,
    'KanomJeenNamYa':2,
    'ShrimpGlassNoodles':3,
    'AmericanFriedRice':4,
    'SausageFriedRice':5})

df['D2 Choice'] = df['Dessert 2'].replace({
    'Pudding': 1,
    'VanillaIceCream': 2,
    'ApplePie': 3,
    'ChocolateCake': 4,
    'ChocolateIcecream': 5})

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
dessert1_names = ['KhanomMawKaeng','MangoStickyRice','LodChongThai','KanomKrok']
#dessert1_fat = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Fat']) for i in range(len(dessert1_names))]
dessert1_fat = [7.375,8.872727272727273, 7.199999999999999, 6.37037037037037, 7.0588235294117645]
#dessert1_protein = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Protein']) for i in range(len(dessert1_names))]
#sum(dessert1_protein)/4
dessert1_protein = [2.962,4.4363636363636365,3.0857142857142854,0.7962962962962963,3.5294117647058822]
#dessert1_carbs = [float(nutrients.loc[nutrients['Dish names']==dessert1_names[i]]['Carbs']) for i in range(len(dessert1_names))]
#sum(dessert1_carbs)/4
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

# For MVMNL if we has 6 categories, 4 with 5 items and 2 with 6 --> total of 32 products
# Would have asc for each item in each category: 32 ascs
# Beta price/rating/cal/discount coeff for every item in every category: 32*4 coeffs
# Assoc param between every item in each category and every other item in each category: 156 for those with 6 items, 125 for those
# with 5. = 812 coeffs
# total of 972 coeffs (incl. those set to 0)

initial_coeffs = [0.01]*1184

def get_coeffs(initial_coeffs):
    num_categories = 6
    num_items_per_category = [6, 5, 6, 5, 5, 5]
    
    coefficients = {}
    
    for cat_num, num_items in enumerate(num_items_per_category, start=1):
        for item_num in range(1, num_items + 1):
            coeff_name = f'asc_{cat_num}{item_num}'
            if item_num == 1:
                coefficients[coeff_name] = 0
            else:
                coeff_index = (cat_num - 1) * num_items + (item_num - 2)
                coefficients[coeff_name] = initial_coeffs[0:32][coeff_index]
        
    coeff_prefixes = ['b_price_', 'b_calories_', 'b_rating_', 'b_discount_']
    
    beta_coefficients = {}
    
    for cat_num, num_items in enumerate(num_items_per_category, start=1):
        for item_num in range(1, num_items + 1):
            for coeff_prefix in coeff_prefixes:
                coeff_name = f'{coeff_prefix}{cat_num}{item_num}'
                if item_num == 1:
                    beta_coefficients[coeff_name] = 0
                else:
                    coeff_index = (cat_num - 1) * num_items + (item_num - 2)
                    beta_coefficients[coeff_name] = initial_coeffs[32:160][coeff_index]
    
    association_coefficients = {}
    
    for cat_from in range(1, num_categories + 1):
        for item_from in range(1, num_items_per_category[cat_from - 1] + 1):
            for cat_to in range(1, num_categories + 1):
                for item_to in range(1, num_items_per_category[cat_to - 1] + 1):
                    coeff_name = f'assoc_{cat_from}_{cat_to}_{item_from}_{item_to}'
                    
                    last_number = int(str(item_to)[-1])
                    penultimate_number = int(str(item_to)[-2]) if len(str(item_to)) > 1 else 0
                    
                    if last_number == 1 or penultimate_number == 1:
                        association_coefficients[coeff_name] = 0
                    else:
                        coeff_index = (cat_from - 1) * sum(num_items_per_category) + \
                                      (item_from - 1) * num_categories + \
                                      (cat_to - 1) * num_items_per_category[cat_to - 1] + \
                                      (item_to - 1)
                        association_coefficients[coeff_name] = initial_coeffs[160:1183][coeff_index]
    
    coeffs = {**coefficients, **beta_coefficients, **association_coefficients}
    return coeffs

coeffs = get_coeffs(initial_coeffs)

def calc_probs_a1(df, coeffs, name, length, cat, c1, c2, c3, c4, c5, c1_len, c2_len, c3_len, c4_len, c5_len):

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
        
        for j in range(1, c1_len + 1):
            for i in range(1, c2_len + 1):
                for m in range(1,c3_len+1):
                    for l in range(1,c4_len+1):
                        for g in range(1,c5_len+1):
                            
            # need to get separate probs for each k value
                            utilities_dict[k].append(choice_probs + np.exp(coeffs[f'assoc_{cat}_{c1}_{k}_{j}'] + coeffs[f'assoc_{cat}_{c2}_{k}_{i}'] + coeffs[f'assoc_{cat}_{c3}_{k}_{m}'] + coeffs[f'assoc_{cat}_{c4}_{k}_{l}'] + coeffs[f'assoc_{cat}_{c5}_{k}_{g}']))

    tot_utility = 0
    for i in range(1,length+1):
        tot_utility += np.sum(np.sum(utilities_dict[i][j] for j in range(len(utilities_dict[1]))))

    probabilities = []
    for k in range(1,length+1):
        probabilities.append([np.sum(utilities_dict[k][j], axis=0) / tot_utility for j in range(len(utilities_dict[1]))])

    return probabilities

cond_probs_a1 = calc_probs_a1(df, coeffs, 'A1', 6, 1, 2, 3, 4, 5, 6, 5, 6, 5, 5, 5)
cond_probs_a2 = calc_probs_a1(df, coeffs, 'A2', 5, 2, 1, 3, 4, 5, 6, 6, 6, 5, 5, 5)
cond_probs_m1 = calc_probs_a1(df, coeffs, 'MC1', 6, 3, 1, 2, 4, 5, 6, 6, 5, 5, 5, 5)
cond_probs_m2 = calc_probs_a1(df, coeffs, 'MC2', 5, 4, 1, 2, 3, 5, 6, 6, 5, 6, 5, 5)
cond_probs_d1 = calc_probs_a1(df, coeffs, 'D1', 5, 5, 1, 2, 3, 4, 6, 6, 5, 6, 5, 5)
cond_probs_d2 = calc_probs_a1(df, coeffs, 'D2', 5, 6, 1, 2, 3, 4, 5, 6, 5, 6, 5, 5)

def calc_len(df):
    lengths_a1 = []
    lengths_m1 = []
    lengths_d1 = []
    lengths_a2 = []
    lengths_m2 = []
    lengths_d2 = []

    categories = ['A1','MC1','D1','A2','MC2','D2']
    for i in range(1,6+1):
        lengths_a1.append(len(df.loc[df[f'{categories[0]} Choice'] == i]))
        lengths_m1.append(len(df.loc[df[f'{categories[1]} Choice'] == i]))
    for i in range(1,5+1):
        lengths_d1.append(len(df.loc[df[f'{categories[2]} Choice'] == i]))
        lengths_a2.append(len(df.loc[df[f'{categories[3]} Choice'] == i]))
        lengths_m2.append(len(df.loc[df[f'{categories[4]} Choice'] == i]))
        lengths_d2.append(len(df.loc[df[f'{categories[5]} Choice'] == i]))

    return lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2

lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2 = calc_len(df)

def calc_log_lik(coeffs,lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2):
    cond_probs_a1 = calc_probs_a1(df, coeffs, 'A1', 6, 1, 2, 3, 4, 5, 6, 5, 6, 5, 5, 5)
    cond_probs_a2 = calc_probs_a1(df, coeffs, 'A2', 5, 2, 1, 3, 4, 5, 6, 6, 6, 5, 5, 5)
    cond_probs_m1 = calc_probs_a1(df, coeffs, 'MC1', 6, 3, 1, 2, 4, 5, 6, 6, 5, 5, 5, 5)
    cond_probs_m2 = calc_probs_a1(df, coeffs, 'MC2', 5, 4, 1, 2, 3, 5, 6, 6, 5, 6, 5, 5)
    cond_probs_d1 = calc_probs_a1(df, coeffs, 'D1', 5, 5, 1, 2, 3, 4, 6, 6, 5, 6, 5, 5)
    cond_probs_d2 = calc_probs_a1(df, coeffs, 'D2', 5, 6, 1, 2, 3, 4, 5, 6, 5, 6, 5, 5)
    
    all_probs = [cond_probs_a1, cond_probs_m1, cond_probs_d1, cond_probs_a2, cond_probs_m2, cond_probs_d2]

    categories = ['A1','MC1','D1','A2','MC2','D2']
    lengths = [6,6,5,5,5,5]
    total_log_lik = 0
    
    log_lik = 0 
    for i in range(1,6+1):
        log_lik += lengths_a1[i-1] * np.log(np.sum(all_probs[0][i-1]))
        log_lik += lengths_m1[i-1] * np.log(np.sum(all_probs[1][i-1]))
    for i in range(1,5+1):
        log_lik += lengths_d1[i-1] * np.log(np.sum(all_probs[2][i-1]))
        log_lik += lengths_a2[i-1] * np.log(np.sum(all_probs[3][i-1]))
        log_lik += lengths_m2[i-1] * np.log(np.sum(all_probs[4][i-1]))
        log_lik += lengths_d2[i-1] * np.log(np.sum(all_probs[5][i-1]))

    total_log_lik += log_lik 
    
    return total_log_lik
"""
def calc_log_lik(coeffs):
    cond_probs_a1 = calc_probs_a1(df, coeffs, 'A1', 6, 1, 2, 3, 4, 5, 6, 5, 6, 5, 5, 5)
    cond_probs_a2 = calc_probs_a1(df, coeffs, 'A2', 5, 2, 1, 3, 4, 5, 6, 6, 6, 5, 5, 5)
    cond_probs_m1 = calc_probs_a1(df, coeffs, 'MC1', 6, 3, 1, 2, 4, 5, 6, 6, 5, 5, 5, 5)
    cond_probs_m2 = calc_probs_a1(df, coeffs, 'MC2', 5, 4, 1, 2, 3, 5, 6, 6, 5, 6, 5, 5)
    cond_probs_d1 = calc_probs_a1(df, coeffs, 'D1', 5, 5, 1, 2, 3, 4, 6, 6, 5, 6, 5, 5)
    cond_probs_d2 = calc_probs_a1(df, coeffs, 'D2', 5, 6, 1, 2, 3, 4, 5, 6, 5, 6, 5, 5)
    
    all_probs = [cond_probs_a1, cond_probs_m1, cond_probs_d1, cond_probs_a2, cond_probs_m2, cond_probs_d2]

    categories = ['A1','MC1','D1','A2','MC2','D2']
    lengths = [6,6,5,5,5,5]
    total_log_lik = 0
    
    log_lik = 0 
    for i in range(1,6+1):
        log_lik += len(df.loc[df[f'{categories[0]} Choice'] == i]) * np.log(np.sum(all_probs[0][i-1]))
        log_lik += len(df.loc[df[f'{categories[1]} Choice'] == i]) * np.log(np.sum(all_probs[1][i-1]))
    for i in range(1,5+1):
        log_lik += len(df.loc[df[f'{categories[2]} Choice'] == i]) * np.log(np.sum(all_probs[2][i-1]))
        log_lik += len(df.loc[df[f'{categories[3]} Choice'] == i]) * np.log(np.sum(all_probs[3][i-1]))
        log_lik += len(df.loc[df[f'{categories[4]} Choice'] == i]) * np.log(np.sum(all_probs[4][i-1]))
        log_lik += len(df.loc[df[f'{categories[5]} Choice'] == i]) * np.log(np.sum(all_probs[5][i-1]))

    total_log_lik += log_lik 
    
    return total_log_lik

calc_log_lik(coeffs)
"""
calc_log_lik(coeffs,lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2)

def objective(initial_coeffs,df,lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2):
    # Call calc_probs with the dataframe and the current set of coefficients
    coeffs = get_coeffs(initial_coeffs)
    log_lik = calc_log_lik(coeffs,lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2)
    print(log_lik)
    return -log_lik

res = minimize(lambda theta: objective(theta,df,lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2), [0.01]*1184, method='SLSQP', options={'disp': True}, tol=0.1)
result = minimize(lambda theta: objective(theta,df,lengths_a1,lengths_m1,lengths_d1,lengths_a2,lengths_m2,lengths_d2), [0.01]*1184, method='TNC', options={'disp': True, 'maxiter': 2}, tol=0.0001)


#%%

# check ordering
def get_norm_probs(a1_cond_probs, m1_cond_probs, d1_cond_probs,a2_cond_probs, m2_cond_probs, d2_cond_probs):
                
    a1_cond_probs_tot = [] 
    m1_cond_probs_tot = []
    d1_cond_probs_tot = []   
    a2_cond_probs_tot = []
    m2_cond_probs_tot = []
    d2_cond_probs_tot = [] 
    
    keys_1 = [(i,j,k,l,g) for i in range(1,7) for j in range(1,6) for k in range(1,6) for l in range(1,6) for g in range(1,6)]  
    keys_2 = [(i,j,k,l,g) for i in range(1,7) for j in range(1,7) for k in range(1,6) for l in range(1,6) for g in range(1,6)]  
    
    for i in range(6):  # Category 1 has 5 items
        a1_cond_probs_dict = dict(zip(keys_1, a1_cond_probs[i]))
        m1_cond_probs_dict = dict(zip(keys_1, m1_cond_probs[i]))
        a1_cond_probs_tot.append(a1_cond_probs_dict)
        m1_cond_probs_tot.append(m1_cond_probs_dict)
        
    for i in range(5):
        d1_cond_probs_dict = dict(zip(keys_2, d1_cond_probs[i]))
        a2_cond_probs_dict = dict(zip(keys_2, a2_cond_probs[i]))
        m2_cond_probs_dict = dict(zip(keys_2, m2_cond_probs[i]))
        d2_cond_probs_dict = dict(zip(keys_2, d2_cond_probs[i]))

        d1_cond_probs_tot.append(d1_cond_probs_dict)
        a2_cond_probs_tot.append(a2_cond_probs_dict)
        m2_cond_probs_tot.append(m2_cond_probs_dict)
        d2_cond_probs_tot.append(d2_cond_probs_dict)


    basket_probabilities = {}
    for i in range(6):  # Category 1 has 5 items
        for j in range(6):  # Category 2 has 5 items
            for k in range(5):  # Category 3 has 6 items
                for l in range(5):
                    for g in range(5):
                        for m in range(5):
                            basket = (i+1, j+1, k+1, l+1, g+1, m+1)  # [a1, m1, d1, a2, m2, d2]
                            m1_prob = m1_cond_probs_tot[basket[1]-1][(basket[0],basket[2], basket[3], basket[4], basket[5])]
                            a1_prob = a1_cond_probs_tot[basket[0]-1][(basket[1],basket[2], basket[3], basket[4], basket[5])]
                            d1_prob = d1_cond_probs_tot[basket[2]-1][(basket[0],basket[1], basket[3], basket[4], basket[5])]
                            m2_prob = m2_cond_probs_tot[basket[4]-1][(basket[0],basket[1], basket[2], basket[3], basket[5])]
                            a2_prob = a2_cond_probs_tot[basket[3]-1][(basket[0],basket[1], basket[2], basket[4], basket[5])]
                            d2_prob = d2_cond_probs_tot[basket[5]-1][(basket[0],basket[1], basket[2], basket[3], basket[4])]
                            tot = m1_prob + a1_prob + d1_prob + m2_prob + a2_prob + d2_prob
                            
                            m1_prob_norm = m1_prob / tot
                            a1_prob_norm = a1_prob / tot
                            d1_prob_norm = d1_prob / tot
                            m2_prob_norm = m2_prob / tot
                            a2_prob_norm = a2_prob / tot
                            d2_prob_norm = d2_prob / tot
                            
                            basket_probabilities[basket] = {
                                'm1_prob': m1_prob_norm,
                                'a1_prob': a1_prob_norm,
                                'd1_prob': d1_prob_norm,
                                'm2_prob': m2_prob_norm,
                                'a2_prob': a2_prob_norm,
                                'd2_prob': d2_prob_norm
                            }
                        # so m_prob is p(m=j | a=k, d=i) etc.
    return basket_probabilities

basket_probs = get_norm_probs(cond_probs_a1,cond_probs_m1,cond_probs_d1,cond_probs_a2,cond_probs_m2,cond_probs_d2)
# 22500

#%%

def opt_mvmnl(basket_probs3,max_fat = 200,max_carbs = 1000,max_protein = 400,max_salt = 2000,max_cals = 2000,min_fat = 0,min_carbs = 0,min_protein = 0):
    problem = LpProblem("Basket_Selection",LpMinimize)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    categories = {
        "App1": 6,
        "Main1": 6,
        "Dessert1": 5,
        "App2": 5,
        "Main2": 5,
        "Dessert2": 5
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
            
            objective_value += arrival_rate * item_vars[(day,'App1',basket[0])] * appetizer1_score[basket[0]-1] * basket_probs3[basket]['a1_prob']
            objective_value += arrival_rate * item_vars[(day,'Main1',basket[1])] * main1_score[basket[1]-1] * basket_probs3[basket]['m1_prob']
            objective_value += arrival_rate * item_vars[(day,'Dessert1',basket[2])] * dessert1_score[basket[2]-1] * basket_probs3[basket]['d1_prob']
            
            objective_value += arrival_rate * item_vars[(day,'App2',basket[3])] * appetizer2_score[basket[3]-1] * basket_probs3[basket]['a2_prob']
            objective_value += arrival_rate * item_vars[(day,'Main2',basket[4])] * main2_score[basket[4]-1] * basket_probs3[basket]['m2_prob']
            objective_value += arrival_rate * item_vars[(day,'Dessert2',basket[5])] * dessert2_score[basket[5]-1] * basket_probs3[basket]['d2_prob']
            
            total_fat += item_vars[(day,'App1',basket[0])] * appetizer1_fat[basket[0]-1]
            total_fat += item_vars[(day,'Main1',basket[1])] * main1_fat[basket[1]-1]
            total_fat += item_vars[(day,'Dessert1',basket[2])] * dessert1_fat[basket[2]-1]
            
            total_fat += item_vars[(day,'App2',basket[3])] * appetizer2_fat[basket[3]-1]
            total_fat += item_vars[(day,'Main2',basket[4])] * main2_fat[basket[4]-1]
            total_fat += item_vars[(day,'Dessert2',basket[5])] * dessert2_fat[basket[5]-1]
            
            total_carbs += item_vars[(day,'App1',basket[0])] * appetizer1_carbs[basket[0]-1]
            total_carbs += item_vars[(day,'Main1',basket[1])] * main1_carbs[basket[1]-1]
            total_carbs += item_vars[(day,'Dessert1',basket[2])] * dessert1_carbs[basket[2]-1]
            
            total_carbs += item_vars[(day,'App2',basket[3])] * appetizer2_carbs[basket[3]-1]
            total_carbs += item_vars[(day,'Main2',basket[4])] * main2_carbs[basket[4]-1]
            total_carbs += item_vars[(day,'Dessert2',basket[5])] * dessert2_carbs[basket[5]-1]
            
            total_protein += item_vars[(day,'App1',basket[0])] * appetizer1_protein[basket[0]-1]
            total_protein += item_vars[(day,'Main1',basket[1])] * main1_protein[basket[1]-1]
            total_protein += item_vars[(day,'Dessert1',basket[2])] * dessert1_protein[basket[2]-1]
            
            total_salt += item_vars[(day,'App1',basket[0])] * appetizer1_salt[basket[0]-1]
            total_salt += item_vars[(day,'Main1',basket[1])] * main1_salt[basket[1]-1]
            total_salt += item_vars[(day,'Dessert1',basket[2])] * dessert1_salt[basket[2]-1]
            
            total_calories += item_vars[(day,'App1',basket[0])] * appetizer1_calories[basket[0]-1]
            total_calories += item_vars[(day,'Main1',basket[1])] * main1_calories[basket[1]-1]
            total_calories += item_vars[(day,'Dessert1',basket[2])] * dessert1_calories[basket[2]-1]
            
            total_protein += item_vars[(day,'App2',basket[3])] * appetizer2_protein[basket[3]-1]
            total_protein += item_vars[(day,'Main2',basket[4])] * main2_protein[basket[4]-1]
            total_protein += item_vars[(day,'Dessert2',basket[5])] * dessert2_protein[basket[5]-1]
            
            total_salt += item_vars[(day,'App2',basket[3])] * appetizer2_salt[basket[3]-1]
            total_salt += item_vars[(day,'Main2',basket[4])] * main2_salt[basket[4]-1]
            total_salt += item_vars[(day,'Dessert2',basket[5])] * dessert2_salt[basket[5]-1]
            
            total_calories += item_vars[(day,'App2',basket[3])] * appetizer2_calories[basket[3]-1]
            total_calories += item_vars[(day,'Main2',basket[4])] * main2_calories[basket[4]-1]
            total_calories += item_vars[(day,'Dessert2',basket[5])] * dessert2_calories[basket[5]-1]
            
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
            problem += item_vars[(days[i], 'App1', item1)] + item_vars[(days[i + 1],'App1', item1)] <= 1        
    for item2 in range(1, 6+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'Main1', item2)] + item_vars[(days[i + 1],'Main1', item2)] <= 1        
    for item3 in range(1, 5+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'Dessert1', item3)] + item_vars[(days[i + 1],'Dessert1', item3)] <= 1 
    for item4 in range(1, 5+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'App2', item4)] + item_vars[(days[i + 1],'App2', item4)] <= 1        
    for item5 in range(1, 5+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'Main2', item5)] + item_vars[(days[i + 1],'Main2', item5)] <= 1        
    for item6 in range(1, 5+1):
        for i in range(len(days) - 1):
            problem += item_vars[(days[i], 'Dessert2', item6)] + item_vars[(days[i + 1],'Dessert2', item6)] <= 1 
    
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


items,item_vars,probem_ = opt_mvmnl(basket_probs,max_fat=600,max_carbs=900,max_protein=500,max_salt=13000,max_cals=9500)
