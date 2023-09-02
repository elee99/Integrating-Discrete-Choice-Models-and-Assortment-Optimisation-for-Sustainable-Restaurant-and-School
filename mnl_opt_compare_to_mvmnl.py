# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:39:33 2023

@author: elfre
"""

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
    non_nan_values = [val for val in data_list if not np.isnan(val)]
    
    average = np.mean(non_nan_values)
    
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

# want to get conditional probabilities over baskets instead, where a basket is one starter, one main and one dessert
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
  
m_probs = calc_probs(main1_df,coeffs_m1,'MC1',6)
a_probs = calc_probs(appetizer1_df,coeffs_a1,'A1',6)
d_probs = calc_probs(dessert2_df,coeffs_d2,'D2',5)

def get_cond_probs(m_probs,a_probs,d_probs):
    basket_probabilities = {}
    for i in range(6):  # Category 1 has 5 items
        for j in range(6):  # Category 2 has 5 items
            for k in range(5):  # Category 3 has 6 items
                basket = (i+1, j+1, k+1)
                m_prob = m_probs[basket[1]-1]
                a_prob = a_probs[basket[0]-1]
                d_prob = d_probs[basket[2]-1]
    
                tot = m_prob + a_prob + d_prob
    
                m_prob_norm = m_prob / tot
                a_prob_norm = a_prob / tot
                d_prob_norm = d_prob / tot
    
                basket_probabilities[basket] = {
                    'm_prob': m_prob_norm,
                    'a_prob': a_prob_norm,
                    'd_prob': d_prob_norm
                }
    return basket_probabilities

basket_probs = get_cond_probs(m_probs,a_probs,d_probs)

#%%

def opt_mnl(basket_probs3,max_fat = 200,max_carbs = 1000,max_protein = 400,max_salt = 2000,max_cals = 2000,min_fat = 0,min_carbs = 0,min_protein = 0,max_occur=2):
    problem = LpProblem("Basket_Selection",LpMinimize)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    categories = {
        "App": 6,
        "Main": 6,
        "Dessert": 5
    }
    
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
            objective_value = 0
            total_fat = 0
            total_protein = 0
            total_carbs = 0
            total_salt = 0
            total_calories = 0
            
            objective_value += arrival_rate * item_vars[(day,'App',basket[0])] * appetizer1_score[basket[0]-1] * basket_probs3[basket]['a_prob']
            objective_value += arrival_rate * item_vars[(day,'Main',basket[1])] * main1_score[basket[1]-1] * basket_probs3[basket]['m_prob']
            objective_value += arrival_rate * item_vars[(day,'Dessert',basket[2])] * dessert2_score[basket[2]-1] * basket_probs3[basket]['d_prob']
            
            total_fat += item_vars[(day,'App',basket[0])] * appetizer1_fat[basket[0]-1]
            total_fat += item_vars[(day,'Main',basket[1])] * main1_fat[basket[1]-1]
            total_fat += item_vars[(day,'Dessert',basket[2])] * dessert2_fat[basket[2]-1]
            
            total_carbs += item_vars[(day,'App',basket[0])] * appetizer1_carbs[basket[0]-1]
            total_carbs += item_vars[(day,'Main',basket[1])] * main1_carbs[basket[1]-1]
            total_carbs += item_vars[(day,'Dessert',basket[2])] * dessert2_carbs[basket[2]-1]
            
            total_protein += item_vars[(day,'App',basket[0])] * appetizer1_protein[basket[0]-1]
            total_protein += item_vars[(day,'Main',basket[1])] * main1_protein[basket[1]-1]
            total_protein += item_vars[(day,'Dessert',basket[2])] * dessert2_protein[basket[2]-1]
            
            total_salt += item_vars[(day,'App',basket[0])] * appetizer1_salt[basket[0]-1]
            total_salt += item_vars[(day,'Main',basket[1])] * main1_salt[basket[1]-1]
            total_salt += item_vars[(day,'Dessert',basket[2])] * dessert2_salt[basket[2]-1]
            
            total_calories += item_vars[(day,'App',basket[0])] * appetizer1_calories[basket[0]-1]
            total_calories += item_vars[(day,'Main',basket[1])] * main1_calories[basket[1]-1]
            total_calories += item_vars[(day,'Dessert',basket[2])] * dessert2_calories[basket[2]-1]
            
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
    
    for day in days:
        for cat, num_items in categories.items():
            problem += lpSum(item_vars[(day, cat, item)] for item in range(1, num_items + 1)) == 1
    
    #item_occurrences = LpVariable.dicts("Item_Occurrence", [(day, cat, item) for day in days for cat,num_items in categories.items() for item in range(1,num_items+1)], cat='Binary')
    max_item_occurrences = max_occur
    
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

items,item_vars,probem_ = opt_mnl(basket_probs,max_fat=60,max_carbs=90,max_protein=50,max_salt=1300,max_cals=950) # 7391980

"""
Dishes selected for App Mon:
FishCake /
Dishes selected for Main Mon:
PadThai /
Dishes selected for Dessert Mon:
Pudding /
Dishes selected for App Tue:
FriedCalamari /
Dishes selected for Main Tue:
PadKrapow /
Dishes selected for Dessert Tue:
VanillaIceCream /
Dishes selected for App Wed:
EggRolls / 
Dishes selected for Main Wed:
GreenChickenCurry /
Dishes selected for Dessert Wed:
Pudding / 
Dishes selected for App Thu:
FriedCalamari / 
Dishes selected for Main Thu:
PadKrapow / 
Dishes selected for Dessert Thu:
VanillaIceCream /
Dishes selected for App Fri:
FishCake /
Dishes selected for Main Fri:
GreenChickenCurry / 
Dishes selected for Dessert Fri:
ChocolateIcecream

differences of MVMNL menu:
apple pie
"""
items2,item_vars2,problem2 = opt_mnl(basket_probs,max_fat=60,max_carbs=90,max_protein=50,max_salt=1300,max_cals=950,max_occur=3)
items3,item_vars3,problem3 = opt_mnl(basket_probs,max_fat=100,max_carbs=150,max_protein=100,max_salt=2000,max_cals=1200)

#%%
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

def get_menu(items,item_vars,problem):
    categories = {
        "App": 6,
        "Main": 6,
        "Dessert": 5
    }
    # Create dictionaries to store the total nutritional values per category
    total_calories_per_cat = {cat: 0 for cat in categories.keys()}
    total_fat_per_cat = {cat: 0 for cat in categories.keys()}
    total_protein_per_cat = {cat: 0 for cat in categories.keys()}
    total_salt_per_cat = {cat: 0 for cat in categories.keys()}
    total_carbs_per_cat = {cat: 0 for cat in categories.keys()}
    total_price_per_cat = {cat: 0 for cat in categories.keys()}
    total_score_per_cat = {cat: 0 for cat in categories.keys()}
    
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        for cat in categories.keys():
            print(f"Dishes selected for {cat} {day}:")
            for dish_name, (dish_cat, dish_item) in dishes_dict.items():
                if cat == dish_cat:
                    if value(item_vars[(day, dish_cat, dish_item)]) == 1:
                        print(f"{dish_name}")
                        
                        dish_calories = calories_dict[dish_name]
                        dish_fat = fat_dict[dish_name]
                        dish_carbs = carbs_dict[dish_name]
                        dish_protein = protein_dict[dish_name]
                        dish_salt = salt_dict[dish_name]
                        dish_score = scores_dict[dish_name]
                        
                        total_calories_per_cat[cat] += dish_calories
                        total_fat_per_cat[cat] += dish_fat
                        total_protein_per_cat[cat] += dish_protein
                        total_salt_per_cat[cat] += dish_salt
                        total_carbs_per_cat[cat] += dish_carbs
                        total_score_per_cat[cat] += dish_score
    
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
        
    return total_calories_per_cat, total_fat_per_cat, total_protein_per_cat, total_salt_per_cat, total_carbs_per_cat, total_score_per_cat, total_price_per_cat

total_calories_per_cat, total_fat_per_cat, total_protein_per_cat, total_salt_per_cat, total_carbs_per_cat, total_score_per_cat, total_price_per_cat = get_menu(items, item_vars, probem_)

def get_optimisation(items,item_vars,problem):
    total_calories_per_cat, total_fat_per_cat, total_protein_per_cat, total_salt_per_cat, total_carbs_per_cat, total_score_per_cat, total_price_per_cat = get_menu(items, item_vars, probem_)

    opt_app = [total_fat_per_cat['App']*9,total_protein_per_cat['App']*4,total_carbs_per_cat['App']*4]
    opt_main = [total_fat_per_cat['Main']*9,total_protein_per_cat['Main']*4,total_carbs_per_cat['Main']*4]
    opt_dess = [total_fat_per_cat['Dessert']*9,total_protein_per_cat['Dessert']*4,total_carbs_per_cat['Dessert']*4]

    optimization = [opt_app, opt_main, opt_dess]
    
    return optimization

optimization1 = get_optimisation(items, item_vars, probem_) # normal
optimization2 = get_optimisation(items2, item_vars2, problem2) # allowing items to occur three times
optimization3 = get_optimisation(items3, item_vars3, problem3) # relaxed constraints

optimization1, opt1_score = get_optimisation(items, item_vars, probem_) # normal
optimization2, opt2_score = get_optimisation(items2, item_vars2, problem2) # allowing items to occur three times
optimization3, opt3_score = get_optimisation(items3, item_vars3, problem3) # relaxed constraints

def get_optimisation(items,item_vars,problem):
    total_calories_per_cat, total_fat_per_cat, total_protein_per_cat, total_salt_per_cat, total_carbs_per_cat, total_score_per_cat, total_price_per_cat = get_menu(items, item_vars, probem_)

    opt_app = [total_fat_per_cat['App']*9,total_protein_per_cat['App']*4,total_carbs_per_cat['App']*4]
    opt_main = [total_fat_per_cat['Main']*9,total_protein_per_cat['Main']*4,total_carbs_per_cat['Main']*4]
    opt_dess = [total_fat_per_cat['Dessert']*9,total_protein_per_cat['Dessert']*4,total_carbs_per_cat['Dessert']*4]

    optimization = [opt_app, opt_main, opt_dess]
    
    opt_score = [total_score_per_cat['App'],total_score_per_cat['Main'],total_score_per_cat['Dessert']]
    
    return optimization, opt_score

optimization1, opt1_score = get_optimisation(items, item_vars, probem_) # normal
optimization2, opt2_score = get_optimisation(items2, item_vars2, problem2) # allowing items to occur three times
optimization3, opt3_score = get_optimisation(items3, item_vars3, problem3) # relaxed constraints

fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width
bar_width = 0.25

# Calculate the positions for the bars
bar_positions = np.arange(3)

# Create the bar plots for each optimization
bar1 = ax.bar(bar_positions, opt1_score, bar_width, label='Optimization 1', color='skyblue')
bar2 = ax.bar(bar_positions + bar_width, opt2_score, bar_width, label='Optimization 2', color='lightgreen')
bar3 = ax.bar(bar_positions + bar_width*2, opt3_score, bar_width, label='Optimization 3', color='purple')

# Set labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Total Score by Category')
ax.set_title('Scores by Category - MVMNL')
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(['App','Main','Dessert'])
ax.legend()

plt.tight_layout()
plt.savefig('opts_mnl.pdf',bbox_inches='tight',format='pdf')
plt.show()

#%%
categories = ['Fat', 'Protein', 'Carbs']
num_categories = len(categories)
dish_categories = ['Appetizer','Main','Dessert']

# Create subplots for each dish category
fig, axes = plt.subplots(nrows=1, ncols=len(optimization1), figsize=(15, 6))

# Loop through each dish category and create radar charts
for i, opt1_data in enumerate(optimization1):
    ax = axes[i]

    # Compute angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

    # Extract data for the current dish category and optimization
    data_opt1 = np.array(opt1_data) / sum(optimization1[i])
    data_opt2 = np.array(optimization2[i]) / sum(optimization2[i])
    data_opt3 = np.array(optimization3[i]) / sum(optimization3[i])
   # data_opt4 = np.array(optimization4[i]) / sum(optimization4[i])

    # Add the data to the radar chart
    ax.plot(angles, data_opt1, label='Optimization 1')
    ax.plot(angles, data_opt2, label='Optimization 2')
    ax.plot(angles, data_opt3, label='Optimization 3')
   # ax.plot(angles, data_opt4, label='Optimization 4')

    # Fill the area under the lines
    ax.fill(angles, data_opt1, alpha=0.25)
    ax.fill(angles, data_opt2, alpha=0.25)
    ax.fill(angles, data_opt3, alpha=0.25)
   # ax.fill(angles, data_opt4, alpha=0.25)

    # Set the labels for each axis
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_title(dish_categories[i])  # Set subplot title
    ax.legend()

# Adjust layout
plt.tight_layout()

results = []
for i in range(5,400,5):
    result, item_vars, problem = opt_mnl(basket_probs,max_fat=60+i,max_carbs=90+i,max_protein=50+i,max_salt=1300+i,max_cals=950+i)
    objective_value = value(problem.objective)
    results.append((result, item_vars, problem, objective_value))
    
for idx, (result, item_vars, problem, objective_value) in enumerate(results):
    print(f"Solution {idx+1}: Objective Value = {objective_value}")

objective_values = [objective_value for _, _, _, objective_value in results]

plt.figure(figsize=(10, 6))
plt.plot(objective_values)

plt.title('Objective Value vs. Constraint Values')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.grid(True)
plt.savefig('obj_vals_mnl.pdf',bbox_inches='tight',format='pdf')
plt.show()

#%%

# results from MV-MNL (did same three optimizations)

# fat, protein, carbs of app, main, dessert
# opt 1
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

data_opt1_overall = []
data_opt2_overall = []
for i in range(0,3):
    data_opt1_overall.append([optimization1[i][j]/sum(optimization1[i]) for j in range(0,3)])
    data_opt2_overall.append([optimization1_mvmnl[i][j]/sum(optimization1_mvmnl[i]) for j in range(0,3)])

fig, axes = plt.subplots(nrows=1, ncols=len(optimization1), figsize=(15, 6))

categories
dish_categories = ['Appetizer','Main','Dessert']
# Loop through each dish category and create radar charts
for i, opt1_data in enumerate(optimization1):
    ax = axes[i]

    # Compute angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

    # Extract data for the current dish category and optimization
   # data_opt1 = np.array(opt1_data) / sum(optimization1[i])
   # data_opt2 = np.array(optimization1_mvmnl[i]) / sum(optimization1_mvmnl[i])

    # Add the data to the radar chart
    ax.plot(angles, data_opt1_overall[i], label='Optimization 1 - MNL')
    ax.plot(angles, data_opt2_overall[i], label='Optimization 1 - MVMNL')

    # Fill the area under the lines
    ax.fill(angles, data_opt1_overall[i], alpha=0.25)
    ax.fill(angles, data_opt2_overall[i], alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_title(dish_categories[i])  # Set subplot title
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.savefig('opt1_comparison.pdf', bbox_inches='tight',format='pdf')
plt.show()
################

data_opt1_overall = []
data_opt2_overall = []
for i in range(0,3):
    data_opt1_overall.append([optimization2[i][j]/sum(optimization2[i]) for j in range(0,3)])
    data_opt2_overall.append([optimization2_mvmnl[i][j]/sum(optimization2_mvmnl[i]) for j in range(0,3)])

fig, axes = plt.subplots(nrows=1, ncols=len(optimization1), figsize=(15, 6))

dish_categories = ['Appetizer','Main','Dessert']
# Loop through each dish category and create radar charts
for i, opt1_data in enumerate(optimization1):
    ax = axes[i]

    # Compute angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

    # Extract data for the current dish category and optimization
    #data_opt1 = np.array(opt1_data) / sum(optimization2[i])
    #data_opt2 = np.array(optimization2_mvmnl[i]) / sum(optimization2_mvmnl[i])

    ax.plot(angles, data_opt1_overall[i], label='Optimization 2 - MNL')
    ax.plot(angles, data_opt2_overall[i], label='Optimization 2 - MVMNL')

    # Fill the area under the lines
    ax.fill(angles, data_opt1_overall[i], alpha=0.25)
    ax.fill(angles, data_opt2_overall[i], alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_title(dish_categories[i])  # Set subplot title
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.savefig('opt2_comparison.pdf', bbox_inches='tight',format='pdf')
plt.show()

###############

data_opt1_overall = []
data_opt2_overall = []
for i in range(0,3):
    data_opt1_overall.append([optimization3[i][j]/sum(optimization3[i]) for j in range(0,3)])
    data_opt2_overall.append([optimization3_mvmnl[i][j]/sum(optimization3_mvmnl[i]) for j in range(0,3)])

fig, axes = plt.subplots(nrows=1, ncols=len(optimization1), figsize=(15, 6))

dish_categories = ['Appetizer','Main','Dessert']
# Loop through each dish category and create radar charts
for i, opt1_data in enumerate(optimization3):
    ax = axes[i]

    # Compute angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

    # Extract data for the current dish category and optimization
    #data_opt1 = np.array(opt1_data) / sum(optimization3[i])
    #data_opt2 = np.array(optimization3_mvmnl[i]) / sum(optimization3_mvmnl[i])
    
    # Add the data to the radar chart
    ax.plot(angles, data_opt1_overall[i], label='Optimization 3 - MNL')
    ax.plot(angles, data_opt2_overall[i], label='Optimization 3 - MVMNL')

    # Fill the area under the lines
    ax.fill(angles, data_opt1_overall[i], alpha=0.25)
    ax.fill(angles, data_opt2_overall[i], alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_title(dish_categories[i])  # Set subplot title
    
    ax.legend()
    
# Adjust layout
plt.tight_layout()
plt.savefig('opt3_comparison.pdf', bbox_inches='tight',format='pdf')
plt.show()

############
fig, axes = plt.subplots(nrows=1, ncols=len(optimization1), figsize=(15, 6))

i = 0
ax = axes[i]

# Compute angles for each category
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

# Extract data for the current dish category and optimization
#data_opt1 = np.array(opt1_data) / sum(optimization3[i])
#data_opt2 = np.array(optimization3_mvmnl[i]) / sum(optimization3_mvmnl[i])

# Add the data to the radar chart
ax.plot(angles,  [0.48103394540863065, 0.3364388947550312+0.01, 0.18252715983633824], label='Optimization 3 - MNL')
ax.plot(angles, [0.48103394540863065, 0.5, 0.18252715983633824], label='Optimization 3 - MVMNL')

# Fill the area under the lines
ax.fill(angles, data_opt1_overall[0], alpha=0.25)
ax.fill(angles, data_opt2_overall[0], alpha=0.25)

ax.set_xticks(angles)
ax.set_xticklabels(categories)
ax.set_title(dish_categories[i])  # Set subplot title
ax.legend()
plt.show()

opt1_score_mvmnl = [2023.8325414549595, 2600.9318654898057, 1544.9851543920515]
opt2_score_mvmnl = [1554.4890009627732, 1559.5081412475015, 1350.491294117647]
opt3_score_mvmnl = [2023.8325414549595, 2080.2200033686536, 1157.0512199592258]
opt1_score
opt2_score
opt3_score

fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar width
bar_width = 0.125

# Calculate the positions for the bars
bar_positions = np.arange(3)

# Create the bar plots for each optimization
bar1 = ax.bar(bar_positions, opt1_score, bar_width, label='Optimization 1 - MNL', color='skyblue')
bar2 = ax.bar(bar_positions + bar_width, opt2_score, bar_width, label='Optimization 2 - MNL', color='lightgreen')
bar3 = ax.bar(bar_positions + bar_width*2, opt3_score, bar_width, label='Optimization 3 - MNL', color='purple')
bar1 = ax.bar(bar_positions + bar_width*3, opt1_score_mvmnl, bar_width, label='Optimization 1 - MVMNL', color='cornflowerblue')
bar2 = ax.bar(bar_positions + bar_width*4, opt2_score_mvmnl, bar_width, label='Optimization 2 - MVMNL', color='green')
bar3 = ax.bar(bar_positions + bar_width*5, opt3_score_mvmnl, bar_width, label='Optimization 3 - MVMNL', color='pink')

# Set labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Total Score by Category')
ax.set_title('Scores by Category')
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(['App','Main','Dessert'])
ax.legend()

plt.tight_layout()
plt.savefig('scores_compared.pdf',bbox_inches='tight',format='pdf')
plt.show()

obj_vals_mvmnl = [6148178.808505385,
 6148178.808505385,
 6070285.14418362,
 6070285.14418362,
 6070285.14418362,
 5797683.064271391,
 5797683.064271391,
 5797683.064271391,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006,
 5654014.607132006]


# Plotting the first subplot
plt.figure(figsize=(15, 6))

# First Subplot
plt.subplot(1, 2, 1)
plt.plot(objective_values)  # Use your actual data for objective_values1
plt.title('Objective values - MNL')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.grid(True)

# Second Subplot
plt.subplot(1, 2, 2)
plt.plot(obj_vals_mvmnl)  # Use your actual data for objective_values2
plt.title('Objective values - MVMNL')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.grid(True)

plt.tight_layout()
plt.savefig('obj_vals_compare.pdf', bbox_inches='tight',format='pdf')
plt.show()

