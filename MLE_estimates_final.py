# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:26:18 2023

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

#%%

df['Soup'].unique()

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

def new_calc_probs(df,name,length,initial_coeffs):
    
    coeffs = {
        'asc_0': 0,
        'asc_1': initial_coeffs[0],
        'asc_2': initial_coeffs[1],
        'asc_3': initial_coeffs[2],
        'asc_4': initial_coeffs[3],
        'asc_5': initial_coeffs[4],
        'asc_6': initial_coeffs[5],
        'b_price': initial_coeffs[6],
        'b_calories': initial_coeffs[7],
        'b_rating': initial_coeffs[8],
        'b_discount': initial_coeffs[9]
    }
    
    #coeffs = initial_coeffs
    
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
     
    log_lik = 0
    for i in range(length+1):
        if probs[i-1] == 0:
            log_lik += 0
        else: 
            log_lik += len(df.loc[df['Choice'] == i]) * np.log(probs[i-1])

    return log_lik

# Biogeme results
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

new_calc_probs(dessert2_df, 'D2', 5, coeffs_d2)
new_calc_probs(dessert1_df, 'D1', 5, coeffs_d1)
new_calc_probs(appetizer1_df, 'A1', 6, coeffs_a1)
new_calc_probs(appetizer2_df, 'A2', 5, coeffs_a2)
new_calc_probs(main1_df, 'MC1', 6, coeffs_m1)
new_calc_probs(main2_df, 'MC2', 5, coeffs_m2)
new_calc_probs(soup_df, 'S', 7, coeffs_s)

def new_objective(coeffs,df,name,length):
    # Call calc_probs with the dataframe and the current set of coefficients
    log_lik = new_calc_probs(df, name,length,coeffs)

    # Negate the log-likelihood since we want to maximize it
    return -log_lik

result = minimize(lambda theta: new_objective(theta,dessert1_df,'D1',5), [0.0015]*11, method='TNC', options={'max_iter':200},tol=1e-10)
result.x
result.fun

res_new_m2 = minimize(lambda theta: new_objective(theta,main2_df,'MC2',5), [0.015]*11, method='TNC', options={'max_iter':200},tol=1e-10)
res_new_m2.fun
res_new_m2.x

# etc. for all categories...

# only category I get different objective values using biogeme estimates is dessert 1 and appetizer (2?) not sure why