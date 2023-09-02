# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:47:26 2023

@author: elfre
"""
# This is way I have used in the report (estimates are from this) so need to update explanation
# Basically havent used proportions 

from time import gmtime, strftime
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

df[['Appetizer 1', 'A1 Price', 'A1 Rating', 'A1 Calories', 'A1 Discount']
   ] = df['Appertizer1'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Appetizer 2', 'A2 Price', 'A2 Rating', 'A2 Calories', 'A2 Discount']
   ] = df['Appetizer2'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Main Course 1', 'MC1 Price', 'MC1 Rating', 'MC1 Calories', 'MC1 Discount']
   ] = df['MainCourse1'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Main Course 2', 'MC2 Price', 'MC2 Rating', 'MC2 Calories', 'MC2 Discount']
   ] = df['MainCourse2'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Soup', 'S Price', 'S Rating', 'S Calories', 'S Discount']
   ] = df['Soup'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Dessert 1', 'D1 Price', 'D1 Rating', 'D1 Calories', 'D1 Discount']
   ] = df['Dessert1'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')
df[['Dessert 2', 'D2 Price', 'D2 Rating', 'D2 Calories', 'D2 Discount']
   ] = df['Dessert2'].str.extract(r'([^()]+)\s*\(([^,]+),([^,]+),([^,]+),([^)]+)\)')

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
    'OnionSoup': 3,
    'TomYumKung': 4,
    'TomKhaGai': 5,
    'LengZab': 6,
    'GlassNoodlesSoup': 7}

dessert_1_dishes = {
    'CoconutCustard': 1,
    'KhanomMawKaeng': 2,
    'MangoStickyRice': 3,
    'LodChongThai': 4,
    'KanomKrok': 5}

dessert2_dishes = {
    'Pudding': 1,
    'VanillaIceCream': 2,
    'ApplePie': 3,
    'ChocolateCake': 4,
    'ChocolateIcecream': 5}

appetizer1_dishes = {
    'FriedCalamari': 1,
    'FriedCalamari ': 1,
    'EggRolls': 2,
    'ShrimpCake': 3,
    'FishCake': 4,
    'FishCake ': 4,
    'FriedShrimpBall': 5,
    'HerbChicken': 6
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
    'ShrimpFriedRice': 6
}

main2_dishes = {
    'ChickenPorridge': 1,
    'KanomJeenNamYa': 2,
    'ShrimpGlassNoodles': 3,
    'AmericanFriedRice': 4,
    'SausageFriedRice': 5
}


def create_df(name, full_name, dishes_dict,):
    specific_df = df[[f'{name} Price', f'{name} Rating', f'{name} Calories', f'{name} Discount', f'{full_name}',
                      'Payment', 'Gender']]

    specific_df = specific_df.dropna()

    # add new columns with choice1 calories/price/rating etc?
    specific_df['Choice'] = specific_df[f'{full_name}'].replace(dishes_dict)

    specific_df['Gender'] = specific_df['Gender'].replace({
        'Female': 0,
        'Male': 1})

    specific_df['Payment'] = specific_df['Payment'].replace({
        'Cash': 0,
        'Cashless': 1})

    specific_df['Choice'] = specific_df['Choice'].astype(int)

    #specific_df.drop(columns=[f'{name} Price', f'{name} Rating', f'{name} Calories', f'{name} Discount', f'{full_name}'], inplace=True)

    return specific_df


dessert1_df = create_df('D1', 'Dessert 1', dessert_1_dishes)
dessert2_df = create_df('D2', 'Dessert 2', dessert2_dishes)
appetizer1_df = create_df('A1', 'Appetizer 1', appetizer1_dishes)
appetizer2_df = create_df('A2', 'Appetizer 2', appetizer2_dishes)
main1_df = create_df('MC1', 'Main Course 1', main1_dishes)
main2_df = create_df('MC2', 'Main Course 2', main2_dishes)
soup_df = create_df('S', 'Soup', soup_dishes)

# %%

def calc_df(df):
    n = len(df)
    p = 0.1
    z = np.random.choice([0, 1], size=n, p=[1 - p, p])
    df['Z'] = z
    df['prob'] = np.where(df['Z'] == 0, 0.9, 0.1)

    # need to calc p(z=1|theta)*p(y|z=1,theta) to get h(z=1|y,theta)
    # p(z=1|theta) = 0.1
    # need to calc p(z=0|theta)*p(y|z=0,theta) to get h(z=0|y,theta)
    # p(z=0|theta) = 0.9

    df_0 = df.loc[df['Z'] == 0]
    df_1 = df.loc[df['Z'] == 1]

    return df_0, df_1, df

# calculating p(y|z,theta)
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
    
    return probs

# Calculating h(z|y,theta) 
def calc_h(df_0,df_1,name,length,coeffs):
    test_0 = new_calc_probs(df_0, name,length,coeffs) 
    test_1 = new_calc_probs(df_1, name,length,coeffs) 
    """
    new_t_0 = [i * 0.9 for i in test_0]
    new_t_1 = [i * 0.1 for i in test_1]

    h_z0 = []
    h_z1 = []
    for i in range(0,length):
        h_z0.append(new_t_0[i] / test_0[i])
        h_z1.append(new_t_1[i] / test_1[i])
    """     
    d1_probs_z1 = [i/2 for i in test_1]
    d1_probs_z0 = [i/2 for i in test_0]
    h_z0 = []
    h_z1 = []
    for i in range(0, length):
        h_z0.append(d1_probs_z1[i]*0.9 / (d1_probs_z0[i]+d1_probs_z1[i])) # p(y|z=1,theta)*p(z=1,theta)
        h_z1.append(d1_probs_z0[i]*0.1 / (d1_probs_z0[i]+d1_probs_z1[i]))
    
    # normalise so h(z=0|...) + h(z=1|...) sums to 1
    norm = sum(h_z0)+sum(h_z1)
    h_z0 = [i/norm for i in h_z0]
    h_z1 = [i/norm for i in h_z1]
    
    return h_z0,h_z1

df_0,df_1,none = calc_df(main1_df)
f = calc_h(df_0,df_1,'MC1',6,[0.01]*12)
sum(f[0]) + sum(f[1])

def objective_new(theta, h_z0, h_z1, df0, df1, df, length, name, lengths_0, lengths_1, epsilon, em_prob0,em_prob1):

    coeffs = {
        'asc_0': 0,
        'asc_1': theta[0],
        'asc_2': theta[1],
        'asc_3': theta[2],
        'asc_4': theta[3],
        'asc_5': theta[4],
        'asc_6': theta[5],
        'b_price': theta[6],
        'b_calories': theta[7],
        'b_rating': theta[8],
        'b_discount': theta[9],
      #  'b_gender': theta[10],
      #  'b_payment': theta[11]
    }
    
    utilities0 = []
    
    df0[f'{name} Calories'] = df0[f'{name} Calories'].astype(float)
    df0[f'{name} Price'] = df0[f'{name} Price'].astype(float)
    df0[f'{name} Rating'] = df0[f'{name} Rating'].astype(float)
    df0[f'{name} Discount'] = df0[f'{name} Discount'].astype(float)
    
    for k in range(1,length+1):
        choice_df = df0.loc[df0['Choice']==k]
        utilities0.append(coeffs[f'asc_{k-1}'] + coeffs['b_price'] * choice_df[f'{name} Price'].iloc[0] + coeffs['b_calories'] * choice_df[f'{name} Calories'].iloc[0] + coeffs['b_discount'] * choice_df[f'{name} Discount'].iloc[0] + coeffs['b_rating'] * choice_df[f'{name} Rating'].iloc[0])
    
    tot = 0
    for utility in utilities0:
        tot += np.exp(utility)
    
    probs = []
    for utility in utilities0:
        probs.append(np.exp(utility)/tot)
        
    utilities1 = []
    
    df1[f'{name} Calories'] = df1[f'{name} Calories'].astype(float)
    df1[f'{name} Price'] = df1[f'{name} Price'].astype(float)
    df1[f'{name} Rating'] = df1[f'{name} Rating'].astype(float)
    df1[f'{name} Discount'] = df1[f'{name} Discount'].astype(float)
    
    for k in range(1,length+1):
        choice_df1 = df1.loc[df1['Choice']==k]
        utilities1.append(coeffs[f'asc_{k-1}'] + coeffs['b_price'] * choice_df1[f'{name} Price'].iloc[0] + coeffs['b_calories'] * choice_df1[f'{name} Calories'].iloc[0] + coeffs['b_discount'] * choice_df1[f'{name} Discount'].iloc[0] + coeffs['b_rating'] * choice_df1[f'{name} Rating'].iloc[0])
    
    tot1 = 0
    for utility in utilities1:
        tot1 += np.exp(utility)
    
    probs1 = []
    for utility in utilities1:
        probs1.append(np.exp(utility)/tot1)
        

    # normalise so add to 1?
    probs_0 = [i/2 for i in probs]
    probs_1 = [i/2 for i in probs1]
    
    #total_0 = [h_z0[i] * np.log(probs_0[i]*em_prob0 + epsilon) for i in range(len(h_z0))]
    #total_1 = [h_z1[i]  * np.log(probs_1[i]*em_prob1 + epsilon) for i in range(len(h_z1))]
    total_0 = [h_z0[i] * lengths_0[i] * np.log(probs_0[i]*em_prob0 + epsilon) for i in range(len(h_z0))]
    total_1 = [h_z1[i] * lengths_1[i] * np.log(probs_1[i]*em_prob1 + epsilon) for i in range(len(h_z1))]

    return sum(total_0)+sum(total_1)

def em_algorithm(initial_values, prev_fun, max_iter, tol, df, name, length,method,em_prob0,em_prob1,epsilon):
    iteration = 0 
    
    df_0, df_1, dessert_df = calc_df(df)
    # getting length of df for each alternative
    lengths_0 = []
    for i in range(1, length+1):
        lengths_0.append(len(df_0.loc[df_0['Choice'] == i]))

    lengths_1 = []
    for i in range(1, length+1):
        lengths_1.append(len(df_1.loc[df_1['Choice'] == i]))
        
    for j in range(1, max_iter):
        iteration += 1
        
        # getting h(z|y,theta) probs
        h_z0,h_z1 = calc_h(df_0,df_1,name,length,initial_values)

        # minimising objective func 
        result = minimize(lambda theta: -objective_new(theta, h_z0, h_z1, df_0, df_1, df, length, name, lengths_0, lengths_1, epsilon, em_prob0,em_prob1), initial_values, method=method)
        
        optimized_theta = result.x
        initial_values = optimized_theta

        print(f"Iteration {iteration}, Objective Value: {result.fun}")
        if prev_fun is not None and abs(prev_fun - result.fun) < tol:
            break  

        prev_fun = result.fun
        np.save('em_results', result.fun)

    return optimized_theta,result.fun

em_algorithm([0.001]*12, float('Inf'), 8, 1e-8, main1_df, 'MC1', 6, 'TNC', 0.9, 0.1, 1e-10)

r = new_calc_probs(main1_df, 'MC1', 6, [0.01]*12)
log_lik = 0
for i in range(0,6):
    log_lik += np.log(r[i])
# -15.29 

#%%


starting_values = [
    [0.001]*12, [0.01]*12, [0.005]*12, [0.05]*12, [0.1]*12,
    [0.0001]*12, [0.0005]*12, [0.2]*12, [0.5]*12, [1]*12
]

def final_em_estimates(starting_values,df,name,length,em_prob0,em_prob1,epsilon, max_iter=10,tol=1e-5):
    num_runs = len(starting_values)

    final_theta = []
    final_res = []
    for i in range(num_runs):
        optimized_theta,result_fun = em_algorithm(starting_values[i], float('inf'), max_iter, tol, df, name, length, 'TNC',em_prob0,em_prob1,epsilon)
        final_theta.append(optimized_theta)
        final_res.append(result_fun)
    
    ind = final_res.index(min(final_res))
    
    print(final_res)
    return final_res[ind],final_theta[ind],starting_values[ind]

a1_fun,a1_est,a1_vals = final_em_estimates(starting_values, appetizer1_df, 'A1', 6,0.9,0.1,1e-10) # done
                         
a2_fun,a2_est,a2_vals = final_em_estimates(starting_values, appetizer2_df, 'A2', 5,0.9,0.1,1e-10) # done

m1_fun,m1_est,m1_vals = final_em_estimates(starting_values, main1_df, 'MC1', 6,0.9,0.1,1e-10) # done

m2_fun,m2_est,m2_vals = final_em_estimates(starting_values, main2_df, 'MC2', 5, 0.9,0.1,1e-10)

d1_fun,d1_est,d1_vals = final_em_estimates(starting_values, dessert1_df, 'D1', 5,0.9,0.1,1e-10) # done

d2_fun,d2_est,d2_vals = final_em_estimates(starting_values, dessert2_df, 'D2', 5,0.9,0.1,1e-10) # done

s_fun,s_est,s_vals = final_em_estimates(starting_values, soup_df, 'S', 7, 0.9,0.1,1e-10) # done

coeffs_list = [a1_est,a2_est,m1_est,m2_est,d1_est,d2_est,s_est]

# calculating probs with the em coeffs and seeing if LL is the same as biogeme

probs_test = new_calc_probs(appetizer1_df, 'A1', 6, a1_est)
log_like = 0
for i in range(1,6+1):
    log_like += len(appetizer1_df.loc[appetizer1_df['Choice'] == i])*np.log(probs_test[i-1]) 
log_like
# 22594.91, same 

probs_m1 = new_calc_probs(main1_df, 'MC1', 6, m1_est)
log_like_m1 = 0 
for i in range(1,6+1):
    log_like_m1 += len(main1_df.loc[main1_df['Choice'] == i])*np.log(probs_m1[i-1]) 
log_like_m1  # 22599.379, same

probs_m2 = new_calc_probs(main2_df, 'MC2', 5, m2_est)
log_like_m2 = 0 
for i in range(1,6):
    log_like_m2 += len(main2_df.loc[main2_df['Choice'] == i])*np.log(probs_m2[i-1]) 
log_like_m2 # 20255.04, same

probs_a2 = new_calc_probs(appetizer2_df, 'A2', 5, a2_est)
log_like_a2 = 0 
for i in range(1,5+1):
    log_like_a2 += len(appetizer2_df.loc[appetizer2_df['Choice'] == i])*np.log(probs_a2[i-1]) 
log_like_a2 # 20276.81

probs_d1 = new_calc_probs(dessert1_df, 'D1', 5, d1_est)
log_like_d1 = 0 
for i in range(1,5+1):
    log_like_d1 += len(dessert1_df.loc[dessert1_df['Choice'] == i])*np.log(probs_d1[i-1]) 
log_like_d1 # 12426.87, slightly diff (more negative) than biogeme

probs_d2 = new_calc_probs(dessert2_df, 'D2', 5, d2_est)
log_like_d2 = 0 
for i in range(1,5+1):
    log_like_d2 += len(dessert2_df.loc[dessert2_df['Choice'] == i])*np.log(probs_d2[i-1]) 
log_like_d2 # 16081, same

probs_s = new_calc_probs(soup_df, 'S', 7, s_est)
log_like_s = 0 
for i in range(1,7+1):
    log_like_s += len(soup_df.loc[soup_df['Choice'] == i])*np.log(probs_s[i-1]) 
log_like_s # 21826.71. Same as Biogeme

coeffs_list = [a1_est,a2_est,m1_est,m2_est,d1_est,d2_est,s_est]

coeffs_df = pd.DataFrame(coeffs_list)

coeffs_df['category'] = ['Appetizer 1','Appetizer 2','Main 1','Main 2','Dessert 1','Dessert 2','Soup']

coeffs_df.set_index('category', inplace=True)

obj_df = pd.DataFrame([a1_fun,a2_fun,m1_fun,m2_fun,d1_fun,d2_fun,s_fun])
obj_df['Starting Values'] = [a1_vals,a2_vals,m1_vals,m2_vals,d1_vals,d2_vals,s_vals]
obj_df['category'] = ['Appetizer 1','Appetizer 2','Main 1','Main 2','Dessert 1','Dessert 2','Soup']

coeffs_df.to_csv('em_coefficients_final.csv', index=True) 
obj_df.to_csv('em_obj_vals_final.csv',index=True)

probs_em = pd.DataFrame([probs_test,probs_a2,probs_m1,probs_m2,probs_d1,probs_d2,probs_s])
probs_em['category'] = ['Appetizer 1','Appetizer 2','Main 1','Main 2','Dessert 1','Dessert 2','Soup']

probs_em.to_csv('probs_em.csv',index=True)

ll_em = pd.DataFrame([log_like,log_like_a2,log_like_m1,log_like_m2,log_like_d1,log_like_d2,log_like_s])
ll_em['category'] = ['Appetizer 1','Appetizer 2','Main 1','Main 2','Dessert 1','Dessert 2','Soup']

ll_em.to_csv('ll_em.csv',index=True)
