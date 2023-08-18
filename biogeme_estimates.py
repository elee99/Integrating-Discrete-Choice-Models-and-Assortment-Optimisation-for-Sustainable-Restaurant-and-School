# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:52:27 2023

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

#%%
# Main Course 1

mc1_df = df[['MC1 Price', 'MC1 Rating', 'MC1 Calories', 'MC1 Discount', 'Main Course 1',
               'Payment','Gender']]

mc1_df = mc1_df.dropna()

# add new columns with choice1 calories/price/rating etc?
mc1_df['Choice'] = mc1_df['Main Course 1'].replace({
    'PadKrapow': 1,
    'PadThai': 2,
    'HainaneseChickenRice': 3,
    'GreenChickenCurry': 4,
    'ShrimpStickyRice': 5,
    'ShrimpFriedRice':6
    # Add more mappings for other main course choices
})

mc1_df['Gender'] = mc1_df['Gender'].replace({
    'Female':0,
    'Male':1})

mc1_df['Payment'] = mc1_df['Payment'].replace({
    'Cash':0,
    'Cashless':1})

mc1_df['Price 1'] = 35
mc1_df['Price 2'] = 90
mc1_df['Price 3'] = 80
mc1_df['Price 4'] = 60
mc1_df['Price 5'] = 60
mc1_df['Price 6'] = 80

mc1_df['Calories 1'] = 372
mc1_df['Calories 2'] = 486
mc1_df['Calories 3'] = 597
mc1_df['Calories 4'] = 240
mc1_df['Calories 5'] = 477
mc1_df['Calories 6'] = 289

mc1_df['Rating 1'] = 5
mc1_df['Rating 2'] = 5
mc1_df['Rating 3'] = 3.5
mc1_df['Rating 4'] = 2.4
mc1_df['Rating 5'] = 2.8
mc1_df['Rating 6'] = 4.3

mc1_df['Discount 1'] = 0
mc1_df['Discount 2'] = 0
mc1_df['Discount 3'] = 0
mc1_df['Discount 4'] = 0
mc1_df['Discount 5'] = 20
mc1_df['Discount 6'] = 0

mc1_df['AV'] = 1

mc1_df['Choice'] = mc1_df['Choice'].astype(int)

mc1_df.drop(columns=['MC1 Price','MC1 Rating','MC1 Discount','MC1 Calories','Main Course 1'], inplace=True)

#%%
# BIOGEME FOR MAINS
database_mc1 = db.Database('mc1', mc1_df)
Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')
Price_6 = Variable('Price 6')


Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')
Calories_6 = Variable('Calories 6')


Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')
Rating_6 = Variable('Rating 6')


Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')
Discount_6 = Variable('Discount 6')

Choice = Variable('Choice')
AV = Variable('AV')

Gender = Variable('Gender')
Payment = Variable('Payment')

Gender_1 = database_mc1.DefineVariable('Gender_1', Gender * (Choice == 1))
Gender_2 = database_mc1.DefineVariable('Gender_2', Gender * (Choice == 2))
Gender_3 = database_mc1.DefineVariable('Gender_3', Gender * (Choice == 3))
Gender_4 = database_mc1.DefineVariable('Gender_4', Gender * (Choice == 4))
Gender_5 = database_mc1.DefineVariable('Gender_5', Gender * (Choice == 5))
Gender_6 = database_mc1.DefineVariable('Gender_6', Gender * (Choice == 6))

Payment_1 = database_mc1.DefineVariable('Payment_1', Payment * (Choice == 1))
Payment_2 = database_mc1.DefineVariable('Payment_2', Payment * (Choice == 2))
Payment_3 = database_mc1.DefineVariable('Payment_3', Payment * (Choice == 3))
Payment_4 = database_mc1.DefineVariable('Payment_4', Payment * (Choice == 4))
Payment_5 = database_mc1.DefineVariable('Payment_5', Payment * (Choice == 5))
Payment_6 = database_mc1.DefineVariable('Payment_6', Payment * (Choice == 6))

# params
asc_1 = Beta ('asc_1', 0, None , None , 1)
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)
asc_6 = Beta ('asc_6', 0, None , None , 0)

b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)
b_gender = Beta('b_gender',0,None,None,0)
b_payment = Beta('b_payment',0,None,None,0)

# utilities
M1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1)# + b_gender * Gender_1 + b_payment * Payment_1)
M2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2)# + b_gender * Gender_2 + b_payment * Payment_2)
M3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3)# + b_gender * Gender_3 + b_payment * Payment_3)
M4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4)# + b_gender * Gender_4 + b_payment * Payment_4)
M5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5)# + b_gender * Gender_5 + b_payment * Payment_5)
M6 = (asc_6 + b_price * Price_6 + b_calories * Calories_6 + b_rating * Rating_6 + b_discount * Discount_6)# + b_gender * Gender_6 + b_payment * Payment_6)

M = {1: M1 , 2: M2 , 3: M3, 4: M4, 5: M5, 6: M6}
av = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV, 6: AV}

logprob_mc1 = models.loglogit(M, av, Choice)
the_biogeme_mains1 = bio.BIOGEME ( database_mc1 , logprob_mc1 )

init_values = {'asc_2': 0.5, 'asc_3': -0.2, 'asc_4': 1.0,'asc_5':0.1,'b_calories':0.01,'b_rating':0.03,'b_discount':0.01,'b_gender':0.01,'b_payment':0.01}

the_biogeme_mains1.modelName = 'mains1_biogeme'
the_biogeme_mains1.calculateNullLoglikelihood ( av )
results_mains1 = the_biogeme_mains1.estimate(init_values= init_values, numberOfDraws=100, algo = 'BFGS')
results_mc1 = results_mains1.getEstimatedParameters ()
print(results_mc1)
results_mc1.iloc[:,0]

# none are significant
np.save('results_mc1',results_mc1)
np.load('results_mc1.npy')

# only 2 iterations?
# Newton with trust region for simple bound constraints: method
results_mains1.getBetaValues()
#
#%%
# BIOGEME FOR MAINS 2
mc2_df = df[['MC2 Price', 'MC2 Rating', 'MC2 Calories', 'MC2 Discount', 'Main Course 2',
               'Payment','Gender']]

mc2_df = mc2_df.dropna()

# add new columns with choice1 calories/price/rating etc?
mc2_df['Choice'] = mc2_df['Main Course 2'].replace({
    'ChickenPorridge':1,
    'KanomJeenNamYa':2,
    'ShrimpGlassNoodles':3,
    'AmericanFriedRice':4,
    'SausageFriedRice':5
    })

mc2_df['Gender'] = mc2_df['Gender'].replace({
    'Female':0,
    'Male':1})

mc2_df['Payment'] = mc2_df['Payment'].replace({
    'Cash':0,
    'Cashless':1})

mc2_df['Price 1'] = 50
mc2_df['Price 2'] = 45
mc2_df['Price 3'] = 90
mc2_df['Price 4'] = 60
mc2_df['Price 5'] = 70

mc2_df['Calories 1'] = 228
mc2_df['Calories 2'] = 81
mc2_df['Calories 3'] = 300
mc2_df['Calories 4'] = 790
mc2_df['Calories 5'] = 610

mc2_df['Rating 1'] = 4.2
mc2_df['Rating 2'] = 2
mc2_df['Rating 3'] = 3
mc2_df['Rating 4'] = 3.6
mc2_df['Rating 5'] = 1.2

mc2_df['Discount 1'] = 15
mc2_df['Discount 2'] = 5
mc2_df['Discount 3'] = 0
mc2_df['Discount 4'] = 0
mc2_df['Discount 5'] = 15

mc2_df['AV'] = 1

mc2_df['Choice'] = mc2_df['Choice'].astype(int)

mc2_df.drop(columns=['MC2 Price','MC2 Rating','MC2 Discount','MC2 Calories','Main Course 2'], inplace=True)
"""
mc2_df['MC2 Price'] = mc2_df['MC2 Price'].astype(float)
mc2_df['MC2 Calories'] = mc2_df['MC2 Calories'].astype(float)
mc2_df['MC2 Rating'] = mc2_df['MC2 Rating'].astype(float)
mc2_df['MC2 Discount'] = mc2_df['MC2 Discount'].astype(float)
mc2_df.drop(columns=['Main Course 2'], inplace=True)
"""
database_mc2 = db.Database('mc2', mc2_df)
Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')
database_mc2.variables

Gender = Variable('Gender')
Payment = Variable('Payment')

Gender_1 = database_mc2.DefineVariable('Gender_1', Gender * (Choice == 1))
Gender_2 = database_mc2.DefineVariable('Gender_2', Gender * (Choice == 2))
Gender_3 = database_mc2.DefineVariable('Gender_3', Gender * (Choice == 3))
Gender_4 = database_mc2.DefineVariable('Gender_4', Gender * (Choice == 4))
Gender_5 = database_mc2.DefineVariable('Gender_5', Gender * (Choice == 5))

Payment_1 = database_mc2.DefineVariable('Payment_1', Payment * (Choice == 1))
Payment_2 = database_mc2.DefineVariable('Payment_2', Payment * (Choice == 2))
Payment_3 = database_mc2.DefineVariable('Payment_3', Payment * (Choice == 3))
Payment_4 = database_mc2.DefineVariable('Payment_4', Payment * (Choice == 4))
Payment_5 = database_mc2.DefineVariable('Payment_5', Payment * (Choice == 5))

AV = Variable('AV')

# params
asc_1 = Beta ('asc_1', 0, None , None , 1)
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)

b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)
b_gender = Beta('b_gender',0,None,None,0)
b_payment = Beta('b_payment',0,None,None,0)

# utilities
M1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1) #+ b_gender * Gender_1 + b_payment * Payment_1)
M2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2) # + b_gender * Gender_2 + b_payment * Payment_2)
M3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3) # + b_gender * Gender_3 + b_payment * Payment_3)
M4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4) #+ b_gender * Gender_4 + b_payment * Payment_4)
M5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5) # + b_gender * Gender_5 + b_payment * Payment_5)
"""
ASC_1 = Beta('ASC_1', 0, None, None, 1)
ASC_2 = Beta('ASC_2', 0, None, None, 0)
ASC_3 = Beta('ASC_3', 0, None, None, 0)
ASC_4 = Beta('ASC_4', 0, None, None, 0)
ASC_5 = Beta('ASC_5', 0, None, None, 0)
# Add more utility functions for other alternatives if needed

# Define explanatory variables (attributes of alternatives)
price = Variable('MC2 Price')
calories = Variable('MC2 Calories')
rating = Variable('MC2 Rating')
discount = Variable('MC2 Discount')
# Add more explanatory variables as required

# Combine utility functions to create the model expression (logit model)
V_1 = ASC_1 + Beta('B_Price', 0, None, None,0) * price + Beta('B_Calories', 0, None, None,0) * calories + Beta('B_Rating', 0, None, None,0) * rating + Beta('B_Discount',0,None,None,0) * discount
V_2 = ASC_2 + Beta('B_Price', 0, None, None,0) * price + Beta('B_Calories', 0, None, None,0) * calories + Beta('B_Rating', 0, None, None,0) * rating + Beta('B_Discount',0,None,None,0) * discount
V_3 = ASC_3 + Beta('B_Price', 0, None, None,0) * price + Beta('B_Calories', 0, None, None,0) * calories + Beta('B_Rating', 0, None, None,0) * rating + Beta('B_Discount',0,None,None,0) * discount
V_4 = ASC_4 + Beta('B_Price', 0, None, None,0) * price + Beta('B_Calories', 0, None, None,0) * calories + Beta('B_Rating', 0, None, None,0) * rating + Beta('B_Discount',0,None,None,0) * discount
V_5 = ASC_5 + Beta('B_Price', 0, None, None,0) * price + Beta('B_Calories', 0, None, None,0) * calories + Beta('B_Rating', 0, None, None,0) * rating + Beta('B_Discount',0,None,None,0) * discount

Vs = {1:V_1,2:V_2,3:V_3,4:V_4,5:V_5}
av2 = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV}
logprob_mc2 = models.loglogit(Vs, av2, Choice)
the_biogeme_mains2 = bio.BIOGEME ( database_mc2 , logprob_mc2 )
the_biogeme_mains2.modelName = 'mains2_biogeme'
the_biogeme_mains2.calculateNullLoglikelihood ( av )
results_mains2 = the_biogeme_mains2.estimate() # estimates stay the same no matter what starting vals...
results_mc2 = results_mains2.getEstimatedParameters ()
print(results_mc2)
"""
# Add more utility expressions for other alternatives if needed

M_2 = {1: M1 , 2: M2 , 3: M3, 4: M4, 5: M5}
av2 = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV}

logprob_mc2 = models.loglogit(M_2, av2, Choice)
the_biogeme_mains2 = bio.BIOGEME ( database_mc2 , logprob_mc2 )

init_values = {'asc_2': 0.5, 'asc_3': -0.2, 'asc_4': 1.0,'asc_5':0.1,'b_calories':0.01,'b_rating':0.03,'b_discount':0.01,'b_gender':0.01,'b_payment':0.01}

the_biogeme_mains2.modelName = 'mains2_biogeme'
the_biogeme_mains2.calculateNullLoglikelihood ( av )
results_mains2 = the_biogeme_mains2.estimate(init_values=init_values,numberOfDraws=100) # estimates stay the same no matter what starting vals...
results_mc2 = results_mains2.getEstimatedParameters ()
print(results_mc2)
np.save('results_mc2',results_mc2)
np.load('results_mc2.npy',allow_pickle=True)
#%%
# ALL MAINS
mnl_data = df[['MC1 Price', 'MC1 Rating', 'MC1 Calories', 'MC1 Discount', 'Main Course 1',
               'MC2 Price','MC2 Rating','MC2 Calories','MC2 Discount', 'Main Course 2',
               'Payment','Gender']]

first_half = mnl_data[['MC1 Price', 'MC1 Rating', 'MC1 Calories', 'MC1 Discount', 'Main Course 1','Gender','Payment']]
second_half =mnl_data[[ 'MC2 Price','MC2 Rating','MC2 Calories','MC2 Discount', 'Main Course 2','Gender','Payment']]
second_half.rename(columns={'MC2 Price':'MC1 Price','MC2 Calories':'MC1 Calories','MC2 Discount':'MC1 Discount','MC2 Rating':'MC1 Rating','Main Course 2':'Main Course 1'},inplace=True)

mnl_data =pd.concat([first_half, second_half],axis=0)

# Remove rows with missing values or invalid entries
mnl_data = mnl_data.dropna()

# Encode categorical variable (Main Course 1) using one-hot encoding
#encoded_data = pd.get_dummies(mnl_data, columns=['Main Course 1'], drop_first=True)

#encoded_data['MC1 Price'] = pd.to_numeric(encoded_data['MC1 Price'], errors='coerce')
#encoded_data['MC1 Rating'] = pd.to_numeric(encoded_data['MC1 Rating'], errors='coerce')
#encoded_data['MC1 Calories'] = pd.to_numeric(encoded_data['MC1 Calories'], errors='coerce')
#encoded_data['MC1 Discount'] = pd.to_numeric(encoded_data['MC1 Discount'], errors='coerce')


# add new columns with choice1 calories/price/rating etc?
mnl_data['Choice'] = mnl_data['Main Course 1'].replace({
    'PadKrapow': 1,
    'PadThai': 2,
    'HainaneseChickenRice': 3,
    'GreenChickenCurry': 4,
    'ShrimpStickyRice': 5,
    'ShrimpFriedRice':6,
    'ShrimpGlassNoodles':7,
    'ChickenPorridge':8,
    'KanomJeenNamYa':9,
    'AmericanFriedRice':10,
    'SausageFriedRice':11
    # Add more mappings for other main course choices
})

mnl_data['Gender'] = mnl_data['Gender'].replace({
    'Female':0,
    'Male':1})

mnl_data['Payment'] = mnl_data['Payment'].replace({
    'Cash':0,
    'Cashless':1})

mnl_data['Price 1'] = 35
mnl_data['Price 2'] = 90
mnl_data['Price 3'] = 80
mnl_data['Price 4'] = 60
mnl_data['Price 5'] = 60
mnl_data['Price 6'] = 80
mnl_data['Price 7'] = 90
mnl_data['Price 8'] = 50
mnl_data['Price 9'] = 45
mnl_data['Price 10'] = 100
mnl_data['Price 11'] = 70

mnl_data['Calories 1'] = 372
mnl_data['Calories 2'] = 486
mnl_data['Calories 3'] = 597
mnl_data['Calories 4'] = 240
mnl_data['Calories 5'] = 477
mnl_data['Calories 6'] = 289
mnl_data['Calories 7'] = 300
mnl_data['Calories 8'] = 228
mnl_data['Calories 9'] = 81
mnl_data['Calories 10'] = 790
mnl_data['Calories 11'] = 610

mnl_data['Rating 1'] = 5
mnl_data['Rating 2'] = 5
mnl_data['Rating 3'] = 3.5
mnl_data['Rating 4'] = 2.4
mnl_data['Rating 5'] = 2.8
mnl_data['Rating 6'] = 4.3
mnl_data['Rating 7'] = 3
mnl_data['Rating 8'] = 4.2
mnl_data['Rating 9'] = 2
mnl_data['Rating 10'] = 3.6
mnl_data['Rating 11'] = 1.2

mnl_data['Discount 1'] = 0
mnl_data['Discount 2'] = 0
mnl_data['Discount 3'] = 0
mnl_data['Discount 4'] = 0
mnl_data['Discount 5'] = 20
mnl_data['Discount 6'] = 0
mnl_data['Discount 7'] = 0
mnl_data['Discount 8'] = 25
mnl_data['Discount 9'] = 5
mnl_data['Discount 10'] = 0
mnl_data['Discount 11'] = 15

mnl_data['AV'] = 1

mnl_data['Choice'] = mnl_data['Choice'].astype(int)

mnl_data.drop(columns=['MC1 Price','MC1 Rating','MC1 Discount','MC1 Calories','Main Course 1'], inplace=True)

#%%
# BIOGEME FOR MAINS
database_new = db.Database('test', mnl_data)
Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')
Price_6 = Variable('Price 6')
Price_7 = Variable('Price 7')
Price_8 = Variable('Price 8')
Price_9 = Variable('Price 9')
Price_10= Variable('Price 10')
Price_11= Variable('Price 11')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')
Calories_6 = Variable('Calories 6')
Calories_7 = Variable('Calories 7')
Calories_8 = Variable('Calories 8')
Calories_9 = Variable('Calories 9')
Calories_10= Variable('Calories 10')
Calories_11= Variable('Calories 11')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')
Rating_6 = Variable('Rating 6')
Rating_7 = Variable('Rating 7')
Rating_8 = Variable('Rating 8')
Rating_9 = Variable('Rating 9')
Rating_10= Variable('Rating 10')
Rating_11= Variable('Rating 11')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')
Discount_6 = Variable('Discount 6')
Discount_7 = Variable('Discount 7')
Discount_8= Variable('Discount 8')
Discount_9 = Variable('Discount 9')
Discount_10= Variable('Discount 10')
Discount_11= Variable('Discount 11')

Gender = Variable('Gender')
Payment = Variable('Payment')

Choice = Variable('Choice')
AV = Variable('AV')

Gender_1 = database_new.DefineVariable('Gender_1', Gender * (Choice == 1))
Gender_2 = database_new.DefineVariable('Gender_2', Gender * (Choice == 2))
Gender_3 = database_new.DefineVariable('Gender_3', Gender * (Choice == 3))
Gender_4 = database_new.DefineVariable('Gender_4', Gender * (Choice == 4))
Gender_5 = database_new.DefineVariable('Gender_5', Gender * (Choice == 5))
Gender_6 = database_new.DefineVariable('Gender_6', Gender * (Choice == 6))
Gender_7 = database_new.DefineVariable('Gender_7', Gender * (Choice == 7))
Gender_8 = database_new.DefineVariable('Gender_8', Gender * (Choice == 8))
Gender_9 = database_new.DefineVariable('Gender_9', Gender * (Choice == 9))
Gender_10= database_new.DefineVariable('Gender_10', Gender * (Choice == 10))
Gender_11= database_new.DefineVariable('Gender_11', Gender * (Choice == 11))

Payment_1 = database_new.DefineVariable('Payment_1', Payment * (Choice == 1))
Payment_2 = database_new.DefineVariable('Payment_2', Payment * (Choice == 2))
Payment_3 = database_new.DefineVariable('Payment_3', Payment * (Choice == 3))
Payment_4 = database_new.DefineVariable('Payment_4', Payment * (Choice == 4))
Payment_5 = database_new.DefineVariable('Payment_5', Payment * (Choice == 5))
Payment_6 = database_new.DefineVariable('Payment_6', Payment * (Choice == 6))
Payment_7 = database_new.DefineVariable('Payment_7', Payment * (Choice == 7))
Payment_8 = database_new.DefineVariable('Payment_8', Payment * (Choice == 8))
Payment_9 = database_new.DefineVariable('Payment_9', Payment * (Choice == 9))
Payment_10= database_new.DefineVariable('Payment_10', Payment * (Choice == 10))
Payment_11= database_new.DefineVariable('Payment_11', Payment * (Choice == 11))

# params
asc_1 = Beta ('asc_1', 0, None , None , 1)
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)
asc_6 = Beta ('asc_6', 0, None , None , 0)
asc_7 = Beta ('asc_7', 0, None , None , 0)
asc_8 = Beta ('asc_8', 0, None , None , 0)
asc_9 = Beta ('asc_9', 0, None , None , 0)
asc_10 = Beta('asc_10',0,None,None,0)
asc_11= Beta ('asc_11', 0, None , None , 0)

b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)
b_payment = Beta('b_payment',0,None,None,0)
b_gender = Beta('b_gender',0,None,None,0)


# utilities
M1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 + b_payment * Payment_1 + b_gender * Gender_1)
M2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 + b_payment * Payment_2 + b_gender * Gender_2)
M3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 + b_payment * Payment_3 + b_gender * Gender_3)
M4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 + b_payment * Payment_4 + b_gender * Gender_4)
M5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 + b_payment * Payment_5 + b_gender * Gender_5)
M6 = (asc_6 + b_price * Price_6 + b_calories * Calories_6 + b_rating * Rating_6 + b_discount * Discount_6 + b_payment * Payment_6 + b_gender * Gender_6)
M7 = (asc_7 + b_price * Price_7 + b_calories * Calories_7 + b_rating * Rating_7 + b_discount * Discount_7 + b_payment * Payment_7 + b_gender * Gender_7)
M8 = (asc_8 + b_price * Price_8 + b_calories * Calories_8 + b_rating * Rating_8 + b_discount * Discount_8 + b_payment * Payment_8 + b_gender * Gender_8)
M9 = (asc_9 + b_price * Price_9 + b_calories * Calories_9 + b_rating * Rating_9 + b_discount * Discount_9 + b_payment * Payment_9 + b_gender * Gender_9)
M10= (asc_10+ b_price * Price_10+ b_calories * Calories_10+ b_rating * Rating_10+ b_discount * Discount_10+ b_payment * Payment_10+ b_gender * Gender_10)
M11= (asc_11+ b_price * Price_11+ b_calories * Calories_11+ b_rating * Rating_11+ b_discount * Discount_11+ b_payment * Payment_11+ b_gender * Gender_11)

M = {1: M1 , 2: M2 , 3: M3, 4: M4, 5: M5, 6: M6, 7: M7, 8: M8, 9: M9, 10: M10, 11: M11}
av = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV, 6: AV, 7: AV, 8: AV, 9: AV, 10: AV, 11: AV}

logprob = models.loglogit(M, av, Choice)
the_biogeme_mains = bio.BIOGEME ( database_new , logprob )
database_new.variables

the_biogeme_mains.modelName = 'mains_biogeme'
the_biogeme_mains.calculateNullLoglikelihood ( av )
results_mains = the_biogeme_mains.estimate()
pandas_results_mains = results_mains.getEstimatedParameters ()
print(pandas_results_mains)

# price and calories significant coefficients 

#%%
# STARTERS
mnl_data2 = df[['A1 Price', 'A1 Rating', 'A1 Calories', 'A1 Discount', 'Appetizer 1',
                'A2 Price', 'A2 Rating', 'A2 Calories', 'A2 Discount',
                'Payment','Gender','Appetizer 2']]

first_half2= mnl_data2[['A1 Price', 'A1 Rating', 'A1 Calories', 'A1 Discount', 'Appetizer 1','Payment','Gender']]
second_half2=mnl_data2[[ 'A2 Price','A2 Rating','A2 Calories','A2 Discount', 'Appetizer 2','Payment','Gender']]
second_half2.rename(columns={'A2 Price':'A1 Price','A2 Calories':'A1 Calories','A2 Discount':'A1 Discount','A2 Rating':'A1 Rating','Appetizer 2': 'Appetizer 1'}, inplace= True)
                             

mnl_data2=pd.concat([first_half2,second_half2],axis=0)

# Remove rows with missing values or invalid entries
mnl_data2 = mnl_data2.dropna()

# Encode categorical variable (Main Course 1) using one-hot encoding
#encoded_data2 = pd.get_dummies(mnl_data2, columns=['Appetizer 1'], drop_first=True)

#encoded_data2['A1 Price'] = pd.to_numeric(encoded_data2['A1 Price'], errors='coerce')
#encoded_data2['A1 Rating'] = pd.to_numeric(encoded_data2['A1 Rating'], errors='coerce')
#encoded_data2['A1 Calories'] = pd.to_numeric(encoded_data2['A1 Calories'], errors='coerce')
#encoded_data2['A1 Discount'] = pd.to_numeric(encoded_data2['A1 Discount'], errors='coerce')

# add new columns with choice1 calories/price/rating etc?
mnl_data2['Choice'] = mnl_data2['Appetizer 1'].replace({
    'FriedCalamari': 1,
    'FriedCalamari ':1,
    'EggRolls': 2,
    'ShrimpCake': 3,
    'FishCake': 4,
    'FishCake ':4,
    'FriedShrimpBall': 5,
    'HerbChicken':6,
    'SaladRoll':7,
    'GrilledShrimp':8,
    'MiangKham':9,
    'CrispyCrab':10,
    'CrispyCrab ':10,
    'ShrimpDumpling':11
    # Add more mappings for other main course choices
})

mnl_data2['Gender'] = mnl_data2['Gender'].replace({
    'Female':0,
    'Male':1})

mnl_data2['Payment'] = mnl_data2['Payment'].replace({
    'Cash':0,
    'Cashless':1})

mnl_data2['Price 1'] = 80
mnl_data2['Price 2'] = 40
mnl_data2['Price 3'] = 120
mnl_data2['Price 4'] = 100
mnl_data2['Price 5'] = 30
mnl_data2['Price 6'] = 60
mnl_data2['Price 7'] = 60
mnl_data2['Price 8'] = 83
mnl_data2['Price 9'] = 90
mnl_data2['Price 10'] = 30
mnl_data2['Price 11'] = 60

mnl_data2['Calories 1'] = 187
mnl_data2['Calories 2'] = 480
mnl_data2['Calories 3'] = 990
mnl_data2['Calories 4'] = 147
mnl_data2['Calories 5'] = 468
mnl_data2['Calories 6'] = 376
mnl_data2['Calories 7'] = 1182
mnl_data2['Calories 8'] = 125
mnl_data2['Calories 9'] = 280
mnl_data2['Calories 10'] = 544
mnl_data2['Calories 11'] = 300

mnl_data2['Rating 1'] = 4
mnl_data2['Rating 2'] = 3.5
mnl_data2['Rating 3'] = 5
mnl_data2['Rating 4'] = 2
mnl_data2['Rating 5'] = 1
mnl_data2['Rating 6'] = 3.1
mnl_data2['Rating 7'] = 3.2
mnl_data2['Rating 8'] = 2.7
mnl_data2['Rating 9'] = 5
mnl_data2['Rating 10'] = 4
mnl_data2['Rating 11'] = 4.8

mnl_data2['Discount 1'] = 0
mnl_data2['Discount 2'] = 5
mnl_data2['Discount 3'] = 20
mnl_data2['Discount 4'] = 20
mnl_data2['Discount 5'] = 0
mnl_data2['Discount 6'] = 10
mnl_data2['Discount 7'] = 10
mnl_data2['Discount 8'] = 10
mnl_data2['Discount 9'] = 20
mnl_data2['Discount 10'] = 0
mnl_data2['Discount 11'] = 5

mnl_data2['AV'] = 1

mnl_data2['Choice'] = mnl_data2['Choice'].astype(int)

mnl_data2.drop(columns=['A1 Price','A1 Rating','A1 Discount','A1 Calories','Appetizer 1'], inplace=True)

#%%
# BIOGEME STARTERS
database_starter = db.Database('starter', mnl_data2)

Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')
Price_6 = Variable('Price 6')
Price_7 = Variable('Price 7')
Price_8 = Variable('Price 8')
Price_9 = Variable('Price 9')
Price_10= Variable('Price 10')
Price_11= Variable('Price 11')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')
Calories_6 = Variable('Calories 6')
Calories_7 = Variable('Calories 7')
Calories_8 = Variable('Calories 8')
Calories_9 = Variable('Calories 9')
Calories_10= Variable('Calories 10')
Calories_11= Variable('Calories 11')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')
Rating_6 = Variable('Rating 6')
Rating_7 = Variable('Rating 7')
Rating_8 = Variable('Rating 8')
Rating_9 = Variable('Rating 9')
Rating_10= Variable('Rating 10')
Rating_11= Variable('Rating 11')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')
Discount_6 = Variable('Discount 6')
Discount_7 = Variable('Discount 7')
Discount_8= Variable('Discount 8')
Discount_9 = Variable('Discount 9')
Discount_10= Variable('Discount 10')
Discount_11= Variable('Discount 11')

Gender = Variable('Gender')
Payment = Variable('Payment')

Choice = Variable('Choice')
AV = Variable('AV')

Gender_1 = database_starter.DefineVariable('Gender_1', Gender * (Choice == 1))
Gender_2 = database_starter.DefineVariable('Gender_2', Gender * (Choice == 2))
Gender_3 = database_starter.DefineVariable('Gender_3', Gender * (Choice == 3))
Gender_4 = database_starter.DefineVariable('Gender_4', Gender * (Choice == 4))
Gender_5 = database_starter.DefineVariable('Gender_5', Gender * (Choice == 5))
Gender_6 = database_starter.DefineVariable('Gender_6', Gender * (Choice == 6))
Gender_7 = database_starter.DefineVariable('Gender_7', Gender * (Choice == 7))
Gender_8 = database_starter.DefineVariable('Gender_8', Gender * (Choice == 8))
Gender_9 = database_starter.DefineVariable('Gender_9', Gender * (Choice == 9))
Gender_10= database_starter.DefineVariable('Gender_10', Gender * (Choice == 10))
Gender_11= database_starter.DefineVariable('Gender_11', Gender * (Choice == 11))

Payment_1 = database_starter.DefineVariable('Payment_1', Payment * (Choice == 1))
Payment_2 = database_starter.DefineVariable('Payment_2', Payment * (Choice == 2))
Payment_3 = database_starter.DefineVariable('Payment_3', Payment * (Choice == 3))
Payment_4 = database_starter.DefineVariable('Payment_4', Payment * (Choice == 4))
Payment_5 = database_starter.DefineVariable('Payment_5', Payment * (Choice == 5))
Payment_6 = database_starter.DefineVariable('Payment_6', Payment * (Choice == 6))
Payment_7 = database_starter.DefineVariable('Payment_7', Payment * (Choice == 7))
Payment_8 = database_starter.DefineVariable('Payment_8', Payment * (Choice == 8))
Payment_9 = database_starter.DefineVariable('Payment_9', Payment * (Choice == 9))
Payment_10= database_starter.DefineVariable('Payment_10', Payment * (Choice == 10))
Payment_11= database_starter.DefineVariable('Payment_11', Payment * (Choice == 11))

# params
asc_1 = Beta ('asc_1', 0, None , None , 1) # setting as reference alternative? So isn't estimated
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)
asc_6 = Beta ('asc_6', 0, None , None , 0)
asc_7 = Beta ('asc_7', 0, None , None , 0)
asc_8 = Beta ('asc_8', 0, None , None , 0)
asc_9 = Beta ('asc_9', 0, None , None , 0)
asc_10= Beta ('asc_10', 0, None , None , 0)
asc_11= Beta ('asc_11', 0, None , None , 0)

# need to name differently as overwrites biogeme objects
b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)
b_payment = Beta('b_payment',0,None,None,0)
b_gender = Beta('b_gender',0,None,None,0)

# utilities
S1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 + b_payment * Payment_1 + b_gender * Gender_1)
S2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 + b_payment * Payment_2 + b_gender * Gender_2)
S3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 + b_payment * Payment_3 + b_gender * Gender_3)
S4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 + b_payment * Payment_4 + b_gender * Gender_4)
S5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 + b_payment * Payment_5 + b_gender * Gender_5)
S6 = (asc_6 + b_price * Price_6 + b_calories * Calories_6 + b_rating * Rating_6 + b_discount * Discount_6 + b_payment * Payment_6 + b_gender * Gender_6)
S7 = (asc_7 + b_price * Price_7 + b_calories * Calories_7 + b_rating * Rating_7 + b_discount * Discount_7 + b_payment * Payment_7 + b_gender * Gender_7)
S8 = (asc_8 + b_price * Price_8 + b_calories * Calories_8 + b_rating * Rating_8 + b_discount * Discount_8 + b_payment * Payment_8 + b_gender * Gender_8)
S9 = (asc_9 + b_price * Price_9 + b_calories * Calories_9 + b_rating * Rating_9 + b_discount * Discount_9 + b_payment * Payment_9 + b_gender * Gender_9)
S10= (asc_10+ b_price * Price_10+ b_calories * Calories_10+ b_rating * Rating_10+ b_discount * Discount_10+ b_payment * Payment_10+ b_gender * Gender_10)
S11= (asc_11+ b_price * Price_11+ b_calories * Calories_11+ b_rating * Rating_11+ b_discount * Discount_11+ b_payment * Payment_11+ b_gender * Gender_11)


S = {1: S1 , 2: S2 , 3: S3, 4: S4, 5: S5, 6: S6, 7: S7, 8: S8, 9: S9, 10: S10, 11: S11}
av = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV, 6: AV, 7: AV, 8: AV, 9: AV, 10: AV, 11: AV}

logprob2 = models.loglogit(S, av, Choice)
the_biogeme = bio.BIOGEME ( database_starter , logprob2)
database_starter.variables

the_biogeme.modelName = 'starter_biogeme'
the_biogeme.calculateNullLoglikelihood ( av )
results = the_biogeme.estimate()
pandas_results = results.getEstimatedParameters ()
print(pandas_results)
# signif p-values for all now when including 11 appetizers 
# asc (alternative specific constants) represent baseline utility for each alternative when all other attributes are zero
# p vals from test that coeffs are equal to zero
# coeffs estimated by MLE?

#%%
# STARTERS
a1_df = df[['A1 Price', 'A1 Rating', 'A1 Calories', 'A1 Discount', 'Appetizer 1','Payment','Gender','Appetizer 2']]

# add new columns with choice1 calories/price/rating etc?
a1_df['Choice'] = a1_df['Appetizer 1'].replace({
    'FriedCalamari': 1,
    'FriedCalamari ':1,
    'EggRolls': 2,
    'ShrimpCake': 3,
    'FishCake': 4,
    'FishCake ':4,
    'FriedShrimpBall': 5,
    'HerbChicken':6
})

a1_df['Gender'] = a1_df['Gender'].replace({
    'Female':0,
    'Male':1})

a1_df['Payment'] = a1_df['Payment'].replace({
    'Cash':0,
    'Cashless':1})

a1_df['Price 1'] = 80
a1_df['Price 2'] = 40
a1_df['Price 3'] = 120
a1_df['Price 4'] = 100
a1_df['Price 5'] = 30
a1_df['Price 6'] = 60


a1_df['Calories 1'] = 187
a1_df['Calories 2'] = 480
a1_df['Calories 3'] = 990
a1_df['Calories 4'] = 147
a1_df['Calories 5'] = 468
a1_df['Calories 6'] = 376


a1_df['Rating 1'] = 4
a1_df['Rating 2'] = 3.5
a1_df['Rating 3'] = 5
a1_df['Rating 4'] = 2
a1_df['Rating 5'] = 1
a1_df['Rating 6'] = 3.1

a1_df['Discount 1'] = 0
a1_df['Discount 2'] = 5
a1_df['Discount 3'] = 20
a1_df['Discount 4'] = 20
a1_df['Discount 5'] = 0
a1_df['Discount 6'] = 10

a1_df['AV'] = 1

a1_df['Choice'] = a1_df['Choice'].astype(int)

a1_df.drop(columns=['A1 Price','A1 Rating','A1 Discount','A1 Calories','Appetizer 1'], inplace=True)
a1_df.drop(columns=['Appetizer 2'],inplace=True)

#%%
# BIOGEME STARTERS
database_a1 = db.Database('appetizer_1', a1_df)

Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')
Price_6 = Variable('Price 6')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')
Calories_6 = Variable('Calories 6')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')
Rating_6 = Variable('Rating 6')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')
Discount_6 = Variable('Discount 6')

Choice = Variable('Choice')
AV = Variable('AV')
"""
Gender = Variable('Gender')
Payment = Variable('Payment')

Gender_1 = database_a1.DefineVariable('Gender_1', Gender * (Choice == 1))
Gender_2 = database_a1.DefineVariable('Gender_2', Gender * (Choice == 2))
Gender_3 = database_a1.DefineVariable('Gender_3', Gender * (Choice == 3))
Gender_4 = database_a1.DefineVariable('Gender_4', Gender * (Choice == 4))
Gender_5 = database_a1.DefineVariable('Gender_5', Gender * (Choice == 5))
Gender_6 = database_a1.DefineVariable('Gender_6', Gender * (Choice == 6))

Payment_1 = database_a1.DefineVariable('Payment_1', Payment * (Choice == 1))
Payment_2 = database_a1.DefineVariable('Payment_2', Payment * (Choice == 2))
Payment_3 = database_a1.DefineVariable('Payment_3', Payment * (Choice == 3))
Payment_4 = database_a1.DefineVariable('Payment_4', Payment * (Choice == 4))
Payment_5 = database_a1.DefineVariable('Payment_5', Payment * (Choice == 5))
Payment_6 = database_a1.DefineVariable('Payment_6', Payment * (Choice == 6))
"""
# params
asc_1 = Beta ('asc_1', 0, None , None , 1) # setting as reference alternative? So isn't estimated
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)
asc_6 = Beta ('asc_6', 0, None , None , 0)

# need to name differently as overwrites biogeme objects
b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)
"""
b_gender = Beta ('b_gender', 0, None , None , 0)
b_payment = Beta ('b_payment', 0, None , None , 0)

# utilities
S1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 + b_gender * Gender_1 + b_payment * Payment_1)
S2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 + b_gender * Gender_2 + b_payment * Payment_2)
S3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 + b_gender * Gender_3 + b_payment * Payment_3)
S4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 + b_gender * Gender_4 + b_payment * Payment_4)
S5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 + b_gender * Gender_5 + b_payment * Payment_5)
S6 = (asc_6 + b_price * Price_6 + b_calories * Calories_6 + b_rating * Rating_6 + b_discount * Discount_6 + b_gender * Gender_6 + b_payment * Payment_6)
"""
S1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 )
S2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 )
S3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 )
S4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 )
S5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 )
S6 = (asc_6 + b_price * Price_6 + b_calories * Calories_6 + b_rating * Rating_6 + b_discount * Discount_6 )

S = {1: S1 , 2: S2 , 3: S3, 4: S4, 5: S5, 6: S6}
av3 = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV, 6: AV}

logprob3 = models.loglogit(S, av3, Choice)
the_biogeme_a1 = bio.BIOGEME ( database_a1 , logprob3)

the_biogeme_a1.modelName = 'a1_biogeme'
the_biogeme_a1.calculateNullLoglikelihood ( av )
results_a1 = the_biogeme_a1.estimate(numberOfDraws=100,init_values=init_values)
a1_results = results_a1.getEstimatedParameters ()
print(a1_results)

np.save('a1_results',a1_results)
np.load('a1_results.npy')

coeffs = a1_results.iloc[:,1]

def calc_probs(df,initial_coeffs,name,length):
    # calculating overall MNL probabilities
    
    coeffs = {
        'asc_1': initial_coeffs[0],
        'asc_2': initial_coeffs[1],
        'asc_3': initial_coeffs[2],
        'asc_4': initial_coeffs[3],
        'asc_5': initial_coeffs[4],
        'b_price': initial_coeffs[7],
        'b_calories': initial_coeffs[5],
        'b_rating': initial_coeffs[8],
        'b_discount': initial_coeffs[6],
        #'b_gender' : initial_coeffs[7],
        #'b_payment' : initial_coeffs[8]
    }
    
    # getting utilities for each person and each alternative 
    probs_choice = []
    for k in range(1,length+1):
        choice_df = df.loc[df['Choice']==k]
        utilities = []
        for i in range(0,len(choice_df)):
            if k == 1:
                # could add gender and payment also 
                #+ coeffs['b_payment'] * choice_df['Payment'].iloc[i] + coeffs['b_gender'] * choice_df['Gender'].iloc[i]
                utilities.append(coeffs['b_price'] * float(choice_df[f'{name} Price'].iloc[i]) + coeffs['b_calories'] * float(choice_df[f'{name} Calories'].iloc[i]) + coeffs['b_rating'] * float(choice_df[f'{name} Rating'].iloc[i]) + coeffs['b_discount'] * float(choice_df[f'{name} Discount'].iloc[i]) ) #+ coeffs['b_gender']) * float(choice_df['Gender'].iloc[i]) + coeffs['b_payment'] * float(choice_df['Payment'].iloc[i]))
            else:
                utilities.append(coeffs[f'asc_{k-1}'] + coeffs['b_price'] * float(choice_df[f'{name} Price'].iloc[i]) + coeffs['b_calories'] * float(choice_df[f'{name} Calories'].iloc[i]) + coeffs['b_rating'] * float(choice_df[f'{name} Rating'].iloc[i]) + coeffs['b_discount'] * float(choice_df[f'{name} Discount'].iloc[i])) # + coeffs['b_gender'] * float(choice_df['Gender'].iloc[i]) + coeffs['b_payment'] * float(choice_df['Payment'].iloc[i]))


        choice_probs = [np.exp(utility) for utility in utilities]
        probs_choice.append(choice_probs)

    # now getting overall utility for each alternative
    tot_utility = sum([sum(probs_choice[i]) for i in range(0,length)])

    # now getting probs for each alternative
    probabilities_0 = []
    for j in range(0,length):
        probabilities_0.append(sum(probs_choice[j])/tot_utility)
        
    print(probabilities_0)
calc_probs(appetizer1_df,coeffs,'A1',6)

#%%
a2_df = df[['A2 Price', 'A2 Rating', 'A2 Calories', 'A2 Discount', 'Appetizer 2','Payment','Gender']]

# add new columns with choice1 calories/price/rating etc?
a2_df['Choice'] = a2_df['Appetizer 2'].replace({
    'GrilledShrimp': 1,
    'CrispyCrab': 2,
    'MiangKham': 3,
    'ShrimpDumpling': 4,
    'CrispyCrab ': 2,
    'SaladRoll': 5})

a2_df['Gender'] = a2_df['Gender'].replace({
    'Female':0,
    'Male':1})

a2_df['Payment'] = a2_df['Payment'].replace({
    'Cash':0,
    'Cashless':1})

a2_df['Price 1'] = 80
a2_df['Price 2'] = 30
a2_df['Price 3'] = 90
a2_df['Price 4'] = 60
a2_df['Price 5'] = 60


a2_df['Calories 1'] = 125
a2_df['Calories 2'] = 544
a2_df['Calories 3'] = 280
a2_df['Calories 4'] = 300
a2_df['Calories 5'] = 1182


a2_df['Rating 1'] = 3
a2_df['Rating 2'] = 4
a2_df['Rating 3'] = 5
a2_df['Rating 4'] = 3
a2_df['Rating 5'] = 3.2

a2_df['Discount 1'] = 10
a2_df['Discount 2'] = 0
a2_df['Discount 3'] = 20
a2_df['Discount 4'] = 10
a2_df['Discount 5'] = 10

a2_df['AV'] = 1

a2_df['Choice'] = a2_df['Choice'].astype(int)

a2_df.drop(columns=['A2 Price','A2 Rating','A2 Discount','A2 Calories','Appetizer 2'], inplace=True)

# BIOGEME STARTERS
database_a2 = db.Database('appetizer_2', a2_df)

Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')

Choice = Variable('Choice')
AV = Variable('AV')
"""
Gender = Variable('Gender')
Payment = Variable('Payment')

Gender_1 = database_a1.DefineVariable('Gender_1', Gender * (Choice == 1))
Gender_2 = database_a1.DefineVariable('Gender_2', Gender * (Choice == 2))
Gender_3 = database_a1.DefineVariable('Gender_3', Gender * (Choice == 3))
Gender_4 = database_a1.DefineVariable('Gender_4', Gender * (Choice == 4))
Gender_5 = database_a1.DefineVariable('Gender_5', Gender * (Choice == 5))
Gender_6 = database_a1.DefineVariable('Gender_6', Gender * (Choice == 6))

Payment_1 = database_a1.DefineVariable('Payment_1', Payment * (Choice == 1))
Payment_2 = database_a1.DefineVariable('Payment_2', Payment * (Choice == 2))
Payment_3 = database_a1.DefineVariable('Payment_3', Payment * (Choice == 3))
Payment_4 = database_a1.DefineVariable('Payment_4', Payment * (Choice == 4))
Payment_5 = database_a1.DefineVariable('Payment_5', Payment * (Choice == 5))
Payment_6 = database_a1.DefineVariable('Payment_6', Payment * (Choice == 6))
"""
# params
asc_1 = Beta ('asc_1', 0, None , None , 1) # setting as reference alternative? So isn't estimated
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)
asc_6 = Beta ('asc_6', 0, None , None , 0)

# need to name differently as overwrites biogeme objects
b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)
"""
b_gender = Beta ('b_gender', 0, None , None , 0)
b_payment = Beta ('b_payment', 0, None , None , 0)

# utilities
S1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 + b_gender * Gender_1 + b_payment * Payment_1)
S2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 + b_gender * Gender_2 + b_payment * Payment_2)
S3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 + b_gender * Gender_3 + b_payment * Payment_3)
S4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 + b_gender * Gender_4 + b_payment * Payment_4)
S5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 + b_gender * Gender_5 + b_payment * Payment_5)
S6 = (asc_6 + b_price * Price_6 + b_calories * Calories_6 + b_rating * Rating_6 + b_discount * Discount_6 + b_gender * Gender_6 + b_payment * Payment_6)
"""
S1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 )
S2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 )
S3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 )
S4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 )
S5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 )

S = {1: S1 , 2: S2 , 3: S3, 4: S4, 5: S5}
av3 = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV}

logprob3 = models.loglogit(S, av3, Choice)
the_biogeme_a2 = bio.BIOGEME ( database_a2 , logprob3)

the_biogeme_a2.modelName = 'a2_biogeme'
the_biogeme_a2.calculateNullLoglikelihood ( av )
results_a2 = the_biogeme_a2.estimate(init_values=init_values,numberOfDraws=100)
a2_results = results_a2.getEstimatedParameters ()
print(a2_results)

np.save('a2_results',a2_results,allow_pickle=True)
#%%


d1_df = df[['D1 Price', 'D1 Rating', 'D1 Calories', 'D1 Discount', 'Dessert 1','Payment','Gender']]

df['Dessert 1'].unique()

# add new columns with choice1 calories/price/rating etc?
d1_df['Choice'] = d1_df['Dessert 1'].replace({
    'CoconutCustard': 1,
    'KhanomMawKaeng':2,
    'MangoStickyRice': 3,
    'LodChongThai': 3,
    'KanomKrok': 5
})

d1_df['Gender'] = d1_df['Gender'].replace({
    'Female':0,
    'Male':1})

d1_df['Payment'] = d1_df['Payment'].replace({
    'Cash':0,
    'Cashless':1})

d1_df['Price 1'] = 20
d1_df['Price 2'] = 40
d1_df['Price 3'] = 30
d1_df['Price 4'] = 20
d1_df['Price 5'] = 20

d1_df['Calories 1'] = 540
d1_df['Calories 2'] = 244
d1_df['Calories 3'] = 270
d1_df['Calories 4'] = 215
d1_df['Calories 5'] = 240

d1_df['Rating 1'] = 4
d1_df['Rating 2'] = 4
d1_df['Rating 3'] = 5
d1_df['Rating 4'] = 1.8
d1_df['Rating 5'] = 3

d1_df['Discount 1'] = 10
d1_df['Discount 2'] = 20
d1_df['Discount 3'] = 0
d1_df['Discount 4'] = 0
d1_df['Discount 5'] = 5

d1_df['AV'] = 1

d1_df['Choice'] = d1_df['Choice'].astype(int)

d1_df.drop(columns=['D1 Price','D1 Rating','D1 Discount','D1 Calories','Dessert 1'], inplace=True)

database_d1 = db.Database('dessert_1', d1_df)

Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')

Choice = Variable('Choice')
AV = Variable('AV')

# params
asc_1 = Beta ('asc_1', 0, None , None , 1) # setting as reference alternative? So isn't estimated
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)

# need to name differently as overwrites biogeme objects
b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)

# utilities
D1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 )
D2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 )
D3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 )
D4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 )
D5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 )

D = {1: D1 , 2: D2 , 3: D3, 4: D4, 5: D5}
av4 = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV}

logprob4 = models.loglogit(D, av4, Choice)
the_biogeme_d1 = bio.BIOGEME ( database_d1 , logprob4)

the_biogeme_d1.modelName = 'd1_biogeme'
the_biogeme_d1.calculateNullLoglikelihood ( av )
results_d1 = the_biogeme_d1.estimate()
d1_results = results_d1.getEstimatedParameters ()
print(d1_results)

np.save('d1_results', d1_results)

#%%
d2_df = df[['D2 Price', 'D2 Rating', 'D2 Calories', 'D2 Discount', 'Dessert 2','Payment','Gender']]


# add new columns with choice1 calories/price/rating etc?
d2_df['Choice'] = d2_df['Dessert 2'].replace({
    'Pudding': 1,
    'VanillaIceCream': 2,
    'ApplePie': 3,
    'ChocolateCake': 4,
    'ChocolateIcecream': 5})

d2_df['Gender'] = d2_df['Gender'].replace({
    'Female':0,
    'Male':1})

d2_df['Payment'] = d2_df['Payment'].replace({
    'Cash':0,
    'Cashless':1})

d2_df['Price 1'] = 25
d2_df['Price 2'] = 25
d2_df['Price 3'] = 55
d2_df['Price 4'] = 45
d2_df['Price 5'] = 25

d2_df['Calories 1'] = 120
d2_df['Calories 2'] = 330
d2_df['Calories 3'] = 296
d2_df['Calories 4'] = 424
d2_df['Calories 5'] = 335

d2_df['Rating 1'] = 3.6
d2_df['Rating 2'] = 1.5
d2_df['Rating 3'] = 4.3
d2_df['Rating 4'] = 5
d2_df['Rating 5'] = 2

d2_df['Discount 1'] = 10
d2_df['Discount 2'] = 15
d2_df['Discount 3'] = 10
d2_df['Discount 4'] = 0
d2_df['Discount 5'] = 25

d2_df['AV'] = 1

d2_df['Choice'] = d2_df['Choice'].astype(int)

d2_df.drop(columns=['D2 Price','D2 Rating','D2 Discount','D2 Calories','Dessert 2'], inplace=True)

database_d2 = db.Database('dessert_2', d2_df)

Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')

Choice = Variable('Choice')
AV = Variable('AV')

# params
asc_1 = Beta ('asc_1', 0, None , None , 1) # setting as reference alternative? So isn't estimated
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)

# need to name differently as overwrites biogeme objects
b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)

# utilities
D1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 )
D2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 )
D3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 )
D4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 )
D5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 )

D = {1: D1 , 2: D2 , 3: D3, 4: D4, 5: D5}
av4 = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV}

logprob4 = models.loglogit(D, av4, Choice)
the_biogeme_d2 = bio.BIOGEME ( database_d2 , logprob4)

the_biogeme_d2.modelName = 'd2_biogeme'
the_biogeme_d2.calculateNullLoglikelihood ( av )
results_d2 = the_biogeme_d2.estimate()
d2_results = results_d2.getEstimatedParameters ()
print(d2_results)

np.save('results_d2',d2_results)

#%%

s_df = df[['S Price', 'S Rating', 'S Calories', 'S Discount', 'Soup','Payment','Gender']]


# add new columns with choice1 calories/price/rating etc?
s_df['Choice'] = s_df['Soup'].replace({
    'TomJuedMooSap': 1,
    'KaiPalo': 2,
    'OnionSoup':3,
    'TomYumKung':4,
    'TomKhaGai':5,
    'LengZab':6,
    'GlassNoodlesSoup':7})

s_df['Gender'] = d2_df['Gender'].replace({
    'Female':0,
    'Male':1})

s_df['Payment'] = d2_df['Payment'].replace({
   'Cash':0,
    'Cashless':1})

s_df['Price 1'] = 40
s_df['Price 2'] = 50
s_df['Price 3'] = 55
s_df['Price 4'] = 90
s_df['Price 5'] = 60
s_df['Price 6'] = 110
s_df['Price 7'] = 80

s_df['Calories 1'] = 80
s_df['Calories 2'] = 180
s_df['Calories 3'] = 180
s_df['Calories 4'] = 229
s_df['Calories 5'] = 357
s_df['Calories 6'] = 140
s_df['Calories 7'] = 300

s_df['Rating 1'] = 2.7
s_df['Rating 2'] = 3.3
s_df['Rating 3'] = 4.1
s_df['Rating 4'] = 4.3
s_df['Rating 5'] = 3.8
s_df['Rating 6'] = 5
s_df['Rating 7'] = 2.9

s_df['Discount 1'] = 0
s_df['Discount 2'] = 0
s_df['Discount 3'] = 5
s_df['Discount 4'] = 5
s_df['Discount 5'] = 10
s_df['Discount 6'] = 15
s_df['Discount 7'] = 0
0
s_df['AV'] = 1

s_df['Choice'] = s_df['Choice'].astype(int)

s_df.drop(columns=['S Price','S Rating','S Discount','S Calories','Soup'], inplace=True)

database_s = db.Database('soup', s_df)

Price_1 = Variable('Price 1')
Price_2 = Variable('Price 2')
Price_3 = Variable('Price 3')
Price_4 = Variable('Price 4')
Price_5 = Variable('Price 5')
Price_6 = Variable('Price 6')
Price_7 = Variable('Price 7')

Calories_1 = Variable('Calories 1')
Calories_2 = Variable('Calories 2')
Calories_3 = Variable('Calories 3')
Calories_4 = Variable('Calories 4')
Calories_5 = Variable('Calories 5')
Calories_6 = Variable('Calories 6')
Calories_7 = Variable('Calories 7')

Rating_1 = Variable('Rating 1')
Rating_2 = Variable('Rating 2')
Rating_3 = Variable('Rating 3')
Rating_4 = Variable('Rating 4')
Rating_5 = Variable('Rating 5')
Rating_6 = Variable('Rating 6')
Rating_7 = Variable('Rating 7')

Discount_1 = Variable('Discount 1')
Discount_2 = Variable('Discount 2')
Discount_3 = Variable('Discount 3')
Discount_4 = Variable('Discount 4')
Discount_5 = Variable('Discount 5')
Discount_6 = Variable('Discount 6')
Discount_7 = Variable('Discount 7')

Choice = Variable('Choice')
AV = Variable('AV')

# params
asc_1 = Beta ('asc_1', 0, None , None , 1) # setting as reference alternative? So isn't estimated
asc_2 = Beta ('asc_2', 0, None , None , 0)
asc_3 = Beta ('asc_3', 0, None , None , 0)
asc_4 = Beta ('asc_4', 0, None , None , 0)
asc_5 = Beta ('asc_5', 0, None , None , 0)
asc_6 = Beta ('asc_6', 0, None , None , 0)
asc_7 = Beta ('asc_7', 0, None , None , 0)

# need to name differently as overwrites biogeme objects
b_calories = Beta ('b_calories', 0, None , None , 0)
b_rating = Beta ('b_rating', 0, None , None , 0)
b_price = Beta ('b_price', 0, None , None , 0)
b_discount = Beta ('b_discount', 0, None , None , 0)

# utilities
D1 = (asc_1 + b_price * Price_1 + b_calories * Calories_1 + b_rating * Rating_1 + b_discount * Discount_1 )
D2 = (asc_2 + b_price * Price_2 + b_calories * Calories_2 + b_rating * Rating_2 + b_discount * Discount_2 )
D3 = (asc_3 + b_price * Price_3 + b_calories * Calories_3 + b_rating * Rating_3 + b_discount * Discount_3 )
D4 = (asc_4 + b_price * Price_4 + b_calories * Calories_4 + b_rating * Rating_4 + b_discount * Discount_4 )
D5 = (asc_5 + b_price * Price_5 + b_calories * Calories_5 + b_rating * Rating_5 + b_discount * Discount_5 )
D6 = (asc_6 + b_price * Price_6 + b_calories * Calories_6 + b_rating * Rating_6 + b_discount * Discount_6 )
D7 = (asc_7 + b_price * Price_7 + b_calories * Calories_7 + b_rating * Rating_7 + b_discount * Discount_7 )

D = {1: D1 , 2: D2 , 3: D3, 4: D4, 5: D5, 6: D6, 7:D7}
av4 = {1: AV , 2: AV, 3: AV, 4: AV, 5: AV, 6:AV, 7:AV}

logprob4 = models.loglogit(D, av4, Choice)
the_biogeme_s = bio.BIOGEME ( database_s , logprob4)

the_biogeme_s.modelName = 's_biogeme'
the_biogeme_s.calculateNullLoglikelihood ( av )
results_s = the_biogeme_s.estimate()
s_results = results_s.getEstimatedParameters ()
print(s_results)

np.save('s_results',s_results)
