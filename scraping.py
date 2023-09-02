# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:17:59 2023

@author: elfre
"""
# TODO: Add manually amounts for None columns. Calculate sustainability scores

from recipe_scrapers import scrape_me

salad_roll = scrape_me('https://www.allrecipes.com/recipe/24239/vietnamese-fresh-spring-rolls/',wild_mode=True)

salad_roll.title()
salad_roll.ingredients()
salad_roll.yields()
salad_roll.nutrients()  # if available

sfr = scrape_me('https://www.allrecipes.com/recipe/47022/glos-sausage-fried-rice/')

sfr.title()
sfr.ingredients()
sfr.yields()
sfr.nutrients()

pad_thai = scrape_me('https://www.allrecipes.com/recipe/19306/sukhothai-pad-thai/')

def get_ingredients(url):
    dish = scrape_me(url,wild_mode=True)
    ingr = dish.ingredients()
    servings = dish.yields()
    return ingr, servings

def get_nutrients(url):
    dish = scrape_me(url,wild_mode=True)
    nutr = dish.nutrients()
    return nutr

get_nutrients('https://www.allrecipes.com/recipe/19306/sukhothai-pad-thai/')

urls_mains = ['https://www.allrecipes.com/recipe/19306/sukhothai-pad-thai/','https://www.allrecipes.com/recipe/257938/spicy-thai-basil-chicken-pad-krapow-gai/',
              'https://www.allrecipes.com/recipe/246922/must-try-hainanese-chicken-rice/','https://www.allrecipes.com/recipe/141833/thai-green-curry-chicken/',
              'https://www.allrecipes.com/recipe/212940/chicken-arroz-caldo-chicken-rice-porridge/', 'https://www.allrecipes.com/recipe/47022/glos-sausage-fried-rice/',
              'https://www.mirlandraskitchen.com/sticky-honey-garlic-shrimp/#recipe','https://www.simplyrecipes.com/recipes/shrimp_fried_rice/',
              'https://rachelcooksthai.com/noodles-with-fish-curry/#recipe','https://www.simplysuwanee.com/thai-glass-noodle-salad-with-shrimp/','https://hot-thai-kitchen.com/american-fried-rice/']

# American Fried Rice no nutrient info 

urls_starters = ['https://www.allrecipes.com/recipe/24239/vietnamese-fresh-spring-rolls/','https://www.allrecipes.com/recipe/14704/best-egg-rolls/',
                 'https://www.allrecipes.com/recipe/286333/thai-inspired-shrimp-cakes/','https://www.allrecipes.com/recipe/241725/deep-fried-calamari-rings/',
                 'https://www.allrecipes.com/recipe/271154/easy-crispy-vietnamese-shrimp-balls/','https://www.allrecipes.com/recipe/64021/thai-fish-cakes/',
                 'https://www.allrecipes.com/recipe/9011/simple-lemon-herb-chicken/','https://www.allrecipes.com/recipe/12775/spicy-grilled-shrimp/',
                 'https://www.allrecipes.com/recipe/220476/shrimp-and-edamame-dumplings/','https://www.allrecipes.com/recipe/14850/crab-rangoon-i/',
                 'https://www.allrecipes.com/recipe/284874/one-bite-thai-flavor-bomb-salad-wraps-miang-kham/']

urls_desserts = ['https://www.allrecipes.com/recipe/17528/extreme-chocolate-cake/','https://www.allrecipes.com/recipe/12682/apple-pie-by-grandma-ople/',
                 'https://www.allrecipes.com/recipe/56803/very-chocolate-ice-cream/','https://www.allrecipes.com/recipe/233928/how-to-make-vanilla-ice-cream/',
                 'https://www.allrecipes.com/recipe/219490/sweet-sticky-rice-with-mangoes/','https://hot-thai-kitchen.com/lod-chong-singapore/',
                 'https://nishkitchen.com/thai-mango-coconut-pudding-video/','https://inquiringchef.com/thai-coconut-pancakes-kanom-krok/',
                 'https://cookingwithlane.com/thai-custard/']
# coconut custard: 'https://www.cooksifu.com/recipe/thai-coconut-custard-with-mung-beans/'

urls_soups = ['https://thaicaliente.com/gang-jued-woon-sen/','https://thatspicychick.com/thai-spicy-pork-rib-soup/',
              'https://www.allrecipes.com/recipe/13081/tom-yum-koong-soup/','https://www.allrecipes.com/recipe/231515/tom-kha-gai/',
              'https://www.allrecipes.com/recipe/49278/japanese-onion-soup/','https://hot-thai-kitchen.com/kai-palo/',
              ]
# 'https://makrohorecaacademy.com/en/recipes/Leng-Zab'

# no nutrients for Kai Palo, Leng Zab

[get_ingredients(url) for url in urls_mains]

[get_nutrients(url) for url in urls_mains]

# for the optimisation I only need nutrient info
# need to calculate this in proportion to calories listed
# ideally in list format 

# for sustainability scores I need ingredients and their amounts, also proportionate to cals listed
# take into account serving size
# can be in dataframe/csv etc 

#%%
# nutrient info
import re 

calories = {
    'PadKrapow': 372,
    'PadThai': 486,
    'HainaneseChickenRice': 597,
    'GreenChickenCurry': 240,
    'ShrimpStickyRice': 477,
    'ShrimpFriedRice': 289,
    'ChickenPorridge': 228,
    'KanomJeenNamYa': 81,
    'ShrimpGlassNoodles': 300,
    'AmericanFriedRice': 790,
    'SausageFriedRice': 610,
    'TomJuedMooSap': 80,
    'KaiPalo': 180,
    'OnionSoup': 180,
    'TomYumKung': 229,
    'TomKhaGai': 357,
    'LengZab': 140,
    'GlassNoodlesSoup': 300,
    'CoconutCustard': 540,
    'KhanomMawKaeng': 244,
    'MangoStickyRice': 270,
    'LodChongThai': 215,
    'KanomKrok': 240,
    'Pudding': 120,
    'VanillaIceCream': 330,
    'ApplePie': 296,
    'ChocolateCake': 424,
    'ChocolateIcecream': 335,
    'FriedCalamari': 187,
    'EggRolls': 480,
    'ShrimpCake': 990,
    'FishCake': 147,
    'FriedShrimpBall': 468,
    'HerbChicken': 376,
    'GrilledShrimp': 125,
    'CrispyCrab': 544,
    'MiangKham': 280,
    'ShrimpDumpling': 300,
    'SaladRoll': 182 # says 1182??
}

    
def nutr_prop(nutrient_dict,dish_name):
    cals = nutrient_dict['calories']
    match = re.search(r'\d+', cals)
    if match:
        cals_numeric = match.group()
    return calories[dish_name]/float(cals_numeric)


def correct_nutrients(prop,nutrient_dict):

    carbs = nutrient_dict.get('carbohydrateContent', '0') # set to 0 if not present in dict
    chol = nutrient_dict.get('cholesterolContent', '0')
    fibre = nutrient_dict.get('fiberContent', '0')
    protein = nutrient_dict.get('proteinContent', '0')
    sat_fat = nutrient_dict.get('saturatedFatContent', '0')
    salt = nutrient_dict.get('sodiumContent', '0')
    sugar = nutrient_dict.get('sugarContent', '0')
    fat = nutrient_dict.get('fatContent', '0')
    unsat_fat = nutrient_dict.get('unsaturatedFatContent', '0')
    
    if carbs is not None:
        match = re.search(r'\d+', carbs)
        if match:
            carbs_n = match.group()
    if chol is not None:  
        match1 = re.search(r'\d+', chol)
        if match:
            chol_n = match1.group()
    if fibre is not None:
        match2 = re.search(r'\d+', fibre)
        if match:
            fibre_n = match2.group()
    if protein is not None:
        match3 = re.search(r'\d+', protein)
        if match:
            protein_n = match3.group()
    if sat_fat is not None:
        match4 = re.search(r'\d+', sat_fat)
        if match:
            sat_fat_n = match4.group()
    if salt is not None:  
        match5 = re.search(r'\d+', salt)
        if match:
            salt_n = match5.group()
    if sugar is not None:
        match6 = re.search(r'\d+', sugar)
        if match:
            sugar_n = match6.group()
    if fat is not None:
        match7 = re.search(r'\d+', fat)
        if match:
            fat_n = match7.group()
    if unsat_fat is not None:
        match8 = re.search(r'\d+', unsat_fat)
        if match:
            unsat_fat_n = match8.group()

    return float(carbs_n)*prop, float(chol_n)*prop, float(fibre_n)*prop, float(protein_n)*prop, float(sat_fat_n)*prop, float(salt_n)*prop, float(sugar_n)*prop, float(fat_n)*prop, float(unsat_fat_n)*prop

dict_choc_cake = get_nutrients(urls_desserts[0])

choc_corr = correct_nutrients(nutr_prop(dict_choc_cake,'ChocolateCake'),dict_choc_cake)

#%%
import pandas as pd

data = {
    'Dish names': ['PadKrapow', 'PadThai', 'HainaneseChickenRice', 'GreenChickenCurry', 'ShrimpStickyRice', 'ShrimpFriedRice', 'ChickenPorridge', 'KanomJeenNamYa', 'ShrimpGlassNoodles', 'AmericanFriedRice', 'SausageFriedRice', 'TomJuedMooSap', 'KaiPalo', 'OnionSoup', 'TomYumKung', 'TomKhaGai', 'LengZab', 'GlassNoodlesSoup', 'CoconutCustard', 'KhanomMawKaeng', 'MangoStickyRice', 'LodChongThai', 'KanomKrok', 'Pudding', 'VanillaIceCream', 'ApplePie', 'ChocolateCake', 'ChocolateIcecream', 'FriedCalamari', 'EggRolls', 'ShrimpCake', 'FishCake', 'FriedShrimpBall', 'HerbChicken', 'GrilledShrimp', 'CrispyCrab', 'MiangKham', 'ShrimpDumpling', 'SaladRoll'],
    'Calories': [372, 486, 597, 240, 477, 289, 228, 81, 300, 790, 610, 80, 180, 180, 229, 357, 140, 300, 540, 244, 270, 215, 240, 120, 330, 296, 424, 335, 187, 480, 990, 147, 468, 376, 125, 544, 280, 300, 182]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add additional columns
df['Carbs'] = None
df['Chol (mg)'] = None
df['Fiber'] = None
df['Protein'] = None
df['Sat_fat'] = None
df['Salt (mg)'] = None
df['Sugar'] = None
df['Fat'] = None
df['Unsat_fat'] = None

def insert_df(name,df,output):
    
    filtered_rows = df.loc[df['Dish names'] == name]
    
    row_index = filtered_rows.index[0]
    
    print(f"The row index for '{name}' is: {row_index}")
        
    df.loc[row_index, 'Carbs'] = output[0]
    df.loc[row_index, 'Chol (mg)'] = output[1]
    df.loc[row_index, 'Fiber'] = output[2]
    df.loc[row_index, 'Protein'] = output[3]
    df.loc[row_index, 'Sat_fat'] = output[4]
    df.loc[row_index, 'Salt (mg)'] = output[5]
    df.loc[row_index, 'Sugar'] = output[6]
    df.loc[row_index, 'Fat'] = output[7]
    df.loc[row_index, 'Unsat_fat'] = output[8]
    
    
dict_mains = [get_nutrients(url) for url in urls_mains]
dict_starters = [get_nutrients(url) for url in urls_starters]
dict_desserts = [get_nutrients(url) for url in urls_desserts]
dict_soups = [get_nutrients(url) for url in urls_soups]

insert_df('PadThai',df, correct_nutrients(nutr_prop(dict_mains[0],'PadThai'),dict_mains[0]))
insert_df('PadKrapow',df, correct_nutrients(nutr_prop(dict_mains[1],'PadKrapow'),dict_mains[1]))
insert_df('HainaneseChickenRice',df, correct_nutrients(nutr_prop(dict_mains[2],'HainaneseChickenRice'),dict_mains[2]))
insert_df('GreenChickenCurry',df, correct_nutrients(nutr_prop(dict_mains[3],'GreenChickenCurry'),dict_mains[3]))
insert_df('ChickenPorridge',df, correct_nutrients(nutr_prop(dict_mains[4],'ChickenPorridge'),dict_mains[4]))
insert_df('SausageFriedRice',df, correct_nutrients(nutr_prop(dict_mains[5],'SausageFriedRice'),dict_mains[5]))
insert_df('ShrimpStickyRice',df, correct_nutrients(nutr_prop(dict_mains[6],'ShrimpStickyRice'),dict_mains[6]))
insert_df('ShrimpFriedRice',df, correct_nutrients(nutr_prop(dict_mains[7],'ShrimpFriedRice'),dict_mains[7]))
insert_df('KanomJeenNamYa',df, correct_nutrients(nutr_prop(dict_mains[8],'KanomJeenNamYa'),dict_mains[8]))
insert_df('ShrimpGlassNoodles',df, correct_nutrients(nutr_prop(dict_mains[9],'ShrimpGlassNoodles'),dict_mains[9]))

insert_df('SaladRoll',df, correct_nutrients(nutr_prop(dict_starters[0],'SaladRoll'),dict_starters[0]))
insert_df('EggRolls',df, correct_nutrients(nutr_prop(dict_starters[1],'EggRolls'),dict_starters[1]))
insert_df('ShrimpCake',df, correct_nutrients(nutr_prop(dict_starters[2],'ShrimpCake'),dict_starters[2]))
insert_df('FriedCalamari',df, correct_nutrients(nutr_prop(dict_starters[3],'FriedCalamari'),dict_starters[3]))
insert_df('FriedShrimpBall',df, correct_nutrients(nutr_prop(dict_starters[4],'FriedShrimpBall'),dict_starters[4]))
insert_df('FishCake',df, correct_nutrients(nutr_prop(dict_starters[5],'FishCake'),dict_starters[5]))
insert_df('HerbChicken',df, correct_nutrients(nutr_prop(dict_starters[6],'HerbChicken'),dict_starters[6]))
insert_df('GrilledShrimp',df, correct_nutrients(nutr_prop(dict_starters[7],'GrilledShrimp'),dict_starters[7]))
insert_df('ShrimpDumpling',df, correct_nutrients(nutr_prop(dict_starters[8],'ShrimpDumpling'),dict_starters[8]))
insert_df('CrispyCrab',df, correct_nutrients(nutr_prop(dict_starters[9],'CrispyCrab'),dict_starters[9]))
insert_df('MiangKham',df, correct_nutrients(nutr_prop(dict_starters[10],'MiangKham'),dict_starters[10]))

dessert_names = ['ChocolateCake','ApplePie','ChocolateIcecream','VanillaIceCream','MangoStickyRice','LodChongThai','Pudding','KanomKrok','KhanomMawKaeng']

[insert_df(dessert_names[i],df,correct_nutrients(nutr_prop(dict_desserts[i],dessert_names[i]),dict_desserts[i])) for i in range(9)]

soup_names = ['GlassNoodlesSoup','TomJuedMooSap','TomYumKung','TomKhaGai','OnionSoup']
[insert_df(soup_names[i],df,correct_nutrients(nutr_prop(dict_soups[i],soup_names[i]),dict_soups[i])) for i in range(5)]

urls_soups = ['https://thaicaliente.com/gang-jued-woon-sen/','https://thatspicychick.com/thai-spicy-pork-rib-soup/',
              'https://www.allrecipes.com/recipe/13081/tom-yum-koong-soup/','https://www.allrecipes.com/recipe/231515/tom-kha-gai/',
              'https://www.allrecipes.com/recipe/49278/japanese-onion-soup/','https://hot-thai-kitchen.com/kai-palo/']

import numpy as np
np.save('nutrients_df.npy',df)

#%%

mains_ingr = [get_ingredients(url) for url in urls_mains] # incl american fried rice
main_names = ['PadThai','PadKrapow','HainaneseChickenRice','GreenChickenCurry','ChickenPorridge','SausageFriedRice','ShrimpStickyRice','ShrimpFriedRice','KanomJeenNamYa','ShrimpGlassNoodles','AmericanFriedRice']

starter_ingr = [get_ingredients(url) for url in urls_starters]
starter_names = ['SaladRoll','EggRolls','ShrimpCake','FriedCalamari','FriedShrimpBall','FishCake','HerbChicken','GrilledShrimp','ShrimpDumpling','CrispyCrab','MiangKham']

dessert_names = ['ChocolateCake','ApplePie','ChocolateIcecream','VanillaIceCream','MangoStickyRice','LodChongThai','Pudding','KanomKrok','KhanomMawKaeng']
urls_desserts = ['https://www.allrecipes.com/recipe/17528/extreme-chocolate-cake/','https://www.allrecipes.com/recipe/12682/apple-pie-by-grandma-ople/',
                 'https://www.allrecipes.com/recipe/56803/very-chocolate-ice-cream/','https://www.allrecipes.com/recipe/233928/how-to-make-vanilla-ice-cream/',
                 'https://www.allrecipes.com/recipe/219490/sweet-sticky-rice-with-mangoes/','https://hot-thai-kitchen.com/lod-chong-singapore/',
                 'https://nishkitchen.com/thai-mango-coconut-pudding-video/','https://inquiringchef.com/thai-coconut-pancakes-kanom-krok/',
                 'https://cookingwithlane.com/thai-custard/','https://www.cooksifu.com/recipe/thai-coconut-custard-with-mung-beans/']
dessert_ingr = [get_ingredients(url) for url in urls_desserts]

soup_names = ['GlassNoodlesSoup','TomJuedMooSap','TomYumKung','TomKhaGai','OnionSoup','KaiPalo']
soup_ingr = [get_ingredients(url) for url in urls_soups]

def split_ingr(ingr_list):
    split_ingredients = []
    for ingredient in ingr_list[0]:
        match = re.match(r'([\d./]+)(?:\s*)(tablespoons|tablespoon|teaspoon|teaspoons|cups|cup|ounces|ounce|oz|Tbsp|tbsp|tsp|Tsp|pounds|pound|grams|g|kilograms)?(?:\s+)(.*)', ingredient, re.IGNORECASE)
        if match:
            amount = match.group(1).strip()
            unit = match.group(2)
            ingredient_name = match.group(3).strip()
            split_ingredients.append([amount, unit, ingredient_name])
    return split_ingredients

df_ingr = pd.DataFrame(columns=['dish', 'quantity', 'measurement', 'ingredient','servings'])

for i in range(len(main_names)):
    ingredients = []  # Store the ingredients for each dish

    split_ingredients = split_ingr(mains_ingr[i])
    dish_name = main_names[i]
    serving_size = mains_ingr[i][1]
    try:
        proportion = nutr_prop(dict_mains[i], dish_name)
    except IndexError:
        proportion = 1
    dish_ingredients = [[amount, unit, ingredient_name, dish_name,serving_size,proportion] for amount, unit, ingredient_name in split_ingredients]

    df_ingr = df_ingr.append(pd.DataFrame(dish_ingredients, columns=['quantity', 'measurement', 'ingredient', 'dish','servings','proportion']), ignore_index=True)

df_ingr_starters = pd.DataFrame(columns=['dish', 'quantity', 'measurement', 'ingredient','servings'])

for i in range(len(starter_names)):
    ingredients = []  # Store the ingredients for each dish

    split_ingredients = split_ingr(starter_ingr[i])
    dish_name = starter_names[i]
    serving_size = starter_ingr[i][1]
    proportion = nutr_prop(dict_starters[i], dish_name)
    dish_ingredients = [[amount, unit, ingredient_name, dish_name,serving_size,proportion] for amount, unit, ingredient_name in split_ingredients]

    df_ingr_starters = df_ingr_starters.append(pd.DataFrame(dish_ingredients, columns=['quantity', 'measurement', 'ingredient', 'dish','servings','proportion']), ignore_index=True)

"""
def create_ingr_df(df,names,ingredients):
    for i in range(len(names)):
        ingredients = []  # Store the ingredients for each dish

        # Process the ingredients for the current dish
        # Assuming `ingr_list` contains the list of ingredients for each dish
        split_ingredients = split_ingr(ingredients[i])
        dish_name = names[i]
        dish_ingredients = [[amount, unit, ingredient_name, dish_name] for amount, unit, ingredient_name in split_ingredients]

        df = df.append(pd.DataFrame(dish_ingredients, columns=['quantity', 'measurement', 'ingredient', 'dish']), ignore_index=True)

create_ingr_df(df_ingr_starters, starter_names, starter_ingr)
"""

df_ingr_desserts = pd.DataFrame(columns=['dish', 'quantity', 'measurement', 'ingredient'])

for i in range(len(dessert_names)):
    ingredients = []  # Store the ingredients for each dish

    split_ingredients = split_ingr(dessert_ingr[i])
    dish_name = dessert_names[i]
    serving_size = dessert_ingr[i][1]
    proportion = nutr_prop(dict_desserts[i], dish_name)
    dish_ingredients = [[amount, unit, ingredient_name, dish_name,serving_size,proportion] for amount, unit, ingredient_name in split_ingredients]

    df_ingr_desserts = df_ingr_desserts.append(pd.DataFrame(dish_ingredients, columns=['quantity', 'measurement', 'ingredient', 'dish','servings','proportion']), ignore_index=True)

df_ingr_soups = pd.DataFrame(columns=['dish', 'quantity', 'measurement', 'ingredient'])

for i in range(len(soup_names)):
    ingredients = []  # Store the ingredients for each dish

    split_ingredients = split_ingr(soup_ingr[i])
    dish_name = soup_names[i]
    serving_size = soup_ingr[i][1]
    try:
        proportion = nutr_prop(dict_soups[i], dish_name)
    except IndexError:
        proportion = 1
    except KeyError:
        proportion = 1
    dish_ingredients = [[amount, unit, ingredient_name, dish_name,serving_size,proportion] for amount, unit, ingredient_name in split_ingredients]

    df_ingr_soups = df_ingr_soups.append(pd.DataFrame(dish_ingredients, columns=['quantity', 'measurement', 'ingredient', 'dish','servings','proportion']), ignore_index=True)

#%%
measurements = {'cup': 240,
                'cups': 240,
                'ounce': 28.35,
                'ounces': 28.35,
                'tablespoon': 14.175,
                'tablespoons': 14.175,
                'tbsp': 14.175,
                'Tbsp': 14.175,
                'teaspoon': 5.69,
                'Teaspoon': 5.69,
                'tsp': 5.69,
                'pound': 453.6,
                'pounds': 453.6,
                'oz': 28.35,
                'g': 1,
                'egg': 44}
import fractions

def convert_to_grams(row):
    quantity = row['quantity']
    measurement = row['measurement']

    try:
        quantity = float(quantity)
    except ValueError:
        try:
            quantity = float(fractions.Fraction(quantity))
        except ValueError:
            quantity = float('nan')

    if measurement is None:
        quantity_grams = None
    elif measurement in measurements:
        quantity_grams = quantity * measurements[measurement]
    else:
        quantity_grams = quantity

    return quantity_grams

df_ingr['grams'] = df_ingr.apply(convert_to_grams, axis=1)
df_ingr['servings'] = df_ingr['servings'].str.extract('(\d+)').astype(int)
df_ingr['grams per serving'] = df_ingr['grams'] / df_ingr['servings']
df_ingr['total grams'] = df_ingr['grams per serving'] * df_ingr['proportion']
df_ingr['grams less than 10'] = df_ingr['total grams'] < 10
df_ingr['grams less than 10'] = df_ingr['grams less than 10'].map({True: 'yes', False: 'no'})

df_ingr_starters['grams'] = df_ingr_starters.apply(convert_to_grams, axis=1)
df_ingr_starters['servings'] = df_ingr_starters['servings'].str.extract('(\d+)').astype(int)
df_ingr_starters['grams per serving'] = df_ingr_starters['grams'] / df_ingr_starters['servings']
df_ingr_starters['total grams'] = df_ingr_starters['grams per serving'] * df_ingr_starters['proportion']
df_ingr_starters['grams less than 10'] = df_ingr_starters['total grams'] < 10
df_ingr_starters['grams less than 10'] = df_ingr_starters['grams less than 10'].map({True: 'yes', False: 'no'})

df_ingr_desserts['grams'] = df_ingr_desserts.apply(convert_to_grams, axis=1)
df_ingr_desserts['servings'] = df_ingr_desserts['servings'].str.extract('(\d+)').astype(int)
df_ingr_desserts['grams per serving'] = df_ingr_desserts['grams'] / df_ingr_desserts['servings']
df_ingr_desserts['grams less than 10'] = df_ingr_desserts['grams per serving'] < 10
df_ingr_desserts['grams less than 10'] = df_ingr_desserts['grams less than 10'].map({True: 'yes', False: 'no'})
df_ingr_desserts['total grams'] = df_ingr_desserts['grams per serving'] * df_ingr_desserts['proportion']
df_ingr_desserts['grams less than 10'] = df_ingr_desserts['total grams'] < 10
df_ingr_desserts['grams less than 10'] = df_ingr_desserts['grams less than 10'].map({True: 'yes', False: 'no'})

df_ingr_soups['grams'] = df_ingr_soups.apply(convert_to_grams, axis=1)
df_ingr_soups['servings'] = df_ingr_soups['servings'].str.extract('(\d+)').astype(int)
df_ingr_soups['grams per serving'] = df_ingr_soups['grams'] / df_ingr_soups['servings']
df_ingr_soups['total grams'] = df_ingr_soups['grams per serving'] * df_ingr_soups['proportion']
df_ingr_soups['grams less than 10'] = df_ingr_soups['total grams'] < 10
df_ingr_soups['grams less than 10'] = df_ingr_soups['grams less than 10'].map({True: 'yes', False: 'no'})

#%%
# saving
"""
df_ingr.to_csv('C:\\Users\\elfre\\ingr_mains.csv', index=False)
df_ingr_starters.to_csv('C:\\Users\\elfre\\ingr_starters.csv', index=False)
df_ingr_desserts.to_csv('C:\\Users\\elfre\\ingr_starters.csv', index=False)
df_ingr_soups.to_csv('C:\\Users\\elfre\\ingr_soups.csv', index=False)

df.to_csv('C:\\Users\\elfre\\nutrients.csv', index=False)

df_ingr = pd.read_csv('ingr_mains.csv')
"""
#%%

# Other ingredients:
    
none_ingr = {
    'Whole chicken': 1400,
    'egg': 44,
    'spring onion (one)': 15,
    'garlic clove': 5,
    'ginger root': 28,
    'chilli pepper': 25,
    'lime juice': 15,
    'onion': 170,
    'lemon': 70,
    'cabbage': 900,
    'carrot': 60,
    'chicken drumstick': 120,
    'apple': 70,
    'pastry': 320,
    'mango': 205,
    'tofu': 280,
    'shallot': 40,
    'button mushroom': 20,
    'zucchini': 323,
    'tofu puff': 20,
    'rice wrapper': 12.8,
    'large shrimp':35,
    'quart': 1130,
    'egg roll wrapper': 25,
    'dumpling wrapper': 12}

df_ingr['measurement'].iloc[4] = 'ounces'
df_ingr['quantity'].iloc[4] = 12

df_ingr['measurement'].iloc[8] = 'ounces'
df_ingr['quantity'].iloc[8] = 12

df_ingr['measurement'].iloc[7] = 'egg'
df_ingr['measurement'].iloc[99] = 'egg'
df_ingr['measurement'].iloc[133] = 'egg'

df_ingr['measurement'].iloc[16] = 'lime juice'
df_ingr['measurement'].iloc[42] = 'lime juice'
df_ingr['measurement'].iloc[47] = 'lime juice'

df_ingr['measurement'].iloc[26] = 'garlic clove'
df_ingr['measurement'].iloc[32] = 'garlic clove'
df_ingr['measurement'].iloc[39] = 'garlic clove'
df_ingr['measurement'].iloc[43] = 'garlic clove'
df_ingr['measurement'].iloc[57] = 'garlic clove'

df_ingr['measurement'].iloc[30] = 'Whole chicken'

df_ingr['measurement'].iloc[31] = 'spring onion (one)'
df_ingr['measurement'].iloc[56] = 'spring onion (one)'

df_ingr['measurement'].iloc[33] = 'ginger root'
df_ingr['measurement'].iloc[40] = 'ginger root'
df_ingr['measurement'].iloc[44] = 'ginger root'
df_ingr['measurement'].iloc[49] = 'ginger root'
df_ingr['measurement'].iloc[67] = 'ginger root'

df_ingr['measurement'].iloc[41] = 'chilli pepper'

df_ingr['measurement'].iloc[65] = 'onion'

df_ingr['measurement'].iloc[66] = 'garlic clove'

df_ingr['measurement'].iloc[134] = 'chicken drumstick'
df_ingr['measurement'].iloc[126] = 'garlic clove'
df_ingr['measurement'].iloc[116] = 'egg'
df_ingr['measurement'].iloc[100] = 'spring onion (one)'
df_ingr['measurement'].iloc[92] = 'tablespoons'
df_ingr['measurement'].iloc[93] = 'teaspoon'
df_ingr['measurement'].iloc[94] = 'spring onion (one)'
df_ingr['measurement'].iloc[87] = 'tablespoons'
df_ingr['measurement'].iloc[88] = 'teaspoon'
df_ingr['measurement'].iloc[89] = 'teaspoon'
df_ingr['measurement'].iloc[90] = 'teaspoon'
df_ingr['measurement'].iloc[82] = 'ounces'
df_ingr['quantity'].iloc[82] = 14.5

df_ingr['measurement'].iloc[83] = 'ounces'
df_ingr['quantity'].iloc[83] = 6

df_ingr['measurement'].iloc[84] = 'spring onion (one)'

df_ingr['measurement'].iloc[72] = 'spring onion (one)'
df_ingr['measurement'].iloc[73] = 'lemon'
df_ingr['measurement'].iloc[76] = 'egg'
df_ingr['measurement'].iloc[78] = 'cabbage'
df_ingr['measurement'].iloc[79] = 'carrot'
df_ingr['measurement'].iloc[48] = 'garlic clove'


def convert_to_grams(row):
    quantity = row['quantity']
    measurement = row['measurement']

    try:
        quantity = float(quantity)
    except ValueError:
        try:
            quantity = float(fractions.Fraction(quantity))
        except ValueError:
            quantity = float('nan')

    if measurement is None:
        quantity_grams = None
    elif measurement in none_ingr:
        quantity_grams = quantity * none_ingr[measurement]
    elif measurement in measurements:
        quantity_grams = quantity * measurements[measurement]
    else:
        quantity_grams = quantity

    return quantity_grams


def convert(df_ingr):
    df_ingr['grams'] = df_ingr.apply(convert_to_grams, axis=1)
    #df_ingr['servings'] = df_ingr['servings'].str.extract('(\d+)').astype(int)
    df_ingr['grams per serving'] = df_ingr['grams'] / df_ingr['servings']
    df_ingr['total grams'] = df_ingr['grams per serving'] * df_ingr['proportion']
    df_ingr['grams more than 10'] = df_ingr['total grams'] > 10 
    df_ingr['grams more than 10'] = df_ingr['grams more than 10'].map({True: 'yes', False: 'no'})

convert(df_ingr)
main_ingredients = df_ingr[df_ingr['grams more than 10']=='yes']

df_ingr_starters['measurement'].iloc[1] = 'rice wrapper'
df_ingr_starters['measurement'].iloc[2] = 'large shrimp'
df_ingr_starters['measurement'].iloc[11] = 'garlic clove' 
df_ingr_starters['measurement'].iloc[22] = 'carrot'
df_ingr_starters['measurement'].iloc[29] = 'egg' 
df_ingr_starters['measurement'].iloc[48] = 'lemon'
df_ingr_starters['measurement'].iloc[50] = 'egg'
df_ingr_starters['measurement'].iloc[56] = 'garlic clove'
df_ingr_starters['measurement'].iloc[58] = 'egg'
df_ingr_starters['measurement'].iloc[67] = 'spring onion (one)'
df_ingr_starters['measurement'].iloc[68] = 'egg'
df_ingr_starters['measurement'].iloc[69] = 'ounces'
df_ingr_starters['quantity'].iloc[69] = 5

df_ingr_starters['measurement'].iloc[70] = 'lemon'
df_ingr_starters['measurement'].iloc[74] = 'garlic clove'
df_ingr_starters['measurement'].iloc[81] = 'lemon'
df_ingr_starters['measurement'].iloc[85] = 'egg'
df_ingr_starters['measurement'].iloc[89] = 'garlic clove'
df_ingr_starters['measurement'].iloc[92] = 'dumpling wrapper'
df_ingr_starters['measurement'].iloc[97] = 'spring onion (one)'
df_ingr_starters['measurement'].iloc[101] = 'ounces'
df_ingr_starters['quantity'].iloc[101] = 8

df_ingr_starters['measurement'].iloc[108] = 'ounces'
df_ingr_starters['quantity'].iloc[108] = 14

df_ingr_starters['measurement'].iloc[109] = 'quart'
df_ingr_starters['measurement'].iloc[114] = 'lime juice'
df_ingr_starters['measurement'].iloc[123] = 'lime juice'
df_ingr_starters['measurement'].iloc[26] = 'carrot'
df_ingr_starters['measurement'].iloc[23] = 'egg roll wrapper'
    
convert(df_ingr_starters)

starter_ingredients = df_ingr_starters[df_ingr_starters['grams more than 10']=='yes']

df_ingr_desserts['measurement'].iloc[6] = 'egg'
df_ingr_desserts['measurement'].iloc[16] = 'apple'
df_ingr_desserts['measurement'].iloc[22] = 'pastry'
df_ingr_desserts['measurement'].iloc[27] = 'egg'
df_ingr_desserts['measurement'].iloc[36] = 'ounces'
df_ingr_desserts['quantity'].iloc[36] = 13.5

df_ingr_desserts['measurement'].iloc[45] = 'cup'
df_ingr_desserts['quantity'].iloc[36] = 0.25

df_ingr_desserts['measurement'].iloc[50] = 'mango'
df_ingr_desserts['measurement'].iloc[55] = 'ounces'
df_ingr_desserts['quantity'].iloc[55] = 15

df_ingr_desserts['measurement'].iloc[59] = 'ounces'
df_ingr_desserts['quantity'].iloc[59] = 15

df_ingr_desserts['measurement'].iloc[61] = 'cup'
df_ingr_desserts['quantity'].iloc[61] = 0.25

df_ingr_desserts['measurement'].iloc[66] = 'egg'

convert(df_ingr_desserts)

dessert_ingredients = df_ingr_desserts[df_ingr_desserts['grams more than 10'] == 'yes']


df_ingr_soups['measurement'].iloc[69] = 'tofu puff'
df_ingr_soups['measurement'].iloc[68] = 'garlic clove'
df_ingr_soups['measurement'].iloc[49] = 'carrot'
df_ingr_soups['measurement'].iloc[48] = 'onion'
df_ingr_soups['measurement'].iloc[44] = 'zucchini'
df_ingr_soups['measurement'].iloc[42] = 'ounces'
df_ingr_soups['quantity'].iloc[42] = 15
df_ingr_soups['measurement'].iloc[39] = 'ginger root'

df_ingr_soups['measurement'].iloc[32] = 'ounces'
df_ingr_soups['quantity'].iloc[32] = 6
df_ingr_soups['measurement'].iloc[31] = 'button mushroom'
df_ingr_soups['measurement'].iloc[27] = 'lime juice'
df_ingr_soups['measurement'].iloc[25] = 'chilli pepper'
df_ingr_soups['measurement'].iloc[19] = 'green onion (one)'
df_ingr_soups['measurement'].iloc[16] = 'tbsp'
df_ingr_soups['measurement'].iloc[15] = 'tbsp'
df_ingr_soups['measurement'].iloc[14] = 'tbsp'
df_ingr_soups['measurement'].iloc[13] = 'chilli pepper'
df_ingr_soups['measurement'].iloc[12] = 'garlic clove'
df_ingr_soups['measurement'].iloc[10] = 'shallot'
df_ingr_soups['measurement'].iloc[2] = 'garlic clove'

df_ingr_soups['measurement'].iloc[0] = 'pound'
df_ingr_soups['quantity'].iloc[0] = 1

df_ingr_soups['measurement'].iloc[6] = 'tofu'
df_ingr_soups['measurement'].iloc[34] = 'ounces'
df_ingr_soups['quantity'].iloc[34] = 14

convert(df_ingr_soups)
soup_ingredients = df_ingr_soups[df_ingr_soups['grams more than 10'] == 'yes']

#%%


all_ingredients = pd.concat([starter_ingredients,soup_ingredients,main_ingredients,dessert_ingredients],ignore_index=True)
all_ingredients['normalised ingredient'] = ''

len(all_ingredients['normalised ingredient'].unique()) # 69 unique ingredients across 37 dishes. So all use quite similar ingredients

# scores with 0 could not find scores for these foods 
# maybe would make more sense to impute average score?

scores = {'rice noodles': 137, 
          'rice paper': 0, 
          'shrimp': 1775, 
          'mint': 113, 
          'coriander': 113,
           'water': 0.02, 
           'hoisin': 22, 
           'pork': 663, 
           'cabbage': 62, 
           'carrot': 113, 
           'peanut oil': 364,   
           'egg roll': 0, 
           'egg': 451, 
           'oil': 320, 
           'flour': 89, 
           'squid': 304, 
           'bread crumbs': 121,
           'cod': 321,
           'chicken': 471, 
           'lemon': 190, 
           'olive oil': 253, 
           'edamame': 200, 
           'bok choy': 0,       
           'dumpling wrapper': 0,
           'soy sauce': 22, 
           'cream cheese': 644, 
           'coconut': 79,
           'sugar': 66,
           'peanuts': 136, 
           'lime': 190, 
           'shallot': 105,
           'ginger': 45, 
           'tofu': 186, 
           'chilli': 202,         
           'fish sauce': 22, 
           'paste': 22,
           'mushrooms': 140, 
           'chicken stock': 22,     
           'coconut milk':79 , 
           'chilli sauce': 22,
           'onion': 105 , 
           'aubergine': 310, 
           'chives': 113,         
           'vinegar': 22,
           'bean sprouts': 0, 
           'basil': 113,
           'rice': 135, 
           'sticky rice': 135 , 
           'sausage': 663,
           'bean sprout': 0,
           'peas': 251, 
           'honey': 166, 
           'spring onion': 105,          
           'noodles': 137,
           'celery': 84,
           'corn': 41, 
           'raisins': 210,
           'ketchup': 154,
           'milk': 176,         
           'cocoa powder': 444, 
           'apple': 90, 
           'butter': 976, 
           'pastry': 658, 
           'double cream': 372,          
           'tapioca': 59, 
           'mango': 440, 
           'jasmine rice': 135, 
           'rice flour': 89, 
           'coconut cream': 187,
           'crab':1175}
    
def convert_to_score(row):
    ingr = row['normalised ingredient']
    quantity = row['total grams']

    if ingr in scores:
        s_score = quantity * scores[ingr]/100
    
    return s_score

all_ingredients['sustainability_score'] = all_ingredients.apply(convert_to_score,axis=1)

total_scores = all_ingredients.groupby('dish')['sustainability_score'].sum()
total_scores_df = total_scores.reset_index(name='total_score')
total_scores_df.sort_values(by='total_score')

all_ingredients.to_csv('C:\\Users\\elfre\\all_ingr.csv', index=False)
total_scores_df.to_csv('C:\\Users\\elfre\\final_scores.csv', index=False)

all_ingredients['Vegetarian'] = 'N'
all_ingredients['Pescetarian'] = 'Y'

#%%
import itertools
combinations = list(itertools.product(starter_names, main_names, dessert_names))

for combination in combinations:
    print(combination)

len(combinations) # 1089 combinations
