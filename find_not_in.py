import pandas as pd

users = pd.read_csv('data/user_profile.csv', header=None, names=['user_id', 'age', 'gender_id', 'job_id', 'city_id', 'label'])
items = pd.read_csv('data/item_profile.csv', header=None, names=['item_id', 'category_id', 'city_id', 'label'])
interactions = pd.read_csv('data/interactions.csv')

not_in_users = iteractions