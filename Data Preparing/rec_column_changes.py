import pandas as pd
data = pd.read_csv('./code/Data Preparing/data_7.csv')
columns = ['min_players_rec', 'max_players_rec', 'min_age_rec', 'max_age_rec', 'min_players', 'max_players', 'min_age', 'max_age']

data['min_players_rec_diff'] = data['min_players_rec'] - data['min_players']
data['max_players_rec_diff'] = data['max_players_rec'] - data['max_players']
drop_columns = ['min_players_rec', 'max_players_rec', 'min_age_rec', 'max_age_rec', 'max_age']
data.drop(columns=drop_columns, inplace=True)
data.to_csv(f'./code/Data Preparing/data_changed_rec.csv', index=False)