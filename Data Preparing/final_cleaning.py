import pandas as pd
data = pd.read_csv('./code/Data Preparing/data_8.csv')
columns = ['implementation_avg_rating','min_players_rec_diff','max_players_rec_diff']
if True:
    for col in columns:
        col_to_move = data.pop(col)
        data.insert(12, f'{col}', col_to_move)
    col_to_move = data.pop('bgg_id')
    data.insert(1, 'bgg_id', col_to_move)
    col_to_move = data.pop('avg_rating')
    data.insert(2, 'avg_rating', col_to_move)
    col_to_move = data.pop('cooperative')
    data.insert(15, 'cooperative', col_to_move)
data['cooperative'] = data['cooperative'] == True
# Check Group 1: columns 2 to 14 should contain only floats
print("Checking columns expected to contain floats (columns 3 to 14):")
for col in data.columns[2:14]:
    non_null = data[col].dropna()
    # Check if every non-null value is a float
    all_floats = non_null.apply(lambda x: isinstance(x, float)).all()
    missing_count = data[col].isna().sum()
    
    if all_floats:
        print(f"Column '{col}': OK - All non-missing values are floats. Missing values: {missing_count}.")
    else:
        print(f"Column '{col}': ERROR - Contains non-float values! Missing values: {missing_count}.")

# Check Group 2: columns 15 onward should contain only booleans
print("\nChecking columns expected to contain booleans (columns 15 onward):")
for col in data.columns[15:]:
    non_null = data[col].dropna()
    # Check if every non-null value is a boolean
    all_bools = non_null.apply(lambda x: isinstance(x, bool)).all()
    missing_count = data[col].isna().sum()
    if all_bools:
        print(f"Column '{col}': OK - All non-missing values are booleans. Missing values: {missing_count}.")
    else:
        print(f"Column '{col}': ERROR - Contains non-boolean values or missing values! Missing values: {missing_count}.")

# Save the updated data to a new CSV file
data.to_csv(f'./code/Data Preparing/data_9.csv', index=False)