import pandas as pd
drop_columns = ['compilation', 'expansion', 'implementation', 'integration', 'rank', 'add_rank', 'num_votes','stddev_rating','bayes_rating','worst_rating','best_rating','easiest_complexity','hardest_complexity','freebase_id','wikidata_id','wikipedia_id','dbpedia_id','luding_id','spielen_id','published_at','updated_at','scraped_at','min_players_best','max_players_best','lowest_language_dependency','highest_language_dependency']
drop_columns2 = ['cooperative']
data = pd.read_csv('./code/Data Preparing/data.csv')
data.drop(columns=drop_columns2, inplace=True)

# Save the updated data to a new CSV file
data.to_csv(f'./code/Data Preparing/data.csv', index=False)