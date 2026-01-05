import pandas as pd
import numpy as np
import ast
import re
from collections import Counter

# Load the data from the CSV file
file_path = './code/Data Preparing/data_5.csv'
data = pd.read_csv(file_path)

feature = 'mechanic'
IsList = True
#examine = False
examine = False
use_count_binning = False
skip = True

#convert strings in data to lists
#this is used because in my dataset most lists are actually strings so to work with them i need to convert before
def convert_to_list(x):
    if isinstance(x, bool):
        return []  # Handle booleans explicitly
    if pd.notna(x) and isinstance(x, str):
        try:
            result = ast.literal_eval(x) #convert to list if possible
            # If ast.literal_eval returns a boolean, handle it
            if isinstance(result, bool):
                return []
            return result
        except (SyntaxError, ValueError):
            return []  # Return an empty list if conversion fails
    return [] if pd.isna(x) else (x if isinstance(x, list) else [])



if IsList:
    data[feature] = data[feature].map(convert_to_list)

    #this was used to count how often each category appears in the feature (some of this was used if the feature holds a list of categories
    if examine:
        unique_feature_types = set(item for sublist in data[feature].dropna() if isinstance(sublist, list) for item in sublist)
        print(f'There are {len(unique_feature_types)} unique {feature}.')
        designer_names = []
        for lst in data[feature]:
            if isinstance(lst, list):
                for item in lst:
                    if isinstance(item, str) and ':' in item:
                        name = item.split(':')[0].strip()
                        if name != '(Uncredited)':
                            designer_names.append(name)
        counts = Counter(designer_names)
        print(f'There are {len(counts)} unique {feature}.')
        print(counts.items())
        filtered_feature_counts = {k: v for k, v in counts.items() if v > 100}
        print(f'There are {len(filtered_feature_counts)} {feature} with counts greater than 100.')
        print(counts.most_common(200))

#this was used to get the keywords in the 'family' feature
def extract_keywords(lst):
    keywords = set()
    for row in lst:
        for item in row:
            keywords.add(re.split(r':', item)[0].strip())
    return keywords

def create_keyword_counter(data):
    all_feature_types = [item for sublist in data[feature].dropna() if isinstance(sublist, list) for item in sublist]
    feature_counts = Counter(all_feature_types)
    return feature_counts

def keywords(generate = False):
    if generate:
        return extract_keywords(data[f'{feature}'].dropna())
    return ['Crowdfunding', 'Movies', 'Political', 'Traditional Card Games', 'Holidays', 
    'Historical Figures', 'Space', 'Sports Teams', 'Sports', 'Food & Drink', 
    'Video Game Theme', 'TV Shows', 'Medical', 'Region', 'Toys', 
    'Comic Books', 'Mythology', 'Folk Tales & Fairy Tales', 'Books', 
    'Religious', 'Ancient', 'Collectible']

def create_binning_columns(df, column, use_binning=True):
    """
    Creates binary columns for designers based on their overall frequency in the dataset.
    For designers with more than 500 appearances, creates individual binary columns.
    For the remaining designers, bundles them into bins: 100-500, 10-99, 3-9, and less than 3.
    The designer 'Uncredited' is disregarded.
    
    Args:
       df (pd.DataFrame): DataFrame containing the designer column.
       column (str): The column name that holds the designer data in the format 'designer:number'.
       use_binning (bool): If True, apply the binning and bundling; if False, leave the data unchanged.
    
    Returns:
       pd.DataFrame: Updated DataFrame with new binary columns.
    """
    # Extract designer names (ignoring the number) and filter out 'Uncredited'
    designer_names = []
    for lst in df[column]:
        if isinstance(lst, list):
            for item in lst:
                if isinstance(item, str) and ':' in item:
                    name = item.split(':')[0].strip()
                    if name != '(Uncredited)':
                        designer_names.append(name)
    counts = Counter(designer_names)
    
    if use_binning:
        # Define bins based on counts
        individual_designers = {name for name, cnt in counts.items() if cnt > 600}
        bin_very_high_xp = {name for name, cnt in counts.items() if 300 <= cnt <= 600}
        bin_high_xp = {name for name, cnt in counts.items() if 50 <= cnt <= 299}
        bin_average_xp = {name for name, cnt in counts.items() if 3 <= cnt <= 49}
        bin_low_xp = {name for name, cnt in counts.items() if cnt < 3}
        
        # Create individual binary columns for each designer with >500 appearances
        for designer in individual_designers:
            col_name = f"{column}_{designer.replace(' ', '_')}"
            df[col_name] = df[column].apply(
                lambda lst: any(
                    (isinstance(item, str) and item.split(':')[0].strip() == designer)
                    for item in lst
                ) if isinstance(lst, list) else False
            )
        
        # Create bundled binary columns for the remaining designers
        df[f"{column}_very_high_xp"] = df[column].apply(
            lambda lst: any(
                (isinstance(item, str) and item.split(':')[0].strip() in bin_very_high_xp)
                for item in lst
            ) if isinstance(lst, list) else False
        )
        df[f"{column}_high_xp"] = df[column].apply(
            lambda lst: any(
                (isinstance(item, str) and item.split(':')[0].strip() in bin_high_xp)
                for item in lst
            ) if isinstance(lst, list) else False
        )
        df[f"{column}_average_xp"] = df[column].apply(
            lambda lst: any(
                (isinstance(item, str) and item.split(':')[0].strip() in bin_average_xp)
                for item in lst
            ) if isinstance(lst, list) else False
        )
        df[f"{column}_low_xp"] = df[column].apply(
            lambda lst: any(
                (isinstance(item, str) and item.split(':')[0].strip() in bin_low_xp)
                for item in lst
            ) if isinstance(lst, list) else False
        )
    else:
        # Here you could implement the original functionality if needed.
        pass
    
    return df
#creates a new column for each keyword in the list
if examine == False and use_count_binning == False and skip ==False:
    for keyword in keywords(generate = True):
        print(keyword)
        data[f'{feature}_{keyword}'] = data[f'{feature}'].apply(lambda lst: any(item.startswith(keyword + ':') for item in lst))
elif examine == False and use_count_binning == True:
    create_binning_columns(data, feature, use_binning=True)

"""
#this was used to get the subkeywords in the 'family' feature and their counts
feature_name = 'Theme'

def extract_feature_descriptions(lst, feature_name):
    descriptions = []
    for item in lst:
        if item.startswith(f'{feature_name}:'):
            # Extract the description between the first and second colon
            match = re.search(rf'{re.escape(feature_name)}:([^:]+):', item)
            if match:
                descriptions.append(match.group(1).strip())
    return descriptions

all_descriptions = [
    desc for sublist in data['family'].apply(lambda x: extract_feature_descriptions(x, feature_name)) for desc in sublist
    ]

# Count frequency of each description
description_counts = Counter(all_descriptions)

# Output the result
print(f"\n{feature_name} Descriptions and Their Frequencies:")
for description, count in description_counts.most_common():
    print(f"{description}: {count}")
"""

def create_description_columns(df, column_name, keyword, descriptions=None, bundles=None):
    """
    Creates binary columns for provided descriptions of a given keyword, bundles specified descriptions, 
    and an 'Other' column for the rest.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Name of the column containing lists of 'keyword:description:number'.
        keyword (str): The keyword whose descriptions we want to encode.
        descriptions (list): List of individual descriptions to create binary columns for.
        bundles (list of lists): List of bundles where each bundle is a list of descriptions to combine.
        
    Returns:
        pd.DataFrame: Updated dataframe with new binary columns.
    """
    # Defaults
    if descriptions is None:
        descriptions = []
    if bundles is None:
        bundles = []

    # Ensure column is list-like and drop NaNs
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])

    # Extract only descriptions for the given keyword
    keyword_items = df[column_name].apply(lambda x: [item.split(':')[1].strip() for item in x if item.startswith(keyword + ':')])

    # Track used descriptions
    used_descriptions = set()

    # Store new columns in a dictionary to append later
    new_columns = {}

    # Create binary columns for each individual description
    for desc in descriptions:
        col_name = f'{column_name}_{keyword}_{desc.replace(" ", "_").replace(",", "_")}'
        new_columns[col_name] = keyword_items.apply(lambda lst: any(item == desc for item in lst))
        used_descriptions.add(desc)
        #count = new_columns[col_name].sum()
        #print(f"{desc}: {count}")

    # Create binary columns for each bundle
    for bundle in bundles:
        bundle_name = '_'.join(bundle).replace(" ", "_").replace(",", "_")
        col_name = f'{column_name}_{keyword}_{bundle_name}'
        new_columns[col_name] = keyword_items.apply(lambda x: any(desc in x for desc in bundle))
        used_descriptions.update(bundle)
        #count = new_columns[col_name].sum()
        #print(f"{bundle_name}: {count}")

    # Create binary column for "Other" descriptions not in provided lists
    new_columns[f'{column_name}_{keyword}_Other'] = keyword_items.apply(lambda x: any(desc not in used_descriptions for desc in x))
    #count = new_columns[f'{column_name}_{keyword}_Other'].sum()
    #print(f"Other: {count}")

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df

def create_keyword_columns(df, column_name, keywords=None, bundles=None):
    """
    Creates binary columns for provided keywords from a given feature column that contains lists 
    of strings formatted as 'keyword:description:number'. Also creates binary columns for bundles of keywords 
    (combining multiple keywords) and an 'Other' column for any keywords not included in the provided lists.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Name of the column containing lists of 'keyword:description:number'.
        keywords (list): List of individual keywords to create binary columns for.
        bundles (list of lists): List of bundles where each bundle is a list of keywords to combine.
        
    Returns:
        pd.DataFrame: Updated dataframe with new binary columns.
    """
    # Set defaults
    if keywords is None:
        keywords = []
    if bundles is None:
        bundles = []
    
    # Ensure the column is list-like
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])
    
    # Extract the keyword from each item in the list (i.e. the part before the first colon)
    keyword_items = df[column_name].apply(lambda x: [item.split(':')[0].replace('"', '').strip() for item in x])
    
    # To keep track of keywords that are handled
    used_keywords = set()
    
    # Dictionary to collect the new binary columns
    new_columns = {}
    
    # Create a binary column for each individual keyword
    for kw in keywords:
        col_name = f'{column_name}_{kw.replace(" ", "_").replace(",", "_")}'
        new_columns[col_name] = keyword_items.apply(lambda lst: any(item == kw for item in lst))
        used_keywords.add(kw)
    
    # Create binary columns for each bundle of keywords
    for bundle in bundles:
        # Bundle name is made by joining the keywords
        bundle_name = '_'.join(bundle).replace(" ", "_").replace(",", "_")
        col_name = f'{column_name}_{bundle_name}'
        new_columns[col_name] = keyword_items.apply(lambda lst: any(bkw in lst for bkw in bundle))
        used_keywords.update(bundle)
    
    # Create a binary column for "Other" keywords not in the provided keywords/bundles
    #new_columns[f'{column_name}_Other'] = keyword_items.apply(lambda lst: any(kw not in used_keywords for kw in lst))

    # Append the new columns to the dataframe
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df

keywords = ['Creatures', 'Components', 'Category', 'Animals', 'Continents', 'Country', 'Players', 'Brands', 'Mechanism', 'History', 'Theme']

descriptions_above_cutoff = [['Monsters', 'Dragons', 'Aliens / Extraterrestrials', 'Zombies'],
                                       ['Miniatures', 'Map (Continental / National scale)', '3-Dimensional (3D)', 'Dice with Icons', 'Wooden pieces & boards', 'Digital Hybrid – App/Website Required', 'Traditional Playing Cards', 'Sand Timers'],
                                       ['Combinatorial', 'Dungeon Crawler', 'Escape Room Games'],
                                       ['Cats', 'Dinosaurs', 'Dogs'],
                                       ['Europe', 'Asia'],
                                       ['USA', 'France', 'Germany', 'Japan', 'Russia', 'Italy', 'England', 'Soviet Union', 'Spain', 'United Kingdom', 'Belgium', 'China'],
                                       ['Two-Player Only Games', 'Games with Solitaire Rules', 'Wargames with Rules Supporting Only Two Players', 'Solitaire Only Games', 'Wargames with Solitaire Rules', 'Expansions Changing Player Count', 'Solitaire Only Wargames'],
                                       ['Disney'],
                                       ['Campaign Games', 'Roll-and-Write', 'Tableau Building'],
                                       [],
                                       ['Food / Cooking', 'Superheroes', 'Cthulhu Mythos', 'Anime / Manga', 'Anthropomorphic Animals', 'Mystery / Crime', 'Time Travel']
                                       ]

bundles_for_descriptions_below_cutoff = [[['Vampires', 'Ghosts']],
                                  [['Gems/Crystals', 'Marbles'], ['Game Trayz Inside', 'Game Box Used In Play'], ['5 x 5 Grids', '6 x 6 Grids', '7 x 7 Grids', '4 x 4 Grids', '8 x 8 Grids', '9 x 9 Grids', '10 x 10 Grids', '3 x 3 Grids', '11 x 11 Grids', '12 x 12 Grids', 'HexHex5 Grids', '15 x 15 Grids', 'HexHex4 Grids', 'HexHex7 Grids', '13 x 13 Grids', '5 x 6 Grids', '14 x 14 Grids', 'HexHex3 Grids', 'HexHex8 Grids', '16 x 16 Grids', 'HexHex6 Grids', '19 x 19 Grids', '6 x 7 Grids', '7 x 8 Grids', '6 x 8 Grids', '17 x 17 Grids', '7 x 9 Grids', '8 x 10 Grids', '4 x 8 Grids', '9 x 10 Grids', '20 x 20 Grids', '4x4x4 Grids', '8 x 9 Grids', 'HexHex10 Grids', '8 x 11 Grids', '4 x 5 Grids', '5 x 7 Grids', '2 x 2 Grids', '9 x 12 Grids', '3x3x3 Grids', '4 x 6 Grids', '8 x 12 Grids', '5x5x5 Grids', '10 x 20 Grids', '5 x 8 Grids', '18 x 18 Grids', '11 x 12 Grids', '8 x 16 Grids', '7 x 10 Grids', '14 x 15 Grids', '12 x 18 Grids', '9 x 11 Grids', '10 x 11 Grids', '7 x 13 Grids', '6 x 9 Grids', '3 x 6 Grids', '2 x 3 Grids', '7 x 14 Grids', '15 x 17 Grids', '14 x 16 Grids', '5 x 10 Grids', '21 x 21 Grids', '10 x 12 Grids', '24 x 24 Grids', '10 x 15 Grids', '3 x 5 Grids', 'HexHex11 Grids', '11 x 13 Grids', '4 x 7 Grids', '12 x 14 Grids', '6 x 12 Grids', '12 x 16 Grids', '4x4x4x4 Grids', '12 x 13 Grids'], ['Polyominoes', 'Hexagonal Tiles', 'Triangular Dominoes']],
                                  [['Tower Defense', 'Two-Player Fighting Games']],
                                  [['Pigs', 'Sheep', 'Cattle / Cows', 'Horses', 'Camels', 'Goats', 'Donkeys', ],['Wolves', 'Bears', 'Foxes', 'Elephants', 'Pandas', 'Deer / Antelope', 'Tigers', 'Lions', 'Hippopotamuses', 'Raccoons', 'Llamas', 'Giraffes', 'Rhinoceroses', 'Kangaroos', 'Mammoths', 'Zebras', 'Koalas', 'Moose / Elk', 'Sloths', 'Otters', 'Wombats', 'Coyotes'],['Apes / Monkeys', 'Rabbits / Hares', 'Mice', 'Rats', 'Hedgehogs / Porcupines', 'Squirrels', 'Bats', 'Hamsters', 'Badgers', 'Beavers', 'Moles', 'Marmots', 'Gophers', 'Skunks'], ['Turtles', 'Snakes', 'Alligators / Crocodiles', 'Lizards', 'Chameleons'],['Birds', 'Chickens / Roosters', 'Penguins', 'Ducks', 'Crows / Ravens / Magpies', 'Owls', 'Geese', 'Eagles', 'Ostriches', 'Pigeons', 'Turkeys', 'Emus'],['Fish / Fishes', 'Sharks', 'Dolphins', 'Seals', 'Whales'], ['Insects', 'Bees', 'Ants', 'Butterflies', 'Flies', 'Cockroaches', 'Ladybugs / Ladybirds', 'Fleas', 'Grasshoppers', 'Mosquitoes']],
                                  [],
                                  [['Greece', 'Portugal', 'Austria', 'Italy', 'Spain'], ['Poland', 'Ukraine', 'Prussia', 'Hungary', 'Czech Republic', 'Austria-Hungary'], ['Netherlands', 'Scotland', 'Great Britain', 'Wales', 'Ireland', 'Luxembourg'], ['Canada', 'Mexico', 'Cuba', 'Puerto Rico'], ['Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia', 'Uruguay', 'Australia', 'New Zealand', 'Solomon Islands', 'Papua New Guinea'], ['India', 'Vietnam', 'Korea', 'Turkey', 'Israel', 'Iran', 'Iraq'], ['Egypt', 'South Africa', 'Libya', 'Morocco', 'Kenya', 'Ethiopia']],
                                  [['Expansions Adding Solo Play', 'Games with expansions that add solo play', 'Expansions Adding Cooperative Play']],
                                  [],
                                  [['Judging Games', '4X', 'Dice Drafting', 'Legacy', 'Hidden Movement'], ['Finger Flicking Games', 'Tower Stacking', 'Modelling/Shaping', 'Partially overlapping cards'], ['Flip-and-Write', 'Drawing', 'Story Creation', 'Storytelling dice game', 'Storytelling card game', 'Image Interpretation'], ['Give a Clue / Get a Clue', 'Connection', 'Deduction – Blind Man\'s Bluff', 'Bluffing – Dice Cup']],
                                  [['World War II (Eastern Front)', 'World War II (Pacific)', 'World War II (Africa)', 'World War II (Battle of the Bulge)', 'World War II (Italy)', 'World War II (Poland 1939)', 'World War II (Battle of Britain)', 'World War II (Western Front 1940)', 'World War II (Italian Resistance)', 'Warsaw Uprising (Poland, 1944)'], ['Cold War', 'Vietnam War', 'Yom Kippur War', 'Iraq War', 'Iran-Iraq War', 'Gulf War', 'Syrian Civil War', 'Afghanistan War', 'Lebanese Wars 1975-2006', 'Conflicts against the Islamic State', 'Soviet-Afghan War', 'Russo-Ukrainian War'], ['Hundred Years Wars', 'Wars of the Roses', 'Crusades', 'Byzantium', 'Norman Conquest', 'Spanish Civil War', 'Anglo-Zulu War', 'Holy Roman Empire', 'Reconquista', 'Wars of Scottish Independence', 'Wars of the Guelphs and Ghibellines'], ['Roman Empire', 'Punic Wars', 'Roman Civil Wars', 'Roman – Gallic Wars', 'Wars of Alexander the Great', 'Trojan War', 'Greco-Persian Wars (5th century BC)', 'Greek & Persian conflicts of the 4th century BC', 'Early Imperial China', 'Peloponnesian War', 'Macedonian Wars', 'Sicilian Wars', 'Early Muslim conquests'], ['Napoleonic Wars', 'French Revolutionary Wars', 'Seven Years War', 'French and Indian War', 'War of the Spanish Succession', 'American Revolution', 'War of 1812', 'Crimean War', 'Mexican-American War', 'Franco-Prussian War']],
                                  [['Nature', 'Gardening', 'Trees and Forests', 'Deserts', 'Under the Sea', 'Zoos, Aquaria, Safari Parks', 'Volcanoes', 'Ecology', 'Climate Change', 'Endangered / Extinct species', 'Beaches', 'Cacti and Succulents', 'Firefighting', 'Weather', 'Fishing / Angling', 'Hiking', 'Camping', 'US National Parks'], ['Robots', 'Mech Warfare', 'Cyberpunk', 'Mad Science / Mad Scientist', 'Biology', 'Chemistry', 'Physics', 'Astronomy', 'Computer / Information Technology Industry', 'Electricity Generation', 'Hackers', 'Nuclear option', 'Submarines', 'Oil / Gas / Petroleum', 'Teaching Programming', 'Television (TV) Industry', 'Motorcycles', 'Trucks'], ['Witches', 'Kaiju', 'Cryptids', 'King Arthur / The Knights of the Round Table / Camelot', 'Druids', 'Samurai', 'Ninjas', 'Villainy', 'Boss Battle', 'Spooky Old Houses', 'Hell', 'Dreams / Nightmares', 'Chibis', 'Alchemy', 'Psychic Powers', 'Geisha', 'Cannibals / Cannibalism', 'Gladiators'], ['Vikings', 'Native Americans / First Peoples', 'Mayans', 'Aztecs', 'Māori', 'Inuit Peoples', 'African Americans', 'Apache Tribes', 'Aboriginal Australians', 'Silk Road', 'Hanseatic League', 'Romance of the Three Kingdoms', 'King Arthur / The Knights of the Round Table / Camelot', 'Chivalry / Jousting / Tournaments (Medieval Europe)', 'Orient Express', 'Knights Templar'], ['Pirates', 'Treasure Diving', 'Safaris', 'Survival', 'Camping', 'Fishing / Angling', 'Submarines', 'Airships / Blimps / Dirigibles / Zeppelins', 'Hot Air Balloons', 'Floating islands in the sky', 'Temple', 'Motorcycles', 'Helicopters', 'Cruise ships', 'Arcade Video Games', 'Geocaching', 'Rivers'], ['Fantasy Sports', 'Sci-Fi Sports', 'Boardgaming', 'FIFA World Cup', 'Animal Battles', 'Boss Battle', 'Rubik\'s Cube', 'Cereal Games', 'Fictional Games', 'Traffic / Driving', 'Battle Royale', 'Rodeos'], ['Art', 'Painting / Paintings', 'Books / Libraries', 'Fine Art and Art Museums', 'Photography', 'Music Making & Makers', 'Circus', 'Amusement Parks / Theme Parks', 'Movie Industry', 'Television (TV) Industry', 'Memes', 'Disney Theme Parks', 'Fashion', 'Jewelry', 'Dolls', 'Perfume', 'Sewing / Knitting / Cloth-Making', 'Origami', 'Birthdays'], ['City', 'Construction', 'Mining', 'Police', 'Burglary and Heists', 'Jail / Prison (Modern)', 'Mail / Stamps / The Post Office', 'Hotels, Inns, Lodgings', 'Pub, Bars, Bistros...', 'Firefighting', 'Attorneys / Courts', 'Smuggling', 'School / College / University', 'Psychology', 'Stock Market', 'Trash / Garbage', 'Toilets / Urinals', 'Hair / Beards / Animal Fur', 'Traffic / Driving'], ['Retro', 'Steampunk', 'Dieselpunk', 'Solarpunk', 'Cyberpunk', 'Pulp', 'Memes', 'Arcade Video Games', 'Fictional Languages', 'Disney Theme Parks', 'Care Bears', 'Rubik\'s Cube', 'Kaiju', 'Villainy', 'Boss Battle', 'Hackers', 'TV Detectives', 'Movies', 'UFOs', 'Queer / LGBTAIQ', 'Space']]
                                  ]

individual_mechanic_keywords = [
    "Dice Rolling",
    "Hand Management",
    "Variable Player Powers",
    "Set Collection",
    "Cooperative Game",
    "Hexagon Grid",
    "Open Drafting",
    "Modular Board",
    "Simulation",
    "Tile Placement",
    "Grid Movement",
    "Action Points",
    "Area Majority / Influence",
    "Solo / Solitaire Game",
    "Simultaneous Action Selection",
    "Roll / Spin and Move",
    "Take That",
    "Deck, Bag, and Pool Building",
    "Area Movement",
    "Team-Based Game",
    "Push Your Luck",
    "Point to Point Movement",
    "Auction / Bidding",
    "Worker Placement",
    "Memory",
    "Player Elimination",
    "Role Playing",
    "Pattern Building",
    "Storytelling",
    "Scenario / Mission / Campaign Game",
    "Trading",
    "Paper-and-Pencil",
    "Campaign / Battle Card Driven",
    "Variable Set-up",
    "Pick-up and Deliver",
    "Action Queue",
    "Network and Route Building",
    "Pattern Recognition",
    "Secret Unit Deployment",
    "Movement Points",
    "Betting and Bluffing",
    "Voting",
    "Variable Phase Order",
    "Trick-taking",
    "Turn Order",
    "Events",
    "Deduction",
    "End Game Bonuses",
    "Square Grid",
    "Race",
    "Line of Sight",
    "Acting",
    "Zone of Control",
    "Stock Holding",
    "Ratio / Combat Results Table",
    "Real-Time",
    "Card Play Conflict Resolution",
    "Contracts",
    "Chit-Pull System",
    "Deck Construction",
    "Commodity Speculation",
    "Income",
    "Enclosure",
    "Measurement Movement",
    "Rock-Paper-Scissors",
    "Communication Limits",
    "Line Drawing",
    "Market",
    "Action / Event",
    "Track Movement",
    "Hidden Roles",
    "Closed Drafting",
    "Critical Hits and Failures",
    "Action Drafting",
    "Traitor Game",
    "Spelling",
    "Tech Trees / Tech Tracks",
    "Once-Per-Game Abilities",
    "Connections",
    "Grid Coverage",
    "Narrative Choice / Paragraph",
    "Flicking",
    "Ladder Climbing",
    "Three Dimensional Movement",
    "Score-and-Reset Game",
    "Alliances",
    "Command Cards",
    "Stat Check Resolution",
    "Singing",
    "Tags",
    "I Cut, You Choose",
    "Action Timer",
    "Catch the Leader",
    "Legacy Game",
    "Lose a Turn",
    "King of the Hill",
    "Advantage Token",
    "Single Loser Game",
    "Neighbor Scope",
    # Missing keywords added individually:
    "Increase Value of Unchosen Resources",
    "Force Commitment",
    "Hidden Movement",
    "Layering",
    "Rondel",
    "Roles with Asymmetric Information",
    "Semi-Cooperative Game",
    "Targeted Clues"
]

mechanic_keyword_bundles = [
    # 1. Other Bidding Mechanisms
    ["Auction", "Constrained Bidding", "Predictive Bid", "Selection Order Bid", "Order Counters", "Bids As Wagers", "Closed Economy Auction", "Auction Compensation"],
    # 2. Alternate Worker Placement Variants
    ["Worker Placement, Different Worker Types", "Worker Placement with Dice Workers"],
    # 3. Additional Dice Mechanics
    ["Die Icon Resolution", "Re-rolling and Locking", "Different Dice Movement"],
    # 4. Additional Map/Movement Variants
    ["Multiple Maps", "Map Addition", "Map Reduction", "Map Deformation", "Relative Movement", "Impulse Movement", "Area-Impulse", "Pattern Movement", "Moving Multiple Units", "Pieces as Map", "Minimap Resolution", "Programmed Movement", "Movement Template", "Resource to Move"],
    # 5. Additional Card Mechanics
    ["Multi-Use Cards", "Move Through Deck", "Melding and Splaying"],
    # 6. Victory Points Mechanics
    ["Victory Points as a Resource", "Hidden Victory Points"],
    # 7. Matching Mechanics
    ["Matching", "Speed Matching", "Bingo"],
    # 8. Reactive/Chaining Mechanics
    ["Chaining", "Action Retrieval", "Player Judge", "Interrupts", "Bias"],
    # 9. Physical Manipulation Mechanics
    ["Stacking and Balancing", "Slide / Push", "Cube Tower", "Tug of War", "Physical Removal"],
    # 10. End Game Variants
    ["Sudden Death Ending", "Finale Ending", "Highest-Lowest Scoring", "Elapsed Real Time Ending"],
    # 11. Additional Economic Mechanics
    ["Delayed Purchase", "Loans", "Automatic Resource Growth", "Ownership", "Investment"],
    # 12. Additional Player Interaction Mechanics
    ["Negotiation", "Kill Steal", "Bribery", "Hot Potato", "Static Capture", "Prisoner's Dilemma"],
    # 13. Creative/Expressive Mechanics
    ["Drawing", "Questions and Answers", "Induction", "Crayon Rail System"],
    # 14. Action Token/Follow Mechanics
    ["Follow", "Passed Action Token"],
    # 15. Random Production & Mancala
    ["Random Production", "Mancala"],
    # 16. Ordering Mechanics
    ["Ordering", "Resource Queue"]
]



if examine == False:
    data = create_keyword_columns(data, feature, individual_mechanic_keywords, mechanic_keyword_bundles)
    # Drop the original 'family' column
    data.drop(columns=f'{feature}', inplace=True)

    # Save the updated data to a new CSV file
    data.to_csv(f'./code/Data Preparing/data_with_{feature}_columns.csv', index=False)
