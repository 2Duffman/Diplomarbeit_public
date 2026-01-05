import pandas as pd
import json

# Define file paths
INPUT_CSV = "boardgames_ranks.csv"
OUTPUT_JSON = "game_ids.json"

# Load the CSV file
def load_and_filter_ids(csv_file):
    try:
        df = pd.read_csv(csv_file)
        
        # Ensure required columns exist
        if "id" not in df.columns or "usersrated" not in df.columns:
            raise ValueError("CSV file must contain 'id' and 'usersrated' columns")
        
        # Filter games where 'usersrated' is 10 or more
        filtered_df = df[df["usersrated"] >= 10]
        
        # Extract IDs as a list
        game_ids = filtered_df["id"].dropna().astype(int).tolist()
        
        return game_ids
    except Exception as e:
        print(f"Error: {e}")
        return []

# Main function
def main():
    game_ids = load_and_filter_ids(INPUT_CSV)
    
    if game_ids:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(game_ids, f, indent=4)
        print(f"Saved {len(game_ids)} game IDs to {OUTPUT_JSON}")
    else:
        print("No valid game IDs found.")

if __name__ == "__main__":
    main()