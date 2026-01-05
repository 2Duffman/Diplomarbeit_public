import json

# Define file paths
GAME_IDS_JSON = "game_ids.json"
SCRAPED_IDS_JSON = "scraped_ids.json"
OUTPUT_JSON = "filtered_game_ids.json"

# Load game IDs from JSON
def load_ids(json_file):
    try:
        with open(json_file, "r") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error loading {json_file}")
        return set()

# Main function
def main():
    all_game_ids = load_ids(GAME_IDS_JSON)
    scraped_ids = load_ids(SCRAPED_IDS_JSON)
    
    # Filter out already scraped games
    new_game_ids = list(all_game_ids - scraped_ids)
    
    if new_game_ids:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(new_game_ids, f, indent=4)
        print(f"Saved {len(new_game_ids)} new game IDs to {OUTPUT_JSON}")
    else:
        print("No new game IDs to scrape.")

if __name__ == "__main__":
    main()
