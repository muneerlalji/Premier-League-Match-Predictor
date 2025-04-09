from time import sleep
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import random
import time

# Add headers to mimic a browser and avoid being detected as a bot
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.google.com/',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Function to make requests with exponential backoff
def make_request(url, max_retries=5):
    retries = 0
    while retries < max_retries:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            wait_time = (2 ** retries) + random.uniform(0, 1)
            print(f"Rate limited (429). Waiting for {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
        else:
            print(f"Error: Status code {response.status_code}")
            return response
    print(f"Failed to get {url} after {max_retries} retries")
    return None

url = 'https://fbref.com/en/comps/9/stats/Premier-League-Stats'
years = list(range(2023, 2026))
years.reverse()
all_matches = []

# Loop through the years and get the data for each club
for year in years:
    print(f'Getting data for {year}')

    # Get the standings table for the year
    standings_response = make_request(url)
    if not standings_response:
        print(f"Skipping year {year} due to request failure")
        continue
        
    print("response status code:", standings_response.status_code)
    soup = BeautifulSoup(standings_response.text, 'html.parser')
    standings_table = soup.select('table.stats_table')[0] # Select the first table with the class 'stats_table' which is the standings table

    # Get the club URLs from the standings table
    club_extensions = [a.get('href') for a in standings_table.find_all('a')]

    # Filter only squad URL
    club_extensions = [ext for ext in club_extensions if '/squads/' in ext]
    club_urls = [f'https://fbref.com{ext}' for ext in club_extensions] # List of URLs for each club

    # Get url for previous season for next iteration
    previous_season = soup.select('a.prev')[0].get('href')
    url = f'https://fbref.com{previous_season}'

    for club_url in club_urls:
        club_name = club_url.split('/')[-1].replace('-Stats', '').replace('-', ' ')
        print(f'Getting data for {club_name} in {year}')

        # Get the club's matches
        data_response = make_request(club_url)
        if not data_response:
            print(f"Skipping {club_name} due to request failure")
            continue
            
        # Add random delay to appear more human-like
        sleep_time = random.uniform(2, 5)
        print(f"Sleeping for {sleep_time:.2f} seconds...")
        sleep(sleep_time)
        
        try:
            matches = pd.read_html(StringIO(data_response.text), match='Scores & Fixtures')[0]
        except:
            print(f"Could not find Scores & Fixtures table for {club_name}")
            continue
        
        # Get the shooting data
        soup = BeautifulSoup(data_response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        shooting_links = [l for l in links if l and 'all_comps/shooting' in l.get('href')]
        shooting_link = shooting_links[0].get('href')
        shooting_data_response = make_request(f'https://fbref.com{shooting_link}')
        if not shooting_data_response:
            print(f"Skipping shooting data for {club_name}")
            continue
            
        try:
            shooting = pd.read_html(StringIO(shooting_data_response.text), match='Shooting')[0]
            shooting.columns = shooting.columns.droplevel(0)
        except:
            print(f"Could not find Shooting table for {club_name}")
            continue

        # Merge the matches and shooting data
        try:
            club_data = matches.merge(shooting[["Date", "Sh", "SoT"]], on="Date")
        except ValueError:
            print(f'No data for {club_name} in {year}')
            continue

        # Only use Premier League matches and add the year and club name for distinction
        club_data = club_data[club_data["Comp"] == "Premier League"]
        club_data["Season"] = year
        club_data["Club"] = club_name

        # Append the data to the all_matches list
        all_matches.append(club_data)

matches_data = pd.concat(all_matches)
matches_data.columns = [c.lower() for c in matches_data.columns]
matches_data.to_csv('dataset/premier_league_matches.csv', index=False)