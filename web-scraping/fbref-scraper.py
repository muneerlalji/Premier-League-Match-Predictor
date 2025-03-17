from time import sleep
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://fbref.com/en/comps/9/stats/Premier-League-Stats'

standings_html = requests.get(url)
soup = BeautifulSoup(standings_html.text, 'html.parser')
standings_table = soup.select('table.stats_table')[0] # Select the first table with the class 'stats_table' which is the standings table

club_extensions = [a.get('href') for a in standings_table.find_all('a')]
club_urls = [f'https://fbref.com{ext}' for ext in club_extensions] # List of URLs for each club

for url in club_urls:
    data = requests.get(url)
    matches = pd.read_html(data.text, match='Scores & Fixtures')[0]
    sleep(2) # Sleep for 2 seconds to avoid getting blocked by the server