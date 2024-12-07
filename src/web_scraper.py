from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager  # This handles ChromeDriver installation
from bs4 import BeautifulSoup
import re
import requests

class WebScraper:
    def __init__(self):
        """
        Initializes the ImdbScraper class with ChromeDriver managed by webdriver_manager
        """
        # Set up Chrome options for headless mode and custom User-Agent
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode (ideal for Colab)
        chrome_options.add_argument("--no-sandbox")  # Disable sandboxing for Colab
        chrome_options.add_argument("--disable-dev-shm-usage")  # Prevent issues with shared memory
        chrome_options.binary_location = '/usr/bin/google-chrome-stable'  # Path to Chrome binary in Colab

        # Custom User-Agent to simulate a real browser
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        # Use webdriver_manager to automatically handle ChromeDriver installation
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    def close(self):
        """
        Closes the driver
        """
        self.driver.quit()  # Properly close the browser instance

    def get_imdb_infos(self, movie_id=0):
        """
        Scrapes the IMBD webpage of a movie to extract useful information.

        Inputs:
            movie_id (int): The movie's Wikipedia ID.

        Outputs:
            dict: A dictionary containing the movie's information, such as revenue, budget, ratings, etc.
        """
        try:
            # Access Wikipedia API to retrieve the link of the page from the Wikipedia ID
            resp = requests.get(f"http://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={movie_id}&inprop=url&format=json")
            
            # Check if the 'query' and 'pages' keys exist in the API response
            pages = resp.json().get('query', {}).get('pages', {})
            if str(movie_id) not in pages:
                raise ValueError(f"No page found for movie_id {movie_id}")
            
            # Get the Wikipedia movie link
            wikipedia_movie_link = pages[str(movie_id)].get('fullurl', None)
            
            if not wikipedia_movie_link:
                raise ValueError(f"No 'fullurl' found for movie_id {movie_id}")
            
            # Access the movie's Wikipedia page
            html = requests.get(wikipedia_movie_link)
            soup = BeautifulSoup(html.text, 'html.parser')

            # Find the link to its Wikidata page
            tools = soup.find_all('div', {'id': 'vector-page-tools'})
            wikidata_link = tools[0].find('li', {'id': 't-wikibase'}).find('a')['href']
            wikidata_html = requests.get(wikidata_link)

            # Parse the Wikidata page to find its IMDb ID
            new_soup = BeautifulSoup(wikidata_html.text, 'html.parser')
            wiki_imdb = new_soup.find('div', {'id': 'P345'})
            imdb_id = wiki_imdb.find('div', {'class': 'wikibase-snakview-value wikibase-snakview-variation-valuesnak'}).text
            imdb_link = f"https://www.imdb.com/title/{imdb_id}/"

            # Access the IMDb page using Selenium to parse it & extract useful information
            self.driver.get(imdb_link)
            imdb_html = self.driver.page_source

            # Save the HTML to a file for debugging
            with open("imdb_movie_page.html", "w", encoding="utf-8") as file:
                file.write(imdb_html)

            soup = BeautifulSoup(imdb_html, 'html.parser')
            box_office = soup.find('div', {'data-testid': "title-boxoffice-section"})
            details_section = soup.find('div', {'data-testid': "title-details-section"})

            global_revenue, budget, opening_weekend = None, None, None
            currency_revenue, currency_budget, currency_opening_weekend = None, None, None
            rating_score, producer, release_year = None, None, None

            # Global revenue
            try:
                brut_li = box_office.find('li', {'data-testid': "title-boxoffice-cumulativeworldwidegross"})
                global_revenue_raw = brut_li.find('span', {'class': "ipc-metadata-list-item__list-content-item"}).text
                # Remove any extra text like "(estimated)"
                global_revenue_cleaned = re.sub(r'\(.*?\)', '', global_revenue_raw).strip()
                # Extract currency and numeric value
                match = re.match(r'([^\d,]+)([\d,]+)', global_revenue_cleaned)
                if match:
                    currency_revenue = match.group(1).strip()
                    cleaned_revenue = match.group(2).replace(',', '')
                    global_revenue = float(cleaned_revenue)
            except:
                global_revenue = None

            # Budget
            try:
                budget_li = box_office.find('li', {'data-testid': "title-boxoffice-budget"})
                budget_raw = budget_li.find('span', {'class': "ipc-metadata-list-item__list-content-item"}).text
                # Remove any extra text like "(estimated)"
                budget_cleaned = re.sub(r'\(.*?\)', '', budget_raw).strip()
                # Extract currency and numeric value
                match = re.match(r'([^\d,]+)([\d,]+)', budget_cleaned)
                if match:
                    currency_budget = match.group(1).strip()
                    cleaned_budget = match.group(2).replace(',', '')
                    budget = float(cleaned_budget)
            except:
                budget = None

            # Opening weekend
            try:
                opening_weekend_li = box_office.find('li', {'data-testid': "title-boxoffice-openingweekenddomestic"})
                opening_weekend_raw = opening_weekend_li.find('span', {'class': "ipc-metadata-list-item__list-content-item"}).text
                # Remove any extra text like "(estimated)"
                opening_weekend_cleaned = re.sub(r'\(.*?\)', '', opening_weekend_raw).strip()
                # Extract currency and numeric value
                match = re.match(r'([^\d,]+)([\d,]+)', opening_weekend_cleaned)
                if match:
                    currency_opening_weekend = match.group(1).strip()
                    cleaned_opening_weekend = match.group(2).replace(',', '')
                    opening_weekend = float(cleaned_opening_weekend)
            except:
                opening_weekend = None

            # Rating score
            try:
                rating_score_div = soup.find('div', {'data-testid': "hero-rating-bar__aggregate-rating__score"})
                rating_score_raw = rating_score_div.text
                rating_score = float(re.findall(r'\d{1}.\d{1}', rating_score_raw.replace(',', '.').replace('\u202f', ''))[0])
            except:
                rating_score = None

            # Producer
            try:
                producer = soup.find("a", {"class": "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"}).text
            except:
                producer = None

            # Release year
            try:
                date_li = details_section.find('li', {'data-testid': "title-details-releasedate"})
                release_info = date_li.find('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
                release_date = release_info.text.split(' (')[0]  # 'November 30, 2001'
                release_year = release_date.split()[-1]  # Extracts year
            except:
                release_year = None

        except Exception as e:
            print(f"Error during scraping: {e}")
            return None

        return {
            "wikipedia_movie_id": movie_id,
            "movie_box_office_revenue": global_revenue,
            "currency_revenue": currency_revenue,
            "budget": budget,
            "currency_budget": currency_budget,
            "opening_weekend": opening_weekend,
            "currency_opening_weekend": currency_opening_weekend,
            "rating_score": rating_score,
            "producer": producer,
            "release_year": release_year,
        }