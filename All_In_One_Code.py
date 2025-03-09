'''
MACS 30112 Final Project
Global Spatial Distribution Of Open-Source Artificial Intelligence
Group Chick-fil-A
Anqi Wei, Baihui Wang & Charlotte Li
'''

### DATA SCRAPING ###
import requests
import csv
import time
import random

# GitHub API URLs
GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_CONTRIBUTORS_URL = "https://api.github.com/repos/{owner}/{repo}/contributors"

# Multiple API tokens to avoid rate limits
GITHUB_TOKENS = [
    "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",
    "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",
    "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", 
    "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"
]

token_index = 0  # Start with the first token

token_index = 0  # Start with the first token

def get_headers():
    """Rotate GitHub tokens to avoid hitting rate limits."""
    global token_index
    headers = {
        "Authorization": f"token {GITHUB_TOKENS[token_index]}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Mozilla/5.0"
    }
    token_index = (token_index + 1) % len(GITHUB_TOKENS)  # Rotate tokens
    return headers

def request_with_retries(url, params=None):
    """GitHub API request with token rotation and automatic retries."""
    retries = 5
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=get_headers(), params=params, timeout=10)
            
            if response.status_code == 403:  # API rate limit exceeded
                print("Rate limit hit. Waiting...")
                time.sleep(60)  # Wait before retrying
                continue  

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Request failed ({e}), retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    return None  # Return None if all retries fail

def get_ai_repos(topic, max_repos=10000):
    """Fetch AI-related repositories sorted by stars."""
    repos = set()  # Use a set to store unique repositories
    per_page = 500
    total_pages = min(10, (max_repos // per_page) + 1)

    for page in range(1, total_pages + 1):
        print(f"Fetching page {page}/{total_pages}...")

        params = {
            "q": f"topic:{topic} language:python",  # Ensure Python AI repos
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page
        }

        result = request_with_retries(GITHUB_API_URL, params=params)
        if not result or "items" not in result:
            print("No more results or API limit reached.")
            break  

        for repo in result["items"]:
            repos.add((
                repo["full_name"], repo["stargazers_count"], repo["created_at"], repo["forks_count"]
            ))

        if len(repos) >= max_repos:
            break  

        time.sleep(random.uniform(1, 3))  # Avoid hitting API limits

    return list(repos)

def get_contributors(owner, repo, max_users=1000):
    """Fetch contributors and their locations for a given repository."""
    url = GITHUB_CONTRIBUTORS_URL.format(owner=owner, repo=repo)
    users = []
    page = 1
    per_page = 500

    while len(users) < max_users:
        print(f"Fetching contributors for {repo}, page {page}...")
        params = {"per_page": per_page, "page": page}
        result = request_with_retries(url, params=params)

        if not result:
            break  

        for contributor in result:
            username = contributor.get("login", "Unknown")
            profile_url = contributor.get("html_url", "Unknown")
            user_url = contributor.get("url", "")

            location = "Unknown"
            if user_url:
                user_data = request_with_retries(user_url)
                if user_data:
                    location = user_data.get("location", "Unknown")

            users.append((username, location, profile_url))

        if len(result) < per_page:
            break  # No more pages

        page += 1
        time.sleep(random.uniform(1, 3))  

    return users[:max_users]

def save_data(repos, filename="github_ai_repos.csv"):
    """Save the scraped data to a CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Repository", "Stars", "Created At", "Forks"])
        writer.writerows(repos)

def main():
    topic = "ai"
    max_repos = 10000  # Increase this value to scrape more repositories

    # Fetch AI-related repositories
    repos = get_ai_repos(topic, max_repos)
    save_data(repos, "github_ai_repos.csv")

    unique_repos_count = len(set(repo[0] for repo in repos))
    print(f"\n {unique_repos_count} unique AI repositories saved.")

    # Fetch contributor locations for each repo
    contributor_data = []
    for repo_full_name, stars, created_at, forks in repos:
        owner, repo_name = repo_full_name.split("/")
        contributors = get_contributors(owner, repo_name)

        for username, location, profile_url in contributors:
            contributor_data.append([repo_full_name, stars, username, location, profile_url])

    # Save contributor data
    with open("github_contributor_locations.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Repository", "Stars", "Contributor", "Location", "Profile URL"])
        writer.writerows(contributor_data)

    unique_locations = len(set([c[3] for c in contributor_data if c[3] != "Unknown"]))
    print(f"\n {unique_locations} unique contributor locations saved.")

if __name__ == "__main__":
    main()


### DATA CLEANING ###
import pandas as pd
import numpy as np
import time
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import googlemaps 

file_path = "contributors_locations.csv"
print("Loading dataset...")
df = pd.read_csv(file_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print("Dropping rows with missing locations...")
df_cleaned = df.dropna(subset=["Location"]).copy()
print(f"Remaining rows after drop: {df_cleaned.shape[0]}")

df_cleaned["City"] = df_cleaned["Location"].apply(lambda x: x.split(",")[0].strip() if "," in x else np.nan)
df_cleaned["Country"] = df_cleaned["Location"].apply(lambda x: x.split(",")[-1].strip() if "," in x else x)

country_replacements = {
    "UK": "United Kingdom",
    "NL": "Netherlands",
    "USA": "United States",
    "US": "United States",
    "DE": "Germany",
    "FR": "France",
    "CA": "Canada",
    "CN": "China",
    "JP": "Japan",
    "IN": "India",
}

df_cleaned["Country"] = df_cleaned["Country"].replace(country_replacements)
print("Standardized country names.")

geolocator = Nominatim(user_agent="wangbaihui_github_scraper", timeout=10)
gmaps = googlemaps.Client(key="AIzaSyBowgbuKSOBiDBMVr-yEaKDbhrU_z8U0b0")

def is_ip_address(location):
    """Check if the location is an IP address (e.g., 127.0.0.1)."""
    return bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", str(location)))

def geocode_location(location):
    """Geocode location and return (city, country), handling failures properly."""
    try:
        if not location or is_ip_address(location):
            print(f" Skipping invalid location: {location}")
            return "Unknown", "Unknown"
        
        print(f" Geocoding location: {location}")
        
        geo_data = geolocator.geocode(location, exactly_one=True)
        if geo_data:
            address_parts = geo_data.address.split(",")
            city = address_parts[0].strip() if len(address_parts) > 0 else "Unknown"
            country = address_parts[-1].strip() if len(address_parts) > 1 else "Unknown"
            print(f"Geocoded using OpenStreetMap: {city}, {country}")
            return city, country

    except GeocoderTimedOut:
        print(f"Geocoder timed out for {location}. Retrying after delay...")
        time.sleep(5) 
    except Exception as e:
        print(f"OpenStreetMap geocoding failed for {location}: {e}")

    try:
        geo_data = gmaps.geocode(location)
        if geo_data:
            city = geo_data[0]["address_components"][0]["long_name"]
            country = geo_data[-1]["address_components"][-1]["long_name"]
            print(f"Geocoded using Google Maps: {city}, {country}")
            return city, country
    except Exception:
        print(f"Google Maps geocoding failed for {location}")

    return "Unknown", "Unknown"

print("Filling missing city values...")
df_cleaned["City"] = df_cleaned["City"].fillna("Unknown")

city_replacements = df_cleaned.groupby("Country")["City"].agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
df_cleaned.loc[df_cleaned["City"] == "Unknown", "City"] = df_cleaned["Country"].map(city_replacements)
print("Replaced missing cities with most common per country.")

print("Geocoding unknown locations...")
for index, row in df_cleaned.iterrows():
    if row["City"] == "Unknown" or row["Country"] == "Unknown":
        try:
            city, country = geocode_location(row["Location"])
            if pd.notna(city):
                df_cleaned.at[index, "City"] = city
            if pd.notna(country):
                df_cleaned.at[index, "Country"] = country
            time.sleep(2)  
        except Exception as e:
            print(f"Geocoding failed for {row['Location']}: {e}")
            time.sleep(5)  

print("Geocoding completed!")

print("Aggregating data by city and country...")
city_aggregation = df_cleaned.groupby("City").agg(
    total_contributors=("Contributor", "count"),
    total_stars=("Stars", "sum"),
    avg_stars_per_contributor=("Stars", "mean")
).reset_index()

country_aggregation = df_cleaned.groupby("Country").agg(
    total_contributors=("Contributor", "count"),
    total_stars=("Stars", "sum"),
    avg_stars_per_contributor=("Stars", "mean")
).reset_index()

df_cleaned.to_csv("refined_cleaned_contributors_locations.csv", index=False)
city_aggregation.to_csv("refined_city_aggregated_data.csv", index=False)
country_aggregation.to_csv("refined_country_aggregated_data.csv", index=False)

print("\n **Final Statistics**")
print(f"Total cleaned contributors: {df_cleaned.shape[0]}")
print(f"Unique countries: {df_cleaned['Country'].nunique()}")
print(f"Unique cities: {df_cleaned['City'].nunique()}")
print(f"Data saved successfully!\n")

# 

# install package for translation:
# %pip install deep_translator
# %pip install opencc-python-reimplemented  
# %pip install langdetect

# install package for data preprocessing:
# %pip install pycountry

import pandas as pd
import numpy as np


# load the city-level data

city_file = pd.read_csv("refined_city_aggregated_data.csv")

print("City Data Info:", city_file.info())
print("---"*30)
print("City Data Head:", city_file.head())
print("---"*30)
print("City Data Describe:", city_file.describe())

from opencc import OpenCC
from deep_translator import GoogleTranslator

cc = OpenCC('t2s')

# Translate the city/country names to English using Google Translator
def translate_text(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return text  
        
        # Translate Traditional Chinese to Simplified Chinese
        text = cc.convert(text)

        return GoogleTranslator(source='auto', target='en').translate(text)

    except Exception as e:
        print(f"Translation failed for {text}: {e}")
        return text  

# Apply the translation function to the city names
city_file['City_English'] = city_file['City'].apply(translate_text)

print(city_file[['City', 'City_English']].head(50))

city_file = city_file.to_csv("refined_city_2.csv", index=False)

import pandas as pd
import re


file_path = "refined_city_2.csv"
city_file = pd.read_csv(file_path)

# Clearning data again:
## only special characters, only digits, meaningless symbols;
## remove leading/trailing spaces, commas, hyphens;
## combine multiple spaces into one;
## title case the city name.

pattern_invalid = r"^[\s\W\d]+$" 
pattern_special_chars = r"[^\w\s\-,]"  
pattern_strip_commas_hyphens = r"^[\s,-]+|[\s,-]+$"  


def clean_city(city):
    if pd.isna(city) or re.fullmatch(pattern_invalid, str(city)):  
        return None  
    
    city = city.strip()  

    if (city.startswith("'") and city.endswith("'")) or (city.startswith('"') and city.endswith('"')):
        city = city[1:-1]
 
    city = re.sub(pattern_strip_commas_hyphens, '', city)

    city = re.sub(pattern_special_chars, '', city)

    city = re.sub(r'\s+', ' ', city).strip()

    city = city.title()  
    
    return city

city_file['City_English_Cleaned'] = city_file['City_English'].apply(clean_city)

# Attention: 
## There might be some errors in the translation process; manually checks are needed.
## The cleaning process is not perfect; we improved it in the final step.

city_file.to_csv("refined_city_2.csv", index=False)

print("Translation and Cleaning Results are saved as 'City_English_Cleaned' column:")
print(city_file[['City_English', 'City_English_Cleaned']].head(20))
print("---"*30)
print("City Data Head:", city_file.tail(20))

# load the country-level data

country_file = pd.read_csv("refined_country_aggregated_data.csv")

print("Country Data Info:", country_file.info())
print("---"*30)
print("Country Data Head:", country_file.head())
print("---"*30)
print("Country Data Describe:", country_file.describe())

# Apply the translation function to the country names
country_file['Country_English'] = country_file['Country'].apply(translate_text)

print(country_file[['Country', 'Country_English']].head(50))

country_file = country_file.to_csv("refined_country_2.csv", index=False)


# Standardize the country names using pycountry

import pycountry

file_path = "refined_country_2.csv"
country_file = pd.read_csv(file_path)

country_names = {c.name: c.name for c in pycountry.countries}
country_names.update({c.alpha_2: c.name for c in pycountry.countries})
country_names.update({c.alpha_3: c.name for c in pycountry.countries})

def standardize_country(country):
    if pd.isna(country) or not isinstance(country, str) or country.strip() == "":
        return None  

    country = country.strip()

    match = pycountry.countries.get(name=country) or \
            pycountry.countries.get(alpha_2=country) or \
            pycountry.countries.get(alpha_3=country)

    if match:
        return match.name

    for official_name in country_names.keys():
        if official_name.lower() in country.lower():  
            return country_names[official_name]  
        
    return None  

country_file['Country_Standardized'] = country_file['Country_English'].apply(standardize_country)

country_file.to_csv(file_path, index=False)

# See how many countries are standardized
non_null_countries = country_file.dropna(subset=['Country_Standardized'])

print("Country list after standardization:")
print(non_null_countries[['Country_English', 'Country_Standardized']].head(20))
print(f"\nTotal non-null standardized countries: {len(non_null_countries)}")

## Attention:
## The standardization process is not perfect; we improve it in the final step.

# Merge the same country data, and recalculate the indicators
merged_country = country_file .groupby("Country_Standardized", as_index=False).agg({
    "total_contributors": "sum",
    "total_stars": "sum"
})

# `avg_stars_per_contributor`**
merged_country["avg_stars_per_contributor"] = merged_country["total_stars"] / merged_country["total_contributors"]

merged_country.to_csv("country_data_3.csv", index=False)

# Check the results
print("The merged country data:")
print(merged_country.head(20))
print("---"*30)
print("The merged country data describe:")
print(merged_country.describe())
print("---"*30)
print("The top 20 countries with the highest average stars per contributor:")
print(merged_country.sort_values("avg_stars_per_contributor", ascending=False).head(20))

# Manually Checking
# Display unique values in the 'Country_Standardized' column to identify non-country entries
country_file = "country_data_3.csv"

country_3 = pd.read_csv(country_file, on_bad_lines='skip')
unique_values = country_3["Country_Standardized"].unique()

# Display a sample of unique values
print(unique_values)

# Define a mapping of non-sovereign entities to their corresponding sovereign country names
territory_to_country = {
    "American Samoa": "United States",
    "Anguilla": "United Kingdom",
    "Aruba": "Netherlands",
    "British Indian Ocean Territory": "United Kingdom",
    "Cayman Islands": "United Kingdom",
    "Cook Islands": "New Zealand",
    "Faroe Islands": "Denmark",
    "Gibraltar": "United Kingdom",
    "Greenland": "Denmark",
    "Guam": "United States",
    "Holy See (Vatican City State)": "Vatican City",
    "Isle of Man": "United Kingdom",
    "Moldova, Republic of": "Moldova",
    "Montserrat": "United Kingdom",
    "New Caledonia": "France",
    "Northern Mariana Islands": "United States",
    "Puerto Rico": "United States",
    "Réunion": "France",
    "Saint Barthélemy": "France",
    "Saint Helena, Ascension and Tristan da Cunha": "United Kingdom",
    "Korea, Republic of": "South Korea",
    "Turks and Caicos Islands": "United Kingdom",
    "Åland Islands": "Finland",
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "Lao People's Democratic Republic": "Laos",
    "Moldova, Republic of": "Moldova"
}

# Replace non-sovereign entities with their respective country names
country_3["Country_Standardized"] = country_3["Country_Standardized"].replace(territory_to_country)

# Use the existing 'country' DataFrame
country_list = country_3.groupby("Country_Standardized", as_index=False).agg({
    "total_contributors": "sum",
    "total_stars": "sum",
    "avg_stars_per_contributor": "mean"  
})

# Recalculate avg_stars_per_contributor
country_list ["avg_stars_per_contributor"] = country_list ["total_stars"] / country_list ["total_contributors"]

print(country_list.head())

# Save the cleaned country data
country_list.to_csv("country_data_3.csv", index=False)

unique_values = country_3["Country_Standardized"].unique()

# Display a sample of unique values
print(unique_values)

# geocoded data cleaning
import pandas as pd
import html
from google.cloud import translate_v2 as translate
from langdetect import detect, DetectorFactory

file_path = "cleaned_geocoded_data.csv"
print("Loading dataset...")
df = pd.read_csv(file_path)
print(f"Dataset loaded. Shape: {df.shape}")

for col in ["Location", "Country", "City"]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in the dataset.")

translator = translate.Client()
DetectorFactory.seed = 0  

state_abbreviations = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}
global_abbreviations = {
    "USA", "UK", "UAE", "EU", "UN", "NYC", "SF", "LA", "DC", "TX", "FL", "CA", "IL", "WA", "OH", "MA", "PA"
}
known_abbreviations = state_abbreviations.union(global_abbreviations) 

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    if text in known_abbreviations:  
        return text
    lang = detect_language(text)
    if lang == "en":  
        return text
    try:
        translated_text = translator.translate(text, target_language="en")["translatedText"]
        translated_text = html.unescape(translated_text)  
        print(f"Translated '{text}' ({lang}) -> '{translated_text}'")
        return translated_text  
    except Exception as e:
        print(f"Translation failed for: {text} - Error: {e}")
        return text  

print("\n Translating 'Location', 'Country', and 'City'...")
df["Location"] = df["Location"].apply(lambda x: translate_text(x))
df["Country"] = df["Country"].apply(lambda x: translate_text(x))
df["City"] = df["City"].apply(lambda x: translate_text(x))
print("\n Translation complete!")

translated_file = "translated_geocoded_data.csv"
df.to_csv(translated_file, index=False, encoding="utf-8")
print(f"\n Translated dataset saved as '{translated_file}'.")


### DATA VISUALIZATION ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import googlemaps
import time
import json
import os
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
import networkx as nx
from pyvis.network import Network

file_path = "refined_cleaned_contributors_locations.csv"  
df = pd.read_csv(file_path)

repo_contributors = df.groupby("Repository")["Contributor"].nunique().reset_index()
repo_contributors = repo_contributors.merge(df[["Repository", "Stars"]].drop_duplicates(), on="Repository")

repo_contributors = repo_contributors.sort_values(by="Stars", ascending=False).head(1000)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=repo_contributors, x="Stars", y="Contributor", alpha=0.7, edgecolor="black")
plt.xlabel("Number of Stars")
plt.ylabel("Number of Unique Contributors")
plt.title("Repository Popularity vs. Contributor Count")
plt.grid(True)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "refined_country_aggregated_data.csv"
df_country = pd.read_csv(file_path)

df_country = df_country.dropna(subset=["Country"])

df_country = df_country[df_country["Country"] != "Unknown"]

df_country["Country"] = df_country["Country"].apply(lambda x: x.encode("utf-8", "ignore").decode("utf-8"))

df_country = df_country[df_country["Country"].apply(lambda x: x.isascii())]

df_country["total_contributors"] = pd.to_numeric(df_country["total_contributors"], errors="coerce")

df_country = df_country.dropna(subset=["total_contributors"])

top_countries = df_country.sort_values("total_contributors", ascending=False).head(30)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries["total_contributors"], y=top_countries["Country"], color="blue")
plt.xlabel("Total New Contributors")
plt.ylabel("Country")
plt.title("Top 30 Countries with the Highest New GitHub Contributors")
plt.gca().invert_yaxis()
plt.show()

API_KEY = "AIzaSyB6P3uVLPBrebrlE1WG0PXpAXi3Ex1yFRU"  
gmaps = googlemaps.Client(key=API_KEY)

file_path = "refined_cleaned_contributors_locations.csv"
df = pd.read_csv(file_path)

cache_file = "geocode_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        geocode_cache = json.load(f)
else:
    geocode_cache = {}

def get_lat_lon_batch(locations, batch_num):
    """Geocode a batch of locations using Google Maps API with caching and error handling."""
    batch_results = []
    locations_to_geocode = [loc for loc in locations if loc and loc not in geocode_cache]

    print(f"Processing Batch {batch_num}: {len(locations_to_geocode)} new locations...")

    if not locations_to_geocode:
        print(f"Batch {batch_num}: All locations already cached. Skipping.")
        return [geocode_cache.get(loc, (None, None)) for loc in locations]

    try:
        for loc in locations_to_geocode:
            response = gmaps.geocode(loc)
            if response:
                lat = response[0]['geometry']['location']['lat']
                lon = response[0]['geometry']['location']['lng']
                geocode_cache[loc] = (lat, lon)
                print(f"Geocoded: {loc} → ({lat}, {lon})")
            else:
                geocode_cache[loc] = (None, None)
                print(f"No geocode result for: {loc}")

            time.sleep(0.1)  

    
        with open(cache_file, "w") as f:
            json.dump(geocode_cache, f)

    except Exception as e:
        print(f"Error in Batch {batch_num}: {e}")
        return [(None, None)] * len(locations)

    print(f"Batch {batch_num} completed!")
    return [geocode_cache.get(loc, (None, None)) for loc in locations]

batch_size = 10
total_batches = len(df) // batch_size + 1

for batch_num, i in enumerate(range(0, len(df), batch_size), start=1):
    df.loc[i:i+batch_size-1, ["Latitude", "Longitude"]] = get_lat_lon_batch(df["Location"].iloc[i:i + batch_size], batch_num)

df.dropna(subset=["Latitude", "Longitude"], inplace=True)

df.to_csv("geocoded_data.csv", index=False)
print("Geocoded data saved to geocoded_data.csv")

df = pd.read_csv("geocoded_data.csv")

df_small = df.head(1000) 

m = folium.Map(location=[20, 0], zoom_start=2)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df_small.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"Location: {row.get('Location', 'Unknown')}<br>Contributor: {row.get('Contributor', 'N/A')}<br>Stars: {row.get('Stars', 'N/A')}",
        icon=folium.Icon(color="blue", icon="cloud"),
    ).add_to(m)

map_file_path = "top_1000_contributors_map.html"
m.save(map_file_path)

print(f"Smaller map saved with {len(df_small)} markers: {map_file_path}")

# Create heatmap
m = folium.Map(location=[20, 0], zoom_start=2)
HeatMap(df[["Latitude", "Longitude"]].dropna()).add_to(m)

# Save map
m.save("github_contributors_heatmap.html")
print("Heatmap saved as github_contributors_heatmap.html")

total_contributors = df.groupby("Repository")["Contributor"].nunique()
total_stars = df.groupby("Repository")["Stars"].sum()

plt.figure(figsize=(8, 6))
sns.regplot(x=total_contributors, y=total_stars, scatter_kws={"alpha": 0.5})
plt.xlabel("Total Contributors")
plt.ylabel("Total Stars")
plt.title("Correlation Between Contributors and Repository Stars")
plt.show()

# Load dataset
file_path = "refined_country_aggregated_data.csv"
df_country = pd.read_csv(file_path)

# Sort by an approximate "time" column (e.g., contributor index)
df_country = df_country.sort_values("total_contributors", ascending=True).reset_index()

# Simulate cumulative growth
df_country["cumulative_contributors"] = df_country["total_contributors"].cumsum()

# Plot cumulative contributor growth
plt.figure(figsize=(10, 5))
plt.plot(df_country.index, df_country["cumulative_contributors"], marker="o", linestyle="-", color="b")
plt.xlabel("Index (Approximate Time)")
plt.ylabel("Cumulative Contributors")
plt.title("Simulated Growth of GitHub Contributors Over Time (No Timestamps)")
plt.show()

file_path = "refined_cleaned_contributors_locations.csv"  
df = pd.read_csv(file_path)

df = df.dropna(subset=["Contributor", "Repository"])

top_contributors = df["Contributor"].value_counts().head(50).index
top_repos = df["Repository"].value_counts().head(10).index

df_filtered = df[(df["Contributor"].isin(top_contributors)) | (df["Repository"].isin(top_repos))]

G = nx.Graph()
for _, row in df_filtered.iterrows():
    G.add_edge(row["Contributor"], row["Repository"])

pos = nx.kamada_kawai_layout(G) 

node_sizes = [max(G.degree[node] * 10, 50) for node in G.nodes()]
node_colors = ["blue" if node in top_contributors else "red" for node in G.nodes()]

plt.figure(figsize=(14, 10))
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, width=0.5) 
nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors, alpha=0.9)

important_nodes = {node for node, deg in G.degree() if deg > 5}
node_labels = {node: node if node in important_nodes else "" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")

plt.title("Network of Top 100 Contributors & Repositories")
plt.show()

file_path = "refined_cleaned_contributors_locations.csv" 
df = pd.read_csv(file_path)

df = df.dropna(subset=["Contributor", "Repository"])

top_contributors = df["Contributor"].value_counts().head(100).index
top_repos = df["Repository"].value_counts().head(20).index

df_filtered = df[(df["Contributor"].isin(top_contributors)) | (df["Repository"].isin(top_repos))]

G = nx.Graph()
for _, row in df_filtered.iterrows():
    G.add_edge(row["Contributor"], row["Repository"])

net = Network(height="800px", width="100%", notebook=True, bgcolor="#ffffff", font_color="black")

for node in G.nodes():
    net.add_node(node, label=node, color="blue" if node in top_contributors else "red", size=G.degree[node] * 2)

for edge in G.edges():
    net.add_edge(edge[0], edge[1], color="gray")

net.force_atlas_2based()

output_file = "interactive_network.html"
net.show(output_file)
print(f"Network saved to {output_file}, open it in a browser to explore interactively!")


### DATA ANALYSIS ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from iso3166 import countries 

geocode = pd.read_csv('geocoded_data.csv')
# Display the first few rows of the DataFrame
print(geocode.head())
print("---"*30)
print(geocode.describe())

# check missing/'unknown' values
missing_values = geocode.isnull().sum()
unknown_counts = (geocode == "Unknown").sum()
unknown_rows = geocode[geocode.eq("Unknown").any(axis=1)]

print("Missing values in each column:")
print(missing_values)
print("---" * 30)

print("Count of 'unknown' values in each column:")
print(unknown_counts)
print("---" * 30)

print("Rows with 'unknown' values:")
print(unknown_rows)

country_info = pd.read_csv("country_data.csv")

# Display the first few rows of the DataFrame
print(country_info.head())
print("---"*30)
print(country_info.describe())

# check missing/'unknown' values
missing_values = country_info.isnull().sum()
unknown_counts = (country_info == "Unknown").sum()
unknown_rows = country_info[country_info.eq("Unknown").any(axis=1)]

print("Missing values in each column:")
print(missing_values)
print("---" * 30)

print("Count of 'unknown' values in each column:")
print(unknown_counts)
print("---" * 30)

print("Rows with 'unknown' values:")
print(unknown_rows)

print(country_info['total_contributors'].value_counts())

# makr the name for the countries with top n total_contributors/total_stars
top_n = 10

# plot the distribution of total_contributors
top_countries_ctb = country_info.groupby("Country_Standardized")["total_contributors"].max().nlargest(top_n).index

plt.figure(figsize=(8, 5))
sns.stripplot(x='Country_Standardized', y='total_contributors', data=country_info, jitter=True, size=8, alpha=0.7)
plt.xticks([])
plt.xlabel('Countries')  
plt.ylabel('Total Contributors')
plt.title('Distribution of Total Contributors by Country')

# add labels for countries with top N total_contributors
for country in top_countries_ctb:
    highest_value = country_info[country_info["Country_Standardized"] == country]["total_contributors"].max()
    plt.text(country, highest_value, country, ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# plot the distribution of total_stars
top_countries_str = country_info.groupby("Country_Standardized")["total_stars"].max().nlargest(top_n).index

plt.figure(figsize=(8, 5))
sns.stripplot(x='Country_Standardized', y='total_stars', data=country_info, jitter=True, size=8, alpha=0.7)
plt.xticks([])
plt.xlabel('Countries') 
plt.ylabel('Total Stars')
plt.title('Distribution of Total Stars by Country')

# add labels for countries with top N total_stars
for country in top_countries_str:
    highest_value = country_info[country_info["Country_Standardized"] == country]["total_stars"].max()
    plt.text(country, highest_value, country, ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

print("Correlation Matrix :")
corr_matrix = country_info.drop(columns=["Country_Standardized"]).corr()

# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap")
plt.show()

mean_ctb = country_info['total_contributors'].mean()
mean_str = country_info['total_stars'].mean()
numerator = 0
ctb_summation = 0
str_summation = 0
for index, row in country_info.iterrows():
    contributors = row['total_contributors']
    stars = row['total_stars']
    numerator += (contributors - mean_ctb) * (stars - mean_str)
    ctb_summation += (contributors - mean_ctb) ** 2
    str_summation += (stars - mean_str) ** 2
denominator = ctb_summation * str_summation
# Calculate the correlation coefficient
correlation = numerator / np.sqrt(denominator)
print("Correlation between total_contributors and total_stars:", correlation)

# Load datasets
country_data = pd.read_csv('country_data.csv')  # Contains country-specific data including GitHub activity
gdp_data = pd.read_csv('GDP.csv')  # GDP data for various countries
gdp_per_capita_data = pd.read_csv('GDP_per_capita.csv')  # GDP per capita data for various countries

# Extract GDP data for the year 2023 and rename columns for consistency
gdp_data_2023 = gdp_data[['Country Name', '2023']]
gdp_data_2023 = gdp_data_2023.rename(columns={'Country Name': 'Country_Standardized', '2023': 'GDP'})

# Extract GDP per capita data for the year 2023 and rename columns for consistency
gdp_per_capita_data_2023 = gdp_per_capita_data[['Country Name', '2023']]
gdp_per_capita_data_2023 = gdp_per_capita_data_2023.rename(columns={'Country Name': 'Country_Standardized', '2023': 'GDP_per_capita'})

# Merge the datasets on the standardized country name to ensure data consistency
final_data = pd.merge(country_data, gdp_data_2023, on='Country_Standardized', how='inner')
final_data = pd.merge(final_data, gdp_per_capita_data_2023, on='Country_Standardized', how='inner')

# Check for missing values in the merged dataset
print("Missing values in merged dataset:")
print(final_data.isnull().sum())

# Remove any rows with missing values
final_data = final_data.dropna()

# Convert relevant columns to numeric format to ensure compatibility with calculations
final_data['total_contributors'] = pd.to_numeric(final_data['total_contributors'], errors='coerce')
final_data['total_stars'] = pd.to_numeric(final_data['total_stars'], errors='coerce')
final_data['avg_stars_per_contributor'] = pd.to_numeric(final_data['avg_stars_per_contributor'], errors='coerce')
final_data['GDP'] = pd.to_numeric(final_data['GDP'], errors='coerce')
final_data['GDP_per_capita'] = pd.to_numeric(final_data['GDP_per_capita'], errors='coerce')

# Compute correlation between GitHub activity metrics and GDP
correlation_contributors_gdp = final_data['total_contributors'].corr(final_data['GDP'])
correlation_stars_gdp = final_data['total_stars'].corr(final_data['GDP'])
correlation_avg_stars_gdp = final_data['avg_stars_per_contributor'].corr(final_data['GDP'])

# Compute correlation between GitHub activity metrics and GDP per capita
correlation_contributors_gdp_per_capita = final_data['total_contributors'].corr(final_data['GDP_per_capita'])
correlation_stars_gdp_per_capita = final_data['total_stars'].corr(final_data['GDP_per_capita'])
correlation_avg_stars_gdp_per_capita = final_data['avg_stars_per_contributor'].corr(final_data['GDP_per_capita'])

# Print correlation results
print(f"Correlation between total contributors and GDP in 2023: {correlation_contributors_gdp}")
print(f"Correlation between total stars and GDP in 2023: {correlation_stars_gdp}")
print(f"Correlation between average stars per contributor and GDP in 2023: {correlation_avg_stars_gdp}")

print(f"Correlation between total contributors and GDP per capita in 2023: {correlation_contributors_gdp_per_capita}")
print(f"Correlation between total stars and GDP per capita in 2023: {correlation_stars_gdp_per_capita}")
print(f"Correlation between average stars per contributor and GDP per capita in 2023: {correlation_avg_stars_gdp_per_capita}")

# Create scatter plots to visualize correlations between GitHub activity metrics and GDP
plt.figure(figsize=(15, 5))

# Plot total contributors vs GDP
plt.subplot(1, 3, 1)
sns.scatterplot(x='total_contributors', y='GDP', data=final_data)
plt.title(f'Total Contributors vs GDP\nCorrelation: {correlation_contributors_gdp:.2f}')
plt.xlabel('Total Contributors')
plt.ylabel('GDP')

# Annotate top 5 countries with the highest GDP
top_5_contributors_gdp = final_data.nlargest(5, 'GDP')
for i in range(top_5_contributors_gdp.shape[0]):
    plt.text(top_5_contributors_gdp['total_contributors'].iloc[i], top_5_contributors_gdp['GDP'].iloc[i], top_5_contributors_gdp['Country_Standardized'].iloc[i])

# Plot total stars vs GDP
plt.subplot(1, 3, 2)
sns.scatterplot(x='total_stars', y='GDP', data=final_data)
plt.title(f'Total Stars vs GDP\nCorrelation: {correlation_stars_gdp:.2f}')
plt.xlabel('Total Stars')
plt.ylabel('GDP')

# Annotate top 5 countries with the highest GDP
top_5_stars_gdp = final_data.nlargest(5, 'GDP')
for i in range(top_5_stars_gdp.shape[0]):
    plt.text(top_5_stars_gdp['total_stars'].iloc[i], top_5_stars_gdp['GDP'].iloc[i], top_5_stars_gdp['Country_Standardized'].iloc[i])

# Plot average stars per contributor vs GDP
plt.subplot(1, 3, 3)
sns.scatterplot(x='avg_stars_per_contributor', y='GDP', data=final_data)
plt.title(f'Avg Stars per Contributor vs GDP\nCorrelation: {correlation_avg_stars_gdp:.2f}')
plt.xlabel('Avg Stars per Contributor')
plt.ylabel('GDP')

# Annotate top 5 countries with the highest GDP
top_5_avg_stars_gdp = final_data.nlargest(5, 'GDP')
for i in range(top_5_avg_stars_gdp.shape[0]):
    plt.text(top_5_avg_stars_gdp['avg_stars_per_contributor'].iloc[i], top_5_avg_stars_gdp['GDP'].iloc[i], top_5_avg_stars_gdp['Country_Standardized'].iloc[i])

plt.tight_layout()
plt.show()

# Create scatter plots to visualize correlations between GitHub activity metrics and GDP per capita
plt.figure(figsize=(15, 5))

# Plot total contributors vs GDP per capita
plt.subplot(1, 3, 1)
sns.scatterplot(x='total_contributors', y='GDP_per_capita', data=final_data)
plt.title(f'Total Contributors vs GDP per capita\nCorrelation: {correlation_contributors_gdp_per_capita:.2f}')
plt.xlabel('Total Contributors')
plt.ylabel('GDP per capita')

# Annotate top 5 countries with the highest GDP per capita
top_5_contributors_gdp_per_capita = final_data.nlargest(5, 'GDP_per_capita')
for i in range(top_5_contributors_gdp_per_capita.shape[0]):
    plt.text(top_5_contributors_gdp_per_capita['total_contributors'].iloc[i], top_5_contributors_gdp_per_capita['GDP_per_capita'].iloc[i], top_5_contributors_gdp_per_capita['Country_Standardized'].iloc[i])

# Plot total stars vs GDP per capita
plt.subplot(1, 3, 2)
sns.scatterplot(x='total_stars', y='GDP_per_capita', data=final_data)
plt.title(f'Total Stars vs GDP per capita\nCorrelation: {correlation_stars_gdp_per_capita:.2f}')
plt.xlabel('Total Stars')
plt.ylabel('GDP per capita')

# Annotate top 5 countries with the highest GDP per capita
top_5_stars_gdp_per_capita = final_data.nlargest(5, 'GDP_per_capita')
for i in range(top_5_stars_gdp_per_capita.shape[0]):
    plt.text(top_5_stars_gdp_per_capita['total_stars'].iloc[i], top_5_stars_gdp_per_capita['GDP_per_capita'].iloc[i], top_5_stars_gdp_per_capita['Country_Standardized'].iloc[i])

# Plot average stars per contributor vs GDP per capita
plt.subplot(1, 3, 3)
sns.scatterplot(x='avg_stars_per_contributor', y='GDP_per_capita', data=final_data)
plt.title(f'Avg Stars per Contributor vs GDP per capita\nCorrelation: {correlation_avg_stars_gdp_per_capita:.2f}')
plt.xlabel('Avg Stars per Contributor')
plt.ylabel('GDP per capita')

plt.tight_layout()
plt.show()

gerd_data = pd.read_csv('GERD.csv')  # Load Gross Domestic Expenditure on R&D (GERD) data

# Create a mapping of country names to their three-letter country codes (ISO 3166-1 alpha-3 codes)
country_name_to_code = {country.name: country.alpha3 for country in countries}

# Add a new column to the country_data DataFrame that contains the ISO alpha-3 country codes
country_data['Code'] = country_data['Country_Standardized'].map(country_name_to_code)

# Filter GERD data to only include data from the year 2022 and for the indicator "EXPGDP.TOT" (expenditure as a percentage of GDP)
gerd_data_2022 = gerd_data[(gerd_data['year'] == 2022) & (gerd_data['indicatorId'] == 'EXPGDP.TOT')]

# Rename columns to standardize the country code and expenditure naming
gerd_data_2022 = gerd_data_2022.rename(columns={'geoUnit': 'Code', 'value': 'GERD'})

# Merge country data with GERD data on the country code
final_data = pd.merge(country_data, gerd_data_2022[['Code', 'GERD']], on='Code', how='inner')

# Print missing values in the merged dataset
print("Missing values in merged dataset:")
print(final_data.isnull().sum())

# Drop any rows with missing values
final_data = final_data.dropna()

# Convert relevant columns to numeric types for calculations
final_data['total_contributors'] = pd.to_numeric(final_data['total_contributors'], errors='coerce')
final_data['total_stars'] = pd.to_numeric(final_data['total_stars'], errors='coerce')
final_data['avg_stars_per_contributor'] = pd.to_numeric(final_data['avg_stars_per_contributor'], errors='coerce')
final_data['GERD'] = pd.to_numeric(final_data['GERD'], errors='coerce')

# Compute correlations between GitHub activity metrics and GERD
correlation_contributors = final_data['total_contributors'].corr(final_data['GERD'])
correlation_stars = final_data['total_stars'].corr(final_data['GERD'])
correlation_avg_stars = final_data['avg_stars_per_contributor'].corr(final_data['GERD'])

# Print correlation results
print(f"Correlation between total contributors and GERD in 2022: {correlation_contributors}")
print(f"Correlation between total stars and GERD in 2022: {correlation_stars}")
print(f"Correlation between average stars per contributor and GERD in 2022: {correlation_avg_stars}")

# Plot scatterplots of the relationships
plt.figure(figsize=(15, 5))

# Plot total contributors vs. GERD
plt.subplot(1, 3, 1)
sns.scatterplot(x='total_contributors', y='GERD', data=final_data)
plt.title(f'Total Contributors vs GERD\nCorrelation: {correlation_contributors:.2f}')
plt.xlabel('Total Contributors')
plt.ylabel('GERD')
# Annotate top 5 countries with highest GERD
top_5_contributors = final_data.nlargest(5, "GERD")
for i in range(top_5_contributors.shape[0]):
    plt.text(top_5_contributors["total_contributors"].iloc[i], top_5_contributors["GERD"].iloc[i], top_5_contributors["Country_Standardized"].iloc[i])

# Plot total stars vs. GERD
plt.subplot(1, 3, 2)
sns.scatterplot(x='total_stars', y='GERD', data=final_data)
plt.title(f'Total Stars vs GERD\nCorrelation: {correlation_stars:.2f}')
plt.xlabel('Total Stars')
plt.ylabel('GERD')
# Annotate top 5 countries
top_5_stars = final_data.nlargest(5, "GERD")
for i in range(top_5_stars.shape[0]):
    plt.text(top_5_stars["total_stars"].iloc[i], top_5_stars["GERD"].iloc[i], top_5_stars["Country_Standardized"].iloc[i])

# Plot average stars per contributor vs. GERD
plt.subplot(1, 3, 3)
sns.scatterplot(x='avg_stars_per_contributor', y='GERD', data=final_data)
plt.title(f'Avg Stars per Contributor vs GERD\nCorrelation: {correlation_avg_stars:.2f}')
plt.xlabel('Avg Stars per Contributor')
plt.ylabel('GERD')
# Annotate top 5 countries
top_5_avg_stars = final_data.nlargest(5, "GERD")
for i in range(top_5_avg_stars.shape[0]):
    plt.text(top_5_avg_stars["avg_stars_per_contributor"].iloc[i], top_5_avg_stars["GERD"].iloc[i], top_5_avg_stars["Country_Standardized"].iloc[i])

# Adjust layout and show plots
plt.tight_layout()
plt.show()

pop_total_data = pd.read_csv('pop_total.csv')

pop_total_data_2023 = pop_total_data[['Country Name', '2023']]
pop_total_data_2023 = pop_total_data_2023.rename(columns={'Country Name': 'Country_Standardized', '2023': 'Population'})

final_data = pd.merge(country_data, pop_total_data_2023, on='Country_Standardized', how='inner')

final_data['star_per_capita'] = final_data['total_stars'] / final_data['Population']
final_data['contributor_per_capita'] = final_data['total_contributors'] / final_data['Population']
final_data['Geographic_AIOSPI'] = 0.5 * final_data['total_contributors'] + 0.5 * final_data['total_stars']
final_data['Per_Capita_AIOSPI'] = 0.5 * (final_data['total_contributors'] / final_data['Population']) + 0.5 * (final_data['total_stars'] / final_data['Population'])

final_data_sorted = final_data.sort_values(by='Geographic_AIOSPI', ascending=False)
print(final_data_sorted[['Country_Standardized', 'star_per_capita', 'contributor_per_capita', 'Geographic_AIOSPI', 'Per_Capita_AIOSPI']])

plt.figure(figsize=(20, 15))
plt.subplot(2, 2, 1)
sns.barplot(x='star_per_capita', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Star per Capita')
plt.xlabel('Star per Capita')
plt.ylabel('Country')
plt.subplot(2, 2, 2)
sns.barplot(x='contributor_per_capita', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Contributor per Capita')
plt.xlabel('Contributor per Capita')
plt.ylabel('Country')
plt.subplot(2, 2, 3)
sns.barplot(x='Geographic_AIOSPI', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Geographic AIOSPI')
plt.xlabel('Geographic AIOSPI')
plt.ylabel('Country')
plt.subplot(2, 2, 4)
sns.barplot(x='Per_Capita_AIOSPI', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Per Capita AIOSPI')
plt.xlabel('Per Capita AIOSPI')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

# Calculate GitHub stars per capita
final_data['star_per_capita'] = final_data['total_stars'] / final_data['Population']

# Calculate GitHub contributors per capita
final_data['contributor_per_capita'] = final_data['total_contributors'] / final_data['Population']

# Compute Geographic AIOSPI (Absolute Index of Open Source Productivity & Influence)
final_data['Geographic_AIOSPI'] = 0.5 * final_data['total_contributors'] + 0.5 * final_data['total_stars']

# Compute Per Capita AIOSPI (AIOSPI adjusted for population size)
final_data['Per_Capita_AIOSPI'] = 0.5 * (final_data['total_contributors'] / final_data['Population']) + 0.5 * (final_data['total_stars'] / final_data['Population'])

# Sort data by Geographic AIOSPI in descending order
final_data_sorted = final_data.sort_values(by='Geographic_AIOSPI', ascending=False)

# Print selected metrics for each country
print(final_data_sorted[['Country_Standardized', 'star_per_capita', 'contributor_per_capita', 'Geographic_AIOSPI', 'Per_Capita_AIOSPI']])

# Set figure size for plots
plt.figure(figsize=(20, 15))

# Plot Star per Capita for each country
plt.subplot(2, 2, 1)
sns.barplot(x='star_per_capita', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Star per Capita')  # Set title
plt.xlabel('Star per Capita')  # Set x-axis label
plt.ylabel('Country')  # Set y-axis label

# Plot Contributor per Capita for each country
plt.subplot(2, 2, 2)
sns.barplot(x='contributor_per_capita', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Contributor per Capita')
plt.xlabel('Contributor per Capita')
plt.ylabel('Country')

# Plot Geographic AIOSPI for each country
plt.subplot(2, 2, 3)
sns.barplot(x='Geographic_AIOSPI', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Geographic AIOSPI')
plt.xlabel('Geographic AIOSPI')
plt.ylabel('Country')

# Plot Per Capita AIOSPI for each country
plt.subplot(2, 2, 4)
sns.barplot(x='Per_Capita_AIOSPI', y='Country_Standardized', data=final_data_sorted, palette='viridis')
plt.title('Per Capita AIOSPI')
plt.xlabel('Per Capita AIOSPI')
plt.ylabel('Country')

# Adjust layout for better visualization
plt.tight_layout()

# Display the plots
plt.show()

# Select the top 10 countries based on Geographic AIOSPI
top_10_data = final_data.sort_values(by='Geographic_AIOSPI', ascending=False).head(10)

# Print selected metrics for the top 10 countries
print(top_10_data[['Country_Standardized', 'star_per_capita', 'contributor_per_capita', 'Geographic_AIOSPI', 'Per_Capita_AIOSPI']])

# Set figure size for plots
plt.figure(figsize=(20, 15))

# Plot Star per Capita for the top 10 countries
plt.subplot(2, 2, 1)
sns.barplot(x='star_per_capita', y='Country_Standardized', data=top_10_data, palette='viridis')
plt.title('Star per Capita (Top 10 Countries)')
plt.xlabel('Star per Capita')
plt.ylabel('Country')

# Plot Contributor per Capita for the top 10 countries
plt.subplot(2, 2, 2)
sns.barplot(x='contributor_per_capita', y='Country_Standardized', data=top_10_data, palette='viridis')
plt.title('Contributor per Capita (Top 10 Countries)')
plt.xlabel('Contributor per Capita')
plt.ylabel('Country')

# Plot Geographic AIOSPI for the top 10 countries
plt.subplot(2, 2, 3)
sns.barplot(x='Geographic_AIOSPI', y='Country_Standardized', data=top_10_data, palette='viridis')
plt.title('Geographic AIOSPI (Top 10 Countries)')
plt.xlabel('Geographic AIOSPI')
plt.ylabel('Country')

# Plot Per Capita AIOSPI for the top 10 countries
plt.subplot(2, 2, 4)
sns.barplot(x='Per_Capita_AIOSPI', y='Country_Standardized', data=top_10_data, palette='viridis')
plt.title('Per Capita AIOSPI (Top 10 Countries)')
plt.xlabel('Per Capita AIOSPI')
plt.ylabel('Country')

# Adjust layout for better visualization
plt.tight_layout()

# Display the plots
plt.show()

publication_data = pd.read_csv('publication_AI.csv')
gdp_per_capita_data = pd.read_csv('GDP_per_capita.csv')

pop_total_data_2023 = pop_total_data[['Country Name', '2023']]
pop_total_data_2023 = pop_total_data_2023.rename(columns={'Country Name': 'Country_Standardized', '2023': 'Population'})

final_data = pd.merge(country_data, pop_total_data_2023, on='Country_Standardized', how='inner')

final_data['star_per_capita'] = final_data['total_stars'] / final_data['Population']
final_data['contributor_per_capita'] = final_data['total_contributors'] / final_data['Population']
final_data['Geographic_AIOSPI'] = 0.5 * final_data['total_contributors'] + 0.5 * final_data['total_stars']
final_data['Per_Capita_AIOSPI'] = 0.5 * (final_data['total_contributors'] / final_data['Population']) + 0.5 * (final_data['total_stars'] / final_data['Population'])

publication_data_2023 = publication_data[publication_data['Year'] == 2023]
publication_data_2023 = publication_data_2023.rename(columns={'Entity': 'Country_Standardized', 'Number of articles - Field: All': 'Publications'})

final_data = pd.merge(final_data, publication_data_2023[['Country_Standardized', 'Publications']], on='Country_Standardized', how='inner')

gdp_per_capita_data_2023 = gdp_per_capita_data[['Country Name', '2023']]
gdp_per_capita_data_2023 = gdp_per_capita_data_2023.rename(columns={'Country Name': 'Country_Standardized', '2023': 'GDP_per_capita'})

final_data = pd.merge(final_data, gdp_per_capita_data_2023[['Country_Standardized', 'GDP_per_capita']], on='Country_Standardized', how='inner')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# Calculate correlation
correlation_publications_geo_aiospi = final_data['Publications'].corr(final_data['Geographic_AIOSPI'])
correlation_gdp_per_capita_per_capita_aiospi = final_data['GDP_per_capita'].corr(final_data['Per_Capita_AIOSPI'])

print(f"Correlation between Publications and Geographic AIOSPI: {correlation_publications_geo_aiospi}")
print(f"Correlation between GDP per capita and Per Capita AIOSPI: {correlation_gdp_per_capita_per_capita_aiospi}")

# Linear regression: Publications vs Geographic_AIOSPI
X_publications = sm.add_constant(final_data['Publications'])
model_publications = sm.OLS(final_data['Geographic_AIOSPI'], X_publications).fit()
print(model_publications.summary())

# Linear regression: GDP per capita vs Per Capita AIOSPI
X_gdp_per_capita = sm.add_constant(final_data['GDP_per_capita'])
model_gdp_per_capita = sm.OLS(final_data['Per_Capita_AIOSPI'], X_gdp_per_capita).fit()
print(model_gdp_per_capita.summary())

# Visualize regression results
plt.figure(figsize=(15, 10))

# Publications vs Geographic_AIOSPI
plt.subplot(2, 1, 1)
sns.regplot(x='Publications', y='Geographic_AIOSPI', data=final_data)
plt.title(f'Publications vs Geographic AIOSPI\nCorrelation: {correlation_publications_geo_aiospi:.2f}')
plt.xlabel('Publications')
plt.ylabel('Geographic AIOSPI')

# GDP per capita vs Per Capita AIOSPI
plt.subplot(2, 1, 2)
sns.regplot(x='GDP_per_capita', y='Per_Capita_AIOSPI', data=final_data)
plt.title(f'GDP per capita vs Per Capita AIOSPI\nCorrelation: {correlation_gdp_per_capita_per_capita_aiospi:.2f}')
plt.xlabel('GDP per capita')
plt.ylabel('Per Capita AIOSPI')

plt.tight_layout()
plt.show()