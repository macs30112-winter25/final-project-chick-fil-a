import pandas as pd
import numpy as np
import time
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import googlemaps 

file_path = "/Users/wangbaihui/contributors_locations.csv"
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
