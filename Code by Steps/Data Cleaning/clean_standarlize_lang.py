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









