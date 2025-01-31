"""
MACSS 30112 Winter 2025
Final Project
Group Chick-Fil-A
Charlotte Li, Baihui Wang & Anqi Wei
"""

import requests
import csv
import time
import random

# GitHub API settings
GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_TOKEN = "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"  # Replace with a GitHub token
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "Mozilla/5.0"
}

def request_with_retries(url, params=None):
    """Make a request to the given URL with retries."""
    retries = 5
    for i in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            wait_time = (2 ** i) + random.uniform(0, 1)
            print(f"Request failed ({e}), retrying in {wait_time:.2f} seconds...")

def get_most_starred_repos(topic, max_repos=10):
    """Fetch the most starred repositories for a given topic using GitHub API."""
    params = {
        "q": f"topic:{topic}",
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos
    }
    return request_with_retries(GITHUB_API_URL, params=params)

def get_user_location(username):
    """Fetch the location of a GitHub user."""
    user_api_url = f"https://api.github.com/users/{username}"
    data = request_with_retries(user_api_url)
    return data.get("location", "") if data else ""

def get_contributors_locations(owner, repo_name):
    """Fetch the locations of contributors of a GitHub repository."""
    contributors_api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contributors"
    contributors = request_with_retries(contributors_api_url)
    locations = []
    if contributors:
        for contributor in contributors:
            location = get_user_location(contributor["login"])
            if location:
                locations.append(location)
    return locations

def save_to_csv(data, filename):
    """Save data to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Repository", "Stars", "Contributor", "Location"])
        for row in data:
            writer.writerow(row)

def main():
    topic = "ai"
    max_repos = 10
    repos = get_most_starred_repos(topic, max_repos)
    data = []
    for repo in repos.get('items', []):
        owner = repo['owner']['login']
        repo_name = repo['name']
        stars = repo['stargazers_count']
        print(f"Fetching locations for contributors of {repo_name} ({stars} stars)...")
        locations = get_contributors_locations(owner, repo_name)
        for location in locations:
            data.append([repo_name, stars, owner, location])
    
    save_to_csv(data, "contributors_locations.csv")

if __name__ == "__main__":
    main()