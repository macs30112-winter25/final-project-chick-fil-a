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