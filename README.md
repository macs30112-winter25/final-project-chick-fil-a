# Spatial Distribution of Open-Source Artificial Intelligence (AI) Research across the United States - Group Chick-Fil-A
## Project Goals
This study investigates the spatial distribution of open-source artificial intelligence (AI) research across the United States, focusing on how economic development and higher education resources correlate with these patterns. We will draw on GitHub repositories tagged with “AI” and sorted by popularity (currently around 42,559 public repositories), emphasizing those with substantial community recognition. From each selected repository, we plan to extract the owner’s or a primary contributor’s geographic information, using the number of stars as a proxy for technical sophistication or user engagement. This approach will enable us to create a city- and state-level map of AI-related participation.
We will first rely on self-reported user locations in GitHub profiles, then use the Bing Maps API and curated location datasets for geocoding. For users without explicit location data, we will consult university-associated email domains to infer geographical details, aiming to assemble a dataset of at least 5,000 users. Through data visualization, we will compare both overall and per-capita engagement in AI development across U.S. regions, correlating our findings with macroeconomic indicators such as local GDP and the number of higher education institutions. Finally, we will contrast our results with the broader open-source ecosystem to assess the degree of diffusion and potential inequities in AI technology engagement.
## Research Question
Although the information industry is often described as highly decentralized, cutting-edge technology firms and related R&D activities may exhibit stronger spatial clustering than traditional industries, with significant implications for regional economic growth and social equity. GitHub, as a key open-source platform, offers a valuable vantage point for examining digital society’s capacity for innovation and its level of engagement with advanced technologies. While previous research has documented the clustering of AI companies in the U.S., it remains unclear whether open-source collaboration—by its cross-regional nature —leads to more evenly distributed geographic participation. Which U.S. cities or states contribute most to AI open-source projects? Can economic development and the density of higher education institutions explain any observed patterns? By exploring these questions, we aim to shed light on how regional resources shape innovation within the open-source community.
## Overall findings
## Team Member
- Charlotte Li
- Baihui Wang
- Anqi Wei
## Data
- https://github.com/topics/ai: Data on repositories tagged with "AI," sorted by the number of stars, and public user profile information, including location fields. Preliminary tests confirm the availability of sufficient repository and user data for analysis.
- https://nces.ed.gov/ipeds/use-the-data#SearchExistingData: Data on the number and locations of higher education institutions in the U.S.
- https://www.bea.gov/; https://www.census.gov/: Data on GDP, population, and other macroeconomic indicators for U.S. cities and states.
- https://aiindex.stanford.edu/report/: Supplementary insights on AI research trends and regional disparities.
## Library
## Responsibility
- Charlotte Li: Develop and execute the GitHub data scraping script, including API access, repository information collection, and user location data extraction.
- Baihui Wang: Assist in data scraping, perform initial data cleaning and deduplication, aggregate macroeconomic data, and standardize geographic names. Contribute to data visualization and map creation.
- Anqi Wei: Support data scraping and cleaning, conduct regression analysis and correlation testing, and create visualizations, including mapping AI activity distributions.
