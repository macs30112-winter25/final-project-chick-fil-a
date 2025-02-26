# The Spatial Distribution of Open-Source AI: Regional Patterns of Innovation and Participation in the U.S. - Group Chick-Fil-A
## Project Goals
This study examines the geographic distribution of open-source AI research in the U.S. and its relationship with economic development and higher education resources. Using GitHub repositories tagged with “AI” and sorted by popularity, we will extract geographic data from top contributors, leveraging the number of stars as a measure of engagement. We will geocode user locations through GitHub profiles, Bing Maps API, and university email domains, aiming for a dataset of at least 5,000 users. Through data visualization, we will analyze AI participation at city and state levels, correlating it with macroeconomic indicators such as GDP and higher education presence. Finally, we will compare AI engagement with the broader open-source ecosystem to assess diffusion and potential disparities.
## Research Question
- **Top Contributing Regions**: Which U.S. cities or states contribute the most to AI-related open-source projects?
- **Role of Economic Development and Higher Education**: How does economic development or the density of higher education institutions explain these geographic patterns?
- **Comparison with AI Industry Clustering**: Is AI open-source development more geographically decentralized compared to AI companies and R&D centers?
## Files Illustration
- Data Collection: The complete code and output for data scraping
- Data Cleaning: The complete code and output for data cleaning and wrangling, based on city and country level according to our research
- Data Visualization: In progress
- Data Analysis: In progress
## Team Member
- Charlotte Li
- Baihui Wang 
- Anqi Wei
## Data
- [GitHub (AI Repositories)](https://github.com/topics/ai): Data on repositories tagged with "AI," sorted by the number of stars, including public user profile information with location fields. We will use the top 100 most-starred repositories, covering approximately 1,514 contributors and their locations.
- [World Higher Education Database (WHED)](https://www.whed.net/home.php): Provides authoritative information on over 21,000 accredited higher education institutions across 180+ countries, with the latest data from 2024.
- [World Bank (World Development Indicators)](https://databank.worldbank.org/): Global data on GDP, economic growth, and development indicators for over 190 countries, spanning from 2020–2024.
- [Stanford AI Index Report](https://aiindex.stanford.edu/report/): Supplementary insights on AI research trends and regional disparities, using the most recent 2023 report.
- [World Bank Internet Usage Data](https://data.worldbank.org/indicator/IT.NET.USER.ZS): Data on internet penetration rates as a percentage of the population, updated annually, covering data from 2020–2024.
## Responsibility
- Charlotte Li: Develop and execute the GitHub data scraping script, including API access, repository information collection, and user location data extraction.
- Baihui Wang: Assist in data scraping, perform initial data cleaning and deduplication, aggregate macroeconomic data, and standardize geographic names. Contribute to data visualization and map creation.
- Anqi Wei: Support data scraping and cleaning, conduct regression analysis and correlation testing, and create visualizations, including mapping AI activity distributions.
