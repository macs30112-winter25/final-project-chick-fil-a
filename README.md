# The Spatial Distribution of Open-Source AI: Regional Patterns of Innovation and Participation in the U.S. - Group Chick-Fil-A
## Team Member
- Charlotte Li
- Baihui Wang 
- Anqi Wei
## Project Goals
This study examines the geographic distribution of open-source AI research in the U.S. and its relationship with economic development and higher education resources. Using GitHub repositories tagged with “AI” and sorted by popularity, we will extract geographic data from top contributors, leveraging the number of stars as a measure of engagement. We will geocode user locations through GitHub profiles, Bing Maps API, and university email domains, aiming for a dataset of at least 5,000 users. Through data visualization, we will analyze AI participation at city and state levels, correlating it with macroeconomic indicators such as GDP and higher education presence. Finally, we will compare AI engagement with the broader open-source ecosystem to assess diffusion and potential disparities.
## Research Question
- **Top Contributing Regions**: Which cities or countries dominate AI-related open-source projects?
- **Role of Economic Development and Higher Education**: How does economic development or the density of higher education institutions explain these geographic patterns?
## Repo Navigation
- Data：
- Code:
- Report: Include 2 progress reports and 2 versions of presentation (Original & Update).
## Data Sources 
- [GitHub (AI Repositories)](https://github.com/topics/ai): Data on repositories tagged with "AI," sorted by the number of stars, including public user profile information with location fields. We will use the top 100 most-starred repositories, covering approximately 1,514 contributors and their locations.
- [World Bank Total Population](https://www.bing.com/search?pglt=2339&q=world+bank+total+population&cvid=3ad414a6781e449a9a830473747e2d65&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQABhAMgYIAhAAGEAyBggDEAAYQDIGCAQQABhAMgYAyBggFEAAYQDIGCAYQABhAMgYIBxAAGEAyBggIEAAYQNIBQg4MDc5ajBqMagCALACAA&FORM=ANNTA1&PC=U531): Provides up-to-date global population estimates and historical trends based on World Bank data. Covers demographic trends from 2020–2024.  
- [Our World in Data - AI Scholarly Publications](https://ourworldindata.org/grapher/annual-scholarly-publications-on-artificial-intelligence): Tracks the annual number of scholarly publications on artificial intelligence, providing insights into research output growth and trends.  
- [UNESCO GERD as a Percentage of GDP](https://databrowser.uis.unesco.org/): Data on Gross Domestic Expenditure on Research and Development (GERD) as a percentage of GDP, offering a measure of national investment in research and innovation across various countries from 2020–2024.
## Required Libraries

## Presention & Illustration
- [Original Presentation]()
- [Updated Presentation]()
- [Video Illustration]()

## Responsibility
- Charlotte Li: Develop and execute the GitHub data scraping script, including API access, repository information collection, and user location data extraction.
- Baihui Wang: Assist in data scraping, perform data cleaning and deduplication, aggregate macroeconomic data, and standardize geographic names. Contribute to geocoding, data visualization and interactive map creation.
- Anqi Wei: Support data scraping and cleaning, conduct regression analysis and correlation testing, and create visualizations, including mapping AI activity distributions.
