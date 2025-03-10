# Global Spatial Distribution Of Open-Source Artificial Intelligence - Group Chick-Fil-A

## Team Member
- Charlotte Li
- Baihui Wang 
- Anqi Wei

## Project Goals
This project analyzes the global distribution of open-source AI development on GitHub, mapping geographic hotspots of contributors and repositories. It investigates correlations with socioeconomic factors (GDP, R&D investment, population) and academic output while introducing metrics like *Geographic AIOSPI* and *Per Capita AIOSPI* to quantify influence. Through interactive visualizations (HeatMaps, network graphs) and statistical analysis, it aims to identify drivers of AI innovation and inform strategies to bridge global disparities in tech collaboration.

## Research Question
- **Top Contributing Regions**: Which cities or countries dominate AI-related open-source projects?
- **Role of Economic Development and Higher Education**: How does economic development or the density of higher education institutions explain these geographic patterns?

## Repo Navigation
- *All_In_One_Code.py*: Aggregated code for the project. However, the full script exceeds 1,300+ lines, and certain sections (e.g., the geocoding module) may require up to 5 hours to execute due to API rate limits and large-scale data processing. 
- Data: Input files needed for *All_In_One_Code.py*, including external datasets (economic/research metrics) and generated intermediate files.
For easier navigation and execution, we recommend reviewing *Code by Steps* folder. These modular scripts break down the workflow into manageable components, improving clarity and reducing runtime dependencies.
- Code by Steps: Break down modular scripts by data scraping, data cleaning, data visualization, and data analysis.
- Project Report: Include 2 progress reports and 2 versions of presentation (Original & Update).

## Data Sources 
- [GitHub (AI Repositories)](https://github.com/topics/ai): Data on repositories tagged with "AI," sorted by the number of stars, including public user profile information with location fields. We will use the top 100 most-starred repositories, covering approximately 1,514 contributors and their locations.
- [World Bank Total Population](https://www.bing.com/search?pglt=2339&q=world+bank+total+population&cvid=3ad414a6781e449a9a830473747e2d65&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQABhAMgYIAhAAGEAyBggDEAAYQDIGCAQQABhAMgYAyBggFEAAYQDIGCAYQABhAMgYIBxAAGEAyBggIEAAYQNIBQg4MDc5ajBqMagCALACAA&FORM=ANNTA1&PC=U531): Provides up-to-date global population estimates and historical trends based on World Bank data. Covers demographic trends from 2020–2024.  
- [Our World in Data - AI Scholarly Publications](https://ourworldindata.org/grapher/annual-scholarly-publications-on-artificial-intelligence): Tracks the annual number of scholarly publications on artificial intelligence, providing insights into research output growth and trends.  
- [UNESCO GERD as a Percentage of GDP](https://databrowser.uis.unesco.org/): Data on Gross Domestic Expenditure on Research and Development (GERD) as a percentage of GDP, offering a measure of national investment in research and innovation across various countries from 2020–2024.

## Required Packages (can also see *requirements.txt*)
- python>=3.8
- requests==2.28.2
- pandas==2.0.3
- numpy==1.24.3
- geopy==2.3.0
- googlemaps==4.10.0
- pycountry==22.3.5
- matplotlib==3.7.1
- seaborn==0.12.2
- folium==0.14.0
- networkx==3.1
- pyvis==0.3.2
- deep-translator==1.11.4
- opencc-python-reimplemented==0.1.7
- iso3166==2.1.1
- statsmodels==0.14.0
- google-cloud-translate==2.11.1
- langdetect==1.0.9
- pyyaml==6.0.1
- google-api-python-client==2.104.0

## Presention & Illustration
- [Original Presentation](https://github.com/macs30112-winter25/final-project-chick-fil-a/blob/3431ec98d10c2042a8730fef0bcb82da1ba1a9e5/Project%20report/Original%20Slides.pdf)
- [Updated Presentation]()
- [Video Illustration](https://drive.google.com/file/d/1DYlvbbOr0yDNUHhSNlcY_oGL9oRAKPWN/view?usp=sharing)

## Responsibility
- Charlotte Li: Develop and execute the GitHub data scraping script, including API access, repository information collection, and user location data extraction, revise PowerPoint slides incorporating feedback.
- Baihui Wang: Assist in data scraping, perform data cleaning and deduplication, aggregate macroeconomic data, and standardize geographic names. Contribute to geocoding, data visualization and interactive map creation, record instructional video introducing GitHub page features
- Anqi Wei: Support data scraping and cleaning, conduct regression analysis and correlation testing, and create visualizations, including mapping AI activity distributions, reorganize GitHub repository page, update README documentation
