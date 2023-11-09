import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd



# Load the dataset
df = pd.read_csv("dataset/pip_dataset.csv")



# Filter the DataFrame for the year 2019 data
df_2019 = df[(df['year'] == 2019) & (df['ppp_version'] == 2011) & (df['reporting_level'] == "national")]

# Filter the DataFrame for the most recent year data for each country
most_recent_year = df[df.groupby('country')['year'].transform(max) == df['year']]

# Group by country and calculate the mean of headcount_ratio_3000 for each country for 2019
result_2019 = df_2019.groupby('country')['headcount_ratio_3000'].mean()

# Group by country and calculate the mean of headcount_ratio_3000 for the most recent year for each country
result_most_recent_year = most_recent_year.groupby('country')['headcount_ratio_3000'].mean()

# Fill missing values in 2019 data with data from the most recent year
result = result_2019.combine_first(result_most_recent_year)



# Load the world map shapefile
world = gpd.read_file('dataset/World_Countries_Generalized.geojson')
world['COUNTRY'] = world['COUNTRY'].replace("Russian Federation", "Russia")
world['COUNTRY'] = world['COUNTRY'].replace("Congo DRC", "Democratic Republic of Congo")
world['COUNTRY'] = world['COUNTRY'].replace("Czech Republic", "Czechia")
world['COUNTRY'] = world['COUNTRY'].replace("Turkiye", "Turkey")





