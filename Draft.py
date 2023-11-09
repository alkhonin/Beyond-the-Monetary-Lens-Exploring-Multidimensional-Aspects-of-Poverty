import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px

# Load the dataset
df = pd.read_csv("dataset/pip_dataset.csv")

# Filter the DataFrame for the year 2019 data
df_2019 = df[(df['year'] == 2019) & (df['ppp_version'] == 2011) & (df['reporting_level'] == "national")]

# Filter the DataFrame for the most recent year data for each country
most_recent_year = df[df.groupby('country')['year'].transform(max) == df['year']]

result_2019 = df_2019.groupby('country')['headcount_ratio_3000'].mean()

result_most_recent_year = most_recent_year.groupby('country')['headcount_ratio_3000'].mean()

result = result_2019.combine_first(result_most_recent_year)

# Load the world map shapefile and rename some countries names to align  with the dataset
world = gpd.read_file('dataset/World_Countries_Generalized.geojson')
world['COUNTRY'] = world['COUNTRY'].replace("Russian Federation", "Russia")
world['COUNTRY'] = world['COUNTRY'].replace("Congo DRC", "Democratic Republic of Congo")
world['COUNTRY'] = world['COUNTRY'].replace("Czech Republic", "Czechia")
world['COUNTRY'] = world['COUNTRY'].replace("Turkiye", "Turkey")


# Merge the world map data with the result data on country names
merged = world.set_index('COUNTRY').join(result_most_recent_year)

fig = px.choropleth(merged,
                   locations=merged.index,
                   locationmode='country names',
                   color='headcount_ratio_3000',
                   color_continuous_scale='OrRd',
                   hover_name=merged.index,
                   labels={'headcount_ratio_3000': '% of population.'},
                   title='% of population living in households with an income or expenditure per person below $30 a day.')

fig.show()