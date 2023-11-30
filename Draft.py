import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display


def filter_and_aggregate(df: pd.DataFrame, year: int, ppp_version: int, column: str) -> pd.Series:
    """

    :param df: Input DataFrame containing the dataset.
    :param year: The specific year to filter the data.
    :param ppp_version: The PPP version to filter the data.
    :param column: The column name for which the mean is calculated.
    :return: A pandas Series containing the mean values of the specified column
                   aggregated by country after filtering and aggregation.
    """
    filtered_df = df[(df['year'] == year) & (df['ppp_version'] == ppp_version) & (df['reporting_level'] == "national")]
    most_recent_year = df[df.groupby('country')['year'].transform(max) == df['year']]
    result_year = filtered_df.groupby('country')[column].mean()
    result_most_recent_year = most_recent_year.groupby('country')[column].mean()
    result_combined = result_year.combine_first(result_most_recent_year)
    return result_combined


def replace_country_names(world: gpd.geodataframe.GeoDataFrame) -> gpd.geodataframe.GeoDataFrame:
    """
    Replaces specific country names in the input GeoDataFrame.
    :param world: Input GeoDataFrame containing country shapes.
    :return: A new GeoDataFrame with replaced country names.
    """
    world['COUNTRY'] = world['COUNTRY'].replace({
        "Russian Federation": "Russia",
        "Congo DRC": "Democratic Republic of Congo",
        "Czech Republic": "Czechia",
        "Turkiye": "Turkey"
    })
    return world


def create_choropleth_map(world: gpd.geodataframe.GeoDataFrame,
                          result: pd.DataFrame,
                          color_column: str,
                          title: str,
                          color_scale: str = 'OrRd') -> None:
    """
    Creates an interactive choropleth map using Plotly Express.
    :param world: GeoDataFrame containing world map data.
    :param result: DataFrame containing data to be mapped.
    :param color_column: Column in 'result' DataFrame to use for coloring the map.
    :param color_scale: Plotly color scale for the choropleth map.
    :param title: Title for the choropleth map.
    :return: this function does not return.
    """

    merged = world.set_index('COUNTRY').join(result)

    fig = px.choropleth(merged,
                        locations=merged.index,
                        locationmode='country names',
                        color=color_column,
                        color_continuous_scale=color_scale,
                        hover_name=merged.index,
                        labels={color_column: '% of population.'},
                        title=title)

    fig.show()


def time_series(df: pd.DataFrame, column: str, ppp_version: int) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by selecting specific columns, grouping data by 'year' and 'country',
    and calculating the mean of the specified column.

    :param df: Input DataFrame containing necessary columns ('year', 'country', specified column).
    :param column: Name of the column to calculate the mean.
    :param ppp_version: PPP version for filtering the data.
    :return: Processed DataFrame containing time-series data with columns ('year', 'country', specified column).
    """
    # Filter data based on 'ppp_version' and 'reporting_level'
    filtered_df = df[(df['ppp_version'] == ppp_version) & (df['reporting_level'] == "national")]

    # Select specific columns
    filtered_df = filtered_df[['year', 'country', column]]

    # Group data by 'year' and 'country' and calculate the mean of the specified column
    grouped_data = filtered_df.groupby(['year', 'country'])[column].mean().reset_index()

    return grouped_data


def plot_time_series(time_series_data: pd.DataFrame, selected_countries: list) -> None:
    """
    Create an interactive line plot for selected countries using Plotly Express.

    :param time_series_data: Processed DataFrame containing time-series data.
    :param selected_countries: List of country names to be included in the plot.
    """
    fig = px.line(time_series_data[time_series_data['country'].isin(selected_countries)],
                  x='year',
                  y='headcount_ratio_3000',
                  color='country',
                  markers=True,
                  title='Share of Population Living on Less than $30 a Day (1977-2019)',
                  labels={'headcount_ratio_3000': '% of Population Living on <$30/day'},
                  template='plotly_white')

    fig.show()


def create_pie_chart(labels, sizes, title='Pie Chart'):
    """

    :param labels:List of labels for each category.
    :param sizes: List of sizes or percentages for each category.
    :param title: Title for the pie chart. Default is 'Pie Chart'.
    :return:
    """
    # Create a pie chart using Plotly Express
    fig = px.pie(names=labels, values=sizes, title=title)

    # Show the chart
    fig.show()


def preprocess_poverty_and_GDP_data(poverty_file_path, gdp_file_path):
    """
    Load and preprocess poverty and GDP data.

    :param poverty_file_path: Path to the CSV file containing poverty data.
    :param gdp_file_path: Path to the Excel file containing GDP data.
    :return: A merged DataFrame with preprocessed data.
    """
    # Load poverty data and calculate poverty rate
    poverty_data = pd.read_csv(poverty_file_path)
    poverty_data['population_under_poverty'] = (
                                                       poverty_data[
                                                           '‘cost of basic needs’ approach - number below poverty line'] /
                                                       (poverty_data[
                                                            '‘cost of basic needs’ approach - number above poverty line'] +
                                                        poverty_data[
                                                            '‘cost of basic needs’ approach - number below poverty line'])
                                               ) * 100

    # Rename entities to maintain consistency
    poverty_data['Entity'] = poverty_data['Entity'].replace({
        'Eastern Europe and former USSR': 'Eastern Europe',
        'South and South-East Asia': 'South and Southeast Asia'
    })

    # Load GDP data, reshape it, and clean column names
    gdp_data = pd.read_excel(gdp_file_path, header=1, nrows=21)
    gdp_data = gdp_data.rename(columns={
        'Unnamed: 0': 'Year',
        'South and Southeast Asia)': 'South and Southeast Asia',
        'Latin America and Carribean': 'Latin America and Caribbean',
        'East Asia ': 'East Asia'
    })
    melted_gdp = gdp_data.melt(id_vars=['Year'], var_name='Entity', value_name='GDP_per_capita')

    # Merge the poverty and GDP data
    merged_data = pd.merge(melted_gdp, poverty_data, on=['Year', 'Entity'], how='inner')

    # Clean and convert GDP data to numeric
    merged_data['GDP_per_capita'] = (
        merged_data['GDP_per_capita']
        .str.replace('[^\d.]', '', regex=True)
        .replace('', float('nan'))
        .pipe(pd.to_numeric, errors='coerce')
    )

    return merged_data, poverty_data


def create_area_chart(data, entity):
    """
    Create an area chart for a given entity.
    """
    entity_data = data[data['Entity'] == entity]
    fig = px.area(entity_data, x='Year', y='population_under_poverty',
                  labels={'population_under_poverty': 'Percentage of Population'},
                  title=f'Average Share of Population Living in Extreme Poverty - {entity}')
    fig.update_layout(
        xaxis_title='Year', yaxis_title='Percentage of Population',
        plot_bgcolor='white', yaxis=dict(gridcolor='lightgrey', range=[0, 100]),
        xaxis=dict(gridcolor='lightgrey'))
    fig.show()


def create_individual_chart(data, region):
    """
    Create a line chart for a specified region.
    """
    filtered = data[(data['Entity'] == region) & (~data['GDP_per_capita'].isna())]
    fig = px.line(filtered, x='GDP_per_capita', y='population_under_poverty', text='Year', markers=True,
                  title=f'GDP per Capita vs. Poverty Rate Over Time - {region}')
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title='GDP per Capita', yaxis_title='Poverty Rate (%)', showlegend=False)
    fig.show()


def create_comparison_chart(data, regions, standout_region):
    """
    Create a comparison chart for multiple regions with one standout region.
    """
    fig = go.Figure()
    for region in regions:
        filtered = data[(data['Entity'] == region) & (~data['GDP_per_capita'].isna())]
        line_width = 3 if region == standout_region else 1
        opacity = 1 if region == standout_region else 0.5
        fig.add_trace(go.Scatter(
            x=filtered['GDP_per_capita'], y=filtered['population_under_poverty'],
            mode='lines+markers', name=region,
            line=dict(width=line_width, color='black' if region == standout_region else None),
            opacity=opacity))
    fig.update_layout(
        title='GDP per Capita vs. Poverty Rate Over Time',
        xaxis_title='GDP per Capita', xaxis_type='log', yaxis_title='Poverty Rate (%)',
        hovermode='x unified', showlegend=True)
    fig.add_annotation(
        text=standout_region,
        xref="paper", x=0.8,
        yref="paper", y=0.8,
        showarrow=False,
        font=dict(size=20),
        align="right",
        xanchor="right", yanchor="top"
    )
    fig.update_xaxes(title_text='GDP per Capita', tickvals=[1000, 5000, 10000, 20000, 50000])
    fig.show()


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv("dataset/pip_dataset.csv")
    world = gpd.read_file('dataset/World_Countries_Generalized.geojson')
    world = replace_country_names(world)
    result_2019_2011_info = filter_and_aggregate(df, 2019, 2011, 'headcount_ratio_3000')
    create_choropleth_map(world, result_2019_2011_info, 'headcount_ratio_3000',
                          '% of population living in households with an income or expenditure per person below $30 a day.', )

    selected_countries = ['India', 'Poland', 'Spain', 'South Korea', 'Denmark', 'Norway']
    time_series_data = time_series(df, 'headcount_ratio_3000', 2011)
    plot_time_series(time_series_data, selected_countries)


    filtered_df = df[(df['year'] == 2019) & (df['country'] == "World") ]
    labels = ['Less than $30', 'More than $30']
    sizes = [filtered_df['headcount_ratio_3000'].median(), 100 - filtered_df['headcount_ratio_3000'].median()]
    create_pie_chart(labels, sizes, title='$30-a-day-poverty-line')

    labels = ['Below $2.15 a day', '2.15 - $10 a day', '10 - $30 a day', 'Above $30 a day']
    sizes = [
        filtered_df['headcount_ratio_international_povline'].median(),
        filtered_df['headcount_ratio_1000'].median() - filtered_df['headcount_ratio_international_povline'].median(),
        filtered_df['headcount_ratio_3000'].median() - filtered_df['headcount_ratio_1000'].median(),
        100 - filtered_df['headcount_ratio_3000'].median()
    ]
    create_pie_chart(labels, sizes, title='$2.15 poverty line')


    regions = [
        "World", "Western Europe", "Western Offshoots",
        "Middle East and North Africa", "Eastern Europe", "Sub-Saharan Africa",
        "East Asia", "South and Southeast Asia", "Latin America and Caribbean"
    ]
    file2, file1 = preprocess_poverty_and_GDP_data(
        'dataset/share-of-population-living-in-extreme-poverty-cost-of-basic-needs.csv',
        'dataset/GDP_by_region.xlsx')
    create_area_chart(file1, "World")
    create_individual_chart(file2, "World")
    create_comparison_chart(file2, regions, "World")
