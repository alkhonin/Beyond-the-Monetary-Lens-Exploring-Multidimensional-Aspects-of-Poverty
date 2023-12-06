import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display


def filter_and_aggregate(df: pd.DataFrame, year: int, column: str, ppp_version=None, reporting_level=None) -> pd.Series:
    """
    :param df: Input DataFrame containing the dataset.
    :param year: The specific year to filter the data.
    :param column: The column name for which the mean is calculated.
    :param ppp_version: The PPP version to filter the data (optional).
    :param reporting_level: The reporting level to filter the data (optional).
    :return: A pandas Series containing the mean values of the specified column aggregated by country.
    """
    # Apply filters based on provided parameters
    filtered_df = df[df['year'] == year]
    if ppp_version is not None:
        filtered_df = filtered_df[filtered_df['ppp_version'] == ppp_version]
    if reporting_level is not None:
        filtered_df = filtered_df[filtered_df['reporting_level'] == reporting_level]

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


def plot_time_series(time_series_data: pd.DataFrame, selected_countries: list, title: str) -> None:
    """
    Create an interactive line plot for selected countries using Plotly Express, plotting the last column of the DataFrame.

    :param time_series_data: Processed DataFrame containing time-series data.
    :param selected_countries: List of country names to be included in the plot.
    :param title: Title for the plot.
    """
    # Automatically select the last column for plotting
    column_to_plot = time_series_data.columns[-1]

    fig = px.line(time_series_data[time_series_data['country'].isin(selected_countries)],
                  x='year',
                  y=column_to_plot,
                  color='country',
                  markers=True,
                  title=title,
                  labels={column_to_plot: '% of Population'})

    fig.show()

    

def plot_last_column_time_series(time_series_data: pd.DataFrame, selected_countries: list, title: str) -> None:
    """
    Create an interactive line plot for selected countries using Plotly Express, plotting the last column of the DataFrame.

    :param time_series_data: Processed DataFrame containing time-series data.
    :param selected_countries: List of country names to be included in the plot.
    :param title: Title for the plot.
    """
    # Automatically select the last column for plotting
    column_to_plot = time_series_data.columns[-1]

    fig = px.line(time_series_data[time_series_data['country'].isin(selected_countries)],
                  x='year',
                  y=column_to_plot,
                  color='country',
                  markers=True,
                  title=title,
                  labels={column_to_plot: '% of Population'})

    fig.show()



def create_2slice_pie_chart(df, year, country, poverty_threshold, title):
    """
    Create a pie chart for a specified poverty line threshold with error handling.

    :param df: DataFrame containing the data.
    :param year: The year for which the data is required.
    :param country: The country for which the data is required.
    :param poverty_threshold: The threshold value for poverty.
    :param title: Title for the pie chart.
    """
    # Mapping of poverty thresholds to column names
    column_mapping = {
        2.15: 'headcount_ratio_international_povline',
        10: 'headcount_ratio_1000',
        20: 'headcount_ratio_2000',
        30: 'headcount_ratio_3000'
    }

    # Determine the column name based on the poverty threshold
    column_name = column_mapping.get(poverty_threshold)

    # Check if the column name is found in the mapping
    if column_name is None:
        raise ValueError(f"Invalid poverty line provided: {poverty_threshold}. Valid options are 2.15, 10, 20, 30.")

    # Filter the DataFrame
    filtered_df = df[(df['year'] == year) & (df['country'] == country) & (df['ppp_version'] == 2017)]

    # Extract the poverty rate
    poverty_rate = filtered_df[column_name].iloc[0]

    # Create labels and sizes for the pie chart
    labels = [f'Less than ${poverty_threshold}', f'More than ${poverty_threshold}']
    sizes = [poverty_rate, 100 - poverty_rate]

    # Create and display the pie chart
    fig = px.pie(names=labels, values=sizes, title=title)
    fig.show()

def create_detailed_poverty_pie_chart(df, year, country, title):
    # Filter the DataFrame
    filtered_df = df[(df['year'] == year) & (df['country'] == country) & (df['ppp_version'] == 2017)]

    # Extract values for each segment
    poverty_rate_215 = filtered_df['headcount_ratio_international_povline'].iloc[0]
    poverty_rate_1000 = filtered_df['headcount_ratio_1000'].iloc[0]
    poverty_rate_3000 = filtered_df['headcount_ratio_3000'].iloc[0]

    # Calculate sizes for each segment
    sizes = [
        poverty_rate_215,
        poverty_rate_1000 - poverty_rate_215,
        poverty_rate_3000 - poverty_rate_1000,
        100 - poverty_rate_3000
    ]

    # Labels for the pie chart
    labels = ['Below $2.15 a day', '2.15 - $10 a day', '10 - $30 a day', 'Above $30 a day']

    # Create and display the pie chart
    fig = px.pie(names=labels, values=sizes, title=title)
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




def create_comparison_chart(data, regions, standout_region, show_years=False):
    fig = go.Figure()
    for region in regions:
        filtered = data[(data['Entity'] == region) & (~data['GDP_per_capita'].isna())]
        line_width = 3 if region == standout_region else 1
        opacity = 1 if region == standout_region else 0.5
        is_standout = region == standout_region
        text_mode = '+text' if is_standout and show_years else ''
        trace = go.Scatter(
            x=filtered['GDP_per_capita'], y=filtered['population_under_poverty'],
            mode='lines+markers' + text_mode, name=region,
            line=dict(width=line_width, color='black' if is_standout else None),
            opacity=opacity,
            text=filtered['Year'] if is_standout and show_years else None,
            textposition="top right" if is_standout and show_years else None
        )
        fig.add_trace(trace)

    fig.update_layout(
        title='GDP per Capita vs. Poverty Rate Over Time',
        xaxis_title='GDP per Capita', xaxis_type='log', yaxis_title='Poverty Rate (%)',
        hovermode='x unified', showlegend=True)
    fig.add_annotation(
        text=standout_region,
        xref="paper", x=0.8, yref="paper", y=0.8,
        showarrow=False, font=dict(size=20),
        align="right", xanchor="right", yanchor="top"
    )
    fig.update_xaxes(title_text='GDP per Capita', tickvals=[1000, 5000, 10000, 20000, 50000])
    fig.show()


def create_time_series_plot(data, countries, title, y_label):
    """
    Create an interactive line plot for selected countries.

    :param data: DataFrame containing the data.
    :param countries: List of countries to include in the plot.
    :param title: Title of the plot.
    :param y_label: Label for the y-axis.
    """
    filtered_data = data[data['country'].isin(countries)]
    y_column = data.columns[2]
    fig = px.line(filtered_data,
                  x='year',
                  y=y_column,
                  color='country',
                  markers=True,
                  title=title,
                  labels={y_column: y_label},
                  template='plotly_white')

    fig.show()


def create_time_series_plot_2(data, country1, country2):
    # Filter data for the two countries
    filtered_data = data[data['country.name'].isin([country1, country2])]

    # Create the plot
    fig = px.line(filtered_data,
                  x='year',
                  y='value',
                  color='country.name',
                  labels={'country.name': 'Country', 'value': 'Poverty Rate'},
                  title=f'Time Series of Poverty Rate in {country1} and {country2}')

    
    fig.show()


def preprocess_inflation_data(file_path):
    """
    Load, melt, and preprocess inflation data from a CSV file.

    :param file_path: Path to the CSV file containing inflation data.
    :return: A DataFrame with melted and filtered inflation data.
    """
    # Load the data
    inflation_data = pd.read_csv(file_path, header=2)

    # Melt the data
    melted_data = inflation_data.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='Year',
        value_name='Inflation Rate'
    )

    # Convert 'Year' to numerical values and drop NaNs
    melted_data['Year'] = pd.to_numeric(melted_data['Year'], errors='coerce')
    melted_data = melted_data.dropna(subset=['Year'])
    melted_data['Year'] = melted_data['Year'].astype(int)

    # Filter out rows where 'Inflation Rate' is NaN
    filtered_data = melted_data[melted_data['Inflation Rate'].notna()]

    return filtered_data


def create_inflation_chart(data, regions):
    """
    Create a line chart showing inflation rates for specified regions.

    :param data: DataFrame containing the inflation data.
    :param regions: List of regions to include in the chart.
    """
    # Filter data for specified regions and where 'Inflation Rate' is not NaN
    filtered_data = data[data['Country Name'].isin(regions) & data['Inflation Rate'].notna()]

    # Create a line chart
    fig = px.line(
        filtered_data,
        x='Year',
        y='Inflation Rate',
        color='Country Name',
        title='Inflation Rates Over Time',
        labels={'Inflation Rate': 'Annual Inflation Rate (%)'}
    )

    # Show the plot
    fig.show()


def get_inflation_rates(data, country_name, start_year, end_year):
    """
    Return inflation rates for a specific country and time period.

    :param data: DataFrame containing inflation data.
    :param country_name: The name of the country.
    :param start_year: The starting year of the period.
    :param end_year: The ending year of the period.
    :return: DataFrame with years and corresponding inflation rates for the specified country and time period.
    """
    # Filter data for the specified country and time period
    country_data = data[
        (data['Country Name'] == country_name) & (data['Year'] >= start_year) & (data['Year'] <= end_year)]

    # Return the relevant columns
    return country_data[['Year', 'Inflation Rate']]


def adjust_poverty_line(df, initial_poverty_line, start_year, project_next_year=True):
    df = df.copy()
    df.sort_values(by='Year', inplace=True)
    df['Poverty Line'] = initial_poverty_line

    for i in range(1, len(df)):
        if df.iloc[i - 1]['Year'] >= start_year:
            previous_year_inflation = df.iloc[i - 1]['Inflation Rate'] / 100
            df.iloc[i, df.columns.get_loc('Poverty Line')] = df.iloc[i - 1]['Poverty Line'] * (
                    1 + previous_year_inflation)

    if project_next_year:
        last_year_inflation = df.iloc[-1]['Inflation Rate'] / 100
        next_year_poverty_line = df.iloc[-1]['Poverty Line'] * (1 + last_year_inflation)
        next_year = int(df.iloc[-1]['Year']) + 1  # Explicitly convert to int
        new_row = pd.DataFrame(
            {'Year': [next_year], 'Inflation Rate': [None], 'Poverty Line': [next_year_poverty_line]})
        df = pd.concat([df, new_row], ignore_index=True)

    df['Year'] = df['Year'].astype(int)  # Convert 'Year' to int
    return df[df['Year'] >= start_year]


def preprocess_mdm_data(file_path):
    """
    Load, preprocess, and rename columns of MDM data from an Excel file.

    :param file_path: Path to the Excel file containing MDM data.
    :return: A preprocessed DataFrame with renamed columns.
    """
    # Load the data from Excel, skipping the footer
    mdm_data = pd.read_excel(file_path, header=2, skipfooter=2)

    # Rename columns and replace country names
    mdm_data = mdm_data.rename(
        columns={'Unnamed: 15': 'MD_poverty_rate', 'Reporting year': 'year', 'Economy': 'country'})
    mdm_data['country'] = mdm_data['country'].replace({
        "Russian Federation": "Russia",
        "Congo, Dem. Rep.": "Democratic Republic of Congo",
        "Czech Republic": "Czechia",
        "Turkiye": "Turkey",
        "Egypt, Arab Rep.": "Egypt"
    })

    return mdm_data

def metrics_comparision(data, region, metrics):
    """
    Create a line chart showing average values of various metrics over time for a specified region.

    :param data: DataFrame containing the data.
    :param region: The region for which to create the chart.
    :param metrics: List of metrics to include in the chart.
    """
    # Group by 'Region' and 'year', then calculate the mean
    average_by_region_and_year = data.groupby(['Region', 'year'])[metrics].median().reset_index()

    # Filter data for the specified region
    filtered_data = average_by_region_and_year[average_by_region_and_year['Region'] == region]

    min_year = filtered_data['year'].min()
    max_year = filtered_data['year'].max()

    fig = px.line(filtered_data,
                  x='year',
                  y=metrics,
                  title=f'Average Metrics for {region} from {min_year} to {max_year}')

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Values',
        legend_title='Metrics'
    )

    # Show the plot
    fig.show()
