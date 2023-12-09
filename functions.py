"""

Author: Abdulaziz Alkhonin (asa15)

This module provides a suite of functions for processing and visualizing economic and poverty-related data.
It includes capabilities to filter and aggregate data, adjust poverty lines based on inflation,
create various types of plots and charts for data analysis, and preprocess data from different sources.


"""

import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame


def filter_and_aggregate(df: pd.DataFrame, year: int, column: str,
                         ppp_version: int = 2017,
                         reporting_level: str = "national") -> pd.Series:
    """
    Filters a DataFrame for a given year and optionally by PPP version and reporting level,
    then aggregates the data by country and calculates the mean of the specified column.

    :param df: Input DataFrame with columns 'year', 'country', and others.
    :param year: Year to filter the data.
    :param column: Column name for aggregation.
    :param ppp_version: PPP version to filter by, defaults to None.
    :param reporting_level: Reporting level to filter by, defaults to None.
    :return: Series with mean values of the specified column, aggregated by country.
    :raises ValueError: If the specified column is not in the DataFrame.


        >>> import pandas as pd_test
        >>> data = {'year': [2019, 2019, 2020],
        ...         'country': ['A', 'B', 'A'],
        ...         'value': [1, 2, 3],
        ...         'ppp_version': ['v1', 'v1', 'v2']}
        >>> df_test = pd_test.DataFrame(data)
        >>> filter_and_aggregate(df_test, 2019, 'value')
        country
        A    3.0
        B    2.0
        Name: value, dtype: float64
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Filter DataFrame by year
    filtered_df = df[df['year'] == year]

    # Apply additional filters for PPP version and reporting level, if columns exist
    if 'ppp_version' in df.columns:
        filtered_df = filtered_df[filtered_df['ppp_version'] == ppp_version]

    if 'reporting_level' in df.columns:
        filtered_df = filtered_df[filtered_df['reporting_level'] == reporting_level]

    # Aggregate data by country and calculate mean of the specified column
    most_recent_year = df[df.groupby('country')['year'].transform(max) == df['year']]
    result_year = filtered_df.groupby('country')[column].mean()
    result_most_recent_year = most_recent_year.groupby('country')[column].mean()
    result_combined = result_year.combine_first(result_most_recent_year)
    return result_combined


def replace_country_names(world: gpd.GeoDataFrame,
                          replacements: dict = None) -> gpd.GeoDataFrame:
    """
    Replaces specific country names in the input GeoDataFrame based on a provided mapping. 
    If no mapping is provided, a default set of replacements is used.

    :param world: gpd.GeoDataFrame - Input GeoDataFrame containing country shapes.
    :param replacements: dict, optional - Dictionary of country name replacements, 
                         defaults to None. Format: {'Original Name': 'New Name'}
    :return: gpd.GeoDataFrame - A GeoDataFrame with replaced country names.

    :Example:

    >>> world_test = gpd.GeoDataFrame(pd.DataFrame({'COUNTRY': ['Russian Federation', 'Congo DRC', 'Czech Republic']}))
    >>> replaced_world = replace_country_names(world_test)
    >>> 'Russia' in replaced_world['COUNTRY'].values and 'Democratic Republic of Congo' in replaced_world['COUNTRY'].values
    True
    """
    default_replacements = {
        "Russian Federation": "Russia",
        "Congo DRC": "Democratic Republic of Congo",
        "Czech Republic": "Czechia",
        "Turkiye": "Turkey"
    }

    # Use the provided replacements if available, otherwise use the default
    replacements = replacements if replacements is not None else default_replacements

    world['COUNTRY'] = world['COUNTRY'].replace(replacements)
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
    Preprocesses the input DataFrame for time-series analysis by filtering based on PPP version,
    selecting specific columns, grouping data by 'year' and 'country', and calculating the mean
    of the specified column.

    :param df: pd.DataFrame - Input DataFrame containing columns 'year', 'country', 'ppp_version', 'reporting_level',
    and the specified column.

    :param column: str - Name of the column to calculate the mean.
    :param ppp_version: int - PPP version for filtering the data.
    :return: pd.DataFrame - Processed DataFrame with columns 'year', 'country', and the specified column's mean values.

    :Example:

    >>> data = {'year': [2020, 2020, 2021],
    ...         'country': ['A', 'B', 'A'],
    ...         'ppp_version': [2011, 2017, 2011],
    ...         'reporting_level': ['national', 'national', 'national'],
    ...         'value': [1, 2, 3]}
    >>> df_test = pd.DataFrame(data)
    >>> time_series(df_test, 'value', 2011)
       year country  value
    0  2020       A    1.0
    1  2021       A    3.0
    """
    # Check if required columns are present
    required_columns = {'year', 'country', 'ppp_version', 'reporting_level', column}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter data based on 'ppp_version' and 'reporting_level'
    filtered_df = df[(df['ppp_version'] == ppp_version) & (df['reporting_level'] == "national")]

    # Select specific columns
    filtered_df = filtered_df[['year', 'country', column]]

    # Group data by 'year' and 'country' and calculate the mean of the specified column
    grouped_data = filtered_df.groupby(['year', 'country'])[column].mean().reset_index()

    return grouped_data


def plot_time_series(time_series_data: pd.DataFrame, selected_countries: list, title: str) -> None:
    """
    Creates an interactive line plot for selected countries using Plotly Express, plotting the last column of the
    DataFrame.

    :param time_series_data: pd.DataFrame - Processed DataFrame containing time-series data.
    :param selected_countries: list - List of country names to be included in the plot.
    :param title: str - Title for the plot.

    :Example:

    >>> empty_data = pd.DataFrame({})
    >>> plot_time_series(empty_data, ['A', 'B'], 'Empty Data Example')
    Traceback (most recent call last):
        ...
    ValueError: Input DataFrame is empty.

    >>> incomplete_data = pd.DataFrame({'value': [1, 2, 3]})
    >>> plot_time_series(incomplete_data, ['A', 'B'], 'Incomplete Data Example')
    Traceback (most recent call last):
        ...
    ValueError: DataFrame must contain 'year' and 'country' columns.

    >>> data = {'year': [2020, 2021], 'country': ['C', 'D'], 'value': [10, 20]}
    >>> df = pd.DataFrame(data)
    >>> plot_time_series(df, ['A', 'B'], 'No Data for Selected Countries')
    Traceback (most recent call last):
        ...
    ValueError: No data found for the selected countries.
    """
    # Check if DataFrame is empty
    if time_series_data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Ensure the DataFrame contains 'year' and 'country' columns
    if not {'year', 'country'}.issubset(time_series_data.columns):
        raise ValueError("DataFrame must contain 'year' and 'country' columns.")

    # Automatically select the last column for plotting
    column_to_plot = time_series_data.columns[-1]

    # Filter the DataFrame for the selected countries
    filtered_data = time_series_data[time_series_data['country'].isin(selected_countries)]

    # Check if the filtered DataFrame is empty (no countries matched)
    if filtered_data.empty:
        raise ValueError("No data found for the selected countries.")

    # Creating the plot
    fig = px.line(filtered_data,
                  x='year',
                  y=column_to_plot,
                  color='country',
                  markers=True,
                  title=title,
                  labels={column_to_plot: '% of Population'})

    # Display the plot
    fig.show()


def create_2slice_pie_chart(df: pd.DataFrame, year: int, country: str, poverty_threshold: float, title: str) -> None:
    """
    Creates a pie chart for a specified poverty line threshold with error handling.

    :param df: pd.DataFrame - DataFrame containing the data.
    :param year: int - The year for which the data is required.
    :param country: str - The country for which the data is required.
    :param poverty_threshold: float - The threshold value for poverty.
    :param title: str - Title for the pie chart.

    Error Handling Example:

    >>> data = {'year': [2020], 'country': ['Country A'], 'ppp_version': [2017],
    ...         'headcount_ratio_international_povline': [25]}
    >>> df_test = pd.DataFrame(data)
    >>> create_2slice_pie_chart(df_test, 2020, 'Country A', 5, 'Invalid Threshold Example')
    Traceback (most recent call last):
        ...
    ValueError: Invalid poverty line provided: 5. Valid options are 2.15, 10, 20, 30.
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

    # Check if the filtered DataFrame is empty (no data found)
    if filtered_df.empty:
        raise ValueError(f"No data found for country '{country}' in year {year}.")

    # Extract the poverty rate
    poverty_rate = filtered_df[column_name].iloc[0]

    # Create labels and sizes for the pie chart
    labels = [f'Less than ${poverty_threshold}', f'More than ${poverty_threshold}']
    sizes = [poverty_rate, 100 - poverty_rate]

    # Create and display the pie chart
    fig = px.pie(names=labels, values=sizes, title=title)
    fig.show()


def create_detailed_poverty_pie_chart(df: pd.DataFrame, year: int, country: str, title: str) -> None:
    """
    Creates a detailed segmented pie chart representing different poverty thresholds for a specified country and year.

    :param df: pd.DataFrame - DataFrame containing the data.
    :param year: int - The year for which the data is required.
    :param country: str - The country for which the data is required.
    :param title: str - Title for the pie chart.

    Error Handling Example:

    >>> data = {'year': [2017, 2017], 'country': ['Country A', 'Country B'],
    ...         'ppp_version': [2017, 2017],
    ...         'headcount_ratio_international_povline': [10, 20],
    ...         'headcount_ratio_1000': [30, 40],
    ...         'headcount_ratio_3000': [60, 70]}
    >>> df_test = pd.DataFrame(data)
    >>> create_detailed_poverty_pie_chart(df_test, 2018, 'Country A', 'Missing Data Example')
    Traceback (most recent call last):
        ...
    ValueError: No data found for country 'Country A' in year 2018.
    """
    # Filter the DataFrame
    filtered_df = df[(df['year'] == year) & (df['country'] == country) & (df['ppp_version'] == 2017)]

    # Check if the filtered DataFrame is empty (no data found)
    if filtered_df.empty:
        raise ValueError(f"No data found for country '{country}' in year {year}.")

    # Extract values for each segment
    try:
        poverty_rate_215 = filtered_df['headcount_ratio_international_povline'].iloc[0]
        poverty_rate_1000 = filtered_df['headcount_ratio_1000'].iloc[0]
        poverty_rate_3000 = filtered_df['headcount_ratio_3000'].iloc[0]
    except IndexError:
        raise ValueError(f"Poverty data is incomplete for country '{country}' in year {year}.")

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


def load_poverty_data(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the poverty data from a CSV file. This includes calculating the
    percentage of the population under the poverty line.

    :param file_path: str - Path to the CSV file containing poverty data.
    :return: pd.DataFrame - Preprocessed poverty data.

    """

    poverty_data = pd.read_csv(file_path)
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
    return poverty_data


def load_gdp_data(file_path: str) -> pd.DataFrame:
    """
    Loads and reshapes the GDP data from an Excel file. This includes melting the DataFrame
    to have a long-form representation suitable for merging with poverty data.

    :param file_path: str - Path to the Excel file containing GDP data.
    :return: pd.DataFrame - Reshaped GDP data.


    """

    # Load GDP data and reshape it
    gdp_data = pd.read_excel(file_path, header=1, nrows=21)
    gdp_data = gdp_data.rename(columns={
        'Unnamed: 0': 'Year',
        'South and Southeast Asia)': 'South and Southeast Asia',
        'Latin America and Carribean': 'Latin America and Caribbean',
        'East Asia ': 'East Asia'
    })
    melted_gdp = gdp_data.melt(id_vars=['Year'], var_name='Entity', value_name='GDP_per_capita')

    return melted_gdp


def merge_poverty_gdp_data(poverty_file_path: str, gdp_file_path: str) -> tuple[DataFrame, DataFrame]:
    """
    Loads, preprocesses, and merges poverty and GDP data from given file paths by calling
    `load_poverty_data` and `load_gdp_data`.

    :param poverty_file_path: str - Path to the CSV file containing poverty data.
    :param gdp_file_path: str - Path to the Excel file containing GDP data.
    :return: pd.DataFrame - Merged DataFrame with preprocessed poverty and GDP data.


    """

    poverty_data = load_poverty_data(poverty_file_path)
    gdp_data = load_gdp_data(gdp_file_path)

    # Merge the poverty and GDP data on 'Year' and 'Entity'
    merged_data = pd.merge(gdp_data, poverty_data, on=['Year', 'Entity'], how='inner')

    # Clean and convert GDP data to numeric
    merged_data['GDP_per_capita'] = (
        merged_data['GDP_per_capita']
        .str.replace(r'[^\d.]', '', regex=True)
        .replace('', float('nan'))
        .pipe(pd.to_numeric, errors='coerce')
    )
    return merged_data, poverty_data


def create_area_chart(data: pd.DataFrame, entity: str) -> None:
    """
    Creates an area chart for a given entity showing the percentage of the population living in extreme poverty.

    :param data: pd.DataFrame - DataFrame containing the poverty data.
    :param entity: str - The entity for which the area chart is to be created.

    :Example:

    >>> test_data = pd.DataFrame({
    ...     'Entity': ['Entity A', 'Entity A', 'Entity B'],
    ...     'Year': [2000, 2001, 2000],
    ...     'population_under_poverty': [25, 30, 40]
    ... })

    Handling no data available for the specified entity:
    >>> create_area_chart(test_data, 'Entity C')
    Traceback (most recent call last):
        ...
    ValueError: No data available for entity 'Entity C'.

    Handling missing required columns:
    >>> incomplete_data = pd.DataFrame({'Year': [2000, 2001], 'Entity': ['Entity A', 'Entity A']})
    >>> create_area_chart(incomplete_data, 'Entity A')
    Traceback (most recent call last):
        ...
    ValueError: Missing one or more required columns: 'Year', 'population_under_poverty', 'Entity'.
    """
    # Filter data for the specified entity
    entity_data = data[data['Entity'] == entity]

    # Check if there is data for the specified entity
    if entity_data.empty:
        raise ValueError(f"No data available for entity '{entity}'.")

    # Ensure required columns are present
    if not {'Year', 'population_under_poverty', 'Entity'}.issubset(entity_data.columns):
        raise ValueError("Missing one or more required columns: 'Year', 'population_under_poverty', 'Entity'.")

    # Create an area chart
    fig = px.area(entity_data, x='Year', y='population_under_poverty',
                  labels={'population_under_poverty': 'Percentage of Population'},
                  title=f'Average Share of Population Living in Extreme Poverty - {entity}')

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Year', yaxis_title='Percentage of Population',
        plot_bgcolor='white', yaxis=dict(gridcolor='lightgrey', range=[0, 100]),
        xaxis=dict(gridcolor='lightgrey'))

    # Display the chart
    fig.show()


def create_comparison_chart(data: pd.DataFrame, regions: list, standout_region: str, show_years: bool = False) -> None:
    """
    Creates a comparison chart showing the relationship between GDP per capita and poverty rate for specified regions.

    :param data: pd.DataFrame - DataFrame containing poverty and GDP data.
    :param regions: list - List of regions to include in the chart.
    :param standout_region: str - A specific region to highlight in the chart.
    :param show_years: bool - Flag to show year labels on the standout region's data points (default is False).

    """

    fig = go.Figure()
    for region in regions:
        # Filter data for each region and exclude rows with missing GDP data
        filtered = data[(data['Entity'] == region) & (~data['GDP_per_capita'].isna())]

        # Highlight the standout region with thicker lines and different color
        line_width = 3 if region == standout_region else 1
        opacity = 1 if region == standout_region else 0.5
        is_standout = region == standout_region
        text_mode = '+text' if is_standout and show_years else ''

        # Create scatter trace for each region
        trace = go.Scatter(
            x=filtered['GDP_per_capita'], y=filtered['population_under_poverty'],
            mode='lines+markers' + text_mode, name=region,
            line=dict(width=line_width, color='black' if is_standout else None),
            opacity=opacity,
            text=filtered['Year'] if is_standout and show_years else None,
            textposition="top right" if is_standout and show_years else None
        )
        fig.add_trace(trace)

    # Update chart layout
    fig.update_layout(
        title='GDP per Capita vs. Poverty Rate Over Time',
        xaxis_title='GDP per Capita', xaxis_type='log', yaxis_title='Poverty Rate (%)',
        hovermode='x unified', showlegend=True)

    # Add annotation for standout region
    fig.add_annotation(
        text=standout_region,
        xref="paper", x=0.8, yref="paper", y=0.8,
        showarrow=False, font=dict(size=20),
        align="right", xanchor="right", yanchor="top"
    )

    # Set custom ticks for x-axis
    fig.update_xaxes(title_text='GDP per Capita', tickvals=[1000, 5000, 10000, 20000, 50000])

    # Display the chart
    fig.show()


def create_time_series_plot(data: pd.DataFrame, countries: list, title: str, y_label: str) -> None:
    """
    Creates an interactive line plot for selected countries.

    :param data: pd.DataFrame - DataFrame containing the data.
    :param countries: list - List of countries to include in the plot.
    :param title: str - Title of the plot.
    :param y_label: str - Label for the y-axis.

    """
    # Filter data for the specified countries
    filtered_data = data[data['country'].isin(countries)]

    # Automatically select the column for y-axis (assuming it's the third column)
    y_column = data.columns[-1]

    # Create and configure the line plot
    fig = px.line(filtered_data,
                  x='year',  # x-axis represents the years
                  y=y_column,  # y-axis represents the selected data column
                  color='country',  # Color lines by country
                  markers=True,  # Include markers in the plot
                  title=title,  # Set the title of the plot
                  labels={y_column: y_label},  # Label for the y-axis
                  template='plotly_white')  # Use Plotly's white template for the plot

    # Display the plot
    fig.show()


def historical_poverty_data(file_path):
    """
    Read a CSV file and rename a specified column.

    :param file_path: Path to the CSV file.
    :return: DataFrame with the renamed column.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Rename the specified column
    df.rename(columns={'country.name': 'country'}, inplace=True)

    return df


def preprocess_inflation_data(file_path: str) -> pd.DataFrame:
    """
    Load, melt, and preprocess inflation data from a CSV file.

    :param file_path: str - Path to the CSV file containing inflation data.
    :return: pd.DataFrame - A DataFrame with melted and filtered inflation data.

    """
    try:
        # Load the data
        inflation_data = pd.read_csv(file_path, header=2)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    # Melt the data
    melted_data = inflation_data.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='Year',
        value_name='Inflation Rate'
    )

    # Convert 'Year' to numerical values and filter NaNs efficiently
    melted_data['Year'] = pd.to_numeric(melted_data['Year'], errors='coerce')
    melted_data.dropna(subset=['Year', 'Inflation Rate'], inplace=True)
    melted_data['Year'] = melted_data['Year'].astype(int)
    melted_data.rename(columns={'Country Name': 'country', 'Year': 'year'}, inplace=True)

    return melted_data


def get_inflation_rates(data: pd.DataFrame, country_name: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Return inflation rates for a specific country and time period.

    :param data: pd.DataFrame - DataFrame containing inflation data.
    :param country_name: str - The name of the
    country.
    :param start_year: int - The starting year of the period.
    :param end_year: int - The ending year of the
    period. :return: pd.DataFrame - DataFrame with years and corresponding inflation rates for the specified country
    and time period.

    :Example:

    >>> test_data = pd.DataFrame({
    ...     'country': ['Country A', 'Country A', 'Country B'],
    ...     'year': [2000, 2001, 2000],
    ...     'Inflation Rate': [2.5, 3.0, 1.5]
    ... })
    >>> get_inflation_rates(test_data, 'Country A', 2000, 2001)
       year  Inflation Rate
    0  2000             2.5
    1  2001             3.0
    """
    # Filter data for the specified country and time period
    country_data = data[
        (data['country'] == country_name) & (data['year'] >= start_year) & (data['year'] <= end_year)]

    # Check if the filtered data is empty
    if country_data.empty:
        raise ValueError(f"No data available for '{country_name}' in the specified time period.")

    # Return the relevant columns, sorted by year
    return country_data[['year', 'Inflation Rate']].sort_values(by='year')


def adjust_poverty_line(df, initial_poverty_line, start_year, project_next_year=True):
    """
        Adjusts the poverty line in a DataFrame based on annual inflation rates.

        :param df: DataFrame containing 'year' and 'Inflation Rate'.
        :param initial_poverty_line: Initial value of the poverty line to adjust from.
        :param start_year: Year from which to start adjusting the poverty line.
        :param project_next_year: Whether to project the poverty line for the year following the last year in the data.
        :return: DataFrame with adjusted poverty line values from the start year onwards.

        :Example:

    >>> import pandas as pd_test
    >>> from io import StringIO
    >>> mock_data = '''year,Inflation Rate\\n2020,2\\n2021,10\\n2022,4'''
    >>> mock_file = StringIO(mock_data)
    >>> df_test = pd_test.read_csv(mock_file)
    >>> adjusted_df = adjust_poverty_line(df_test, 10, 2020, project_next_year=True)
    >>> adjusted_df[['year', 'Inflation Rate', 'Poverty Line']]
       year Inflation Rate  Poverty Line
    0  2020              2       10.0000
    1  2021             10       10.2000
    2  2022              4       11.2200
    3  2023           None       11.6688
    """

    # Create a copy of the DataFrame and sort it by year
    df = df.copy()
    df.sort_values(by='year', inplace=True)

    # Initialize the poverty line with the initial value
    df['Poverty Line'] = initial_poverty_line

    # Iterate through each row to adjust the poverty line based on previous year's inflation
    for i in range(1, len(df)):
        if df.iloc[i - 1]['year'] >= start_year:
            previous_year_inflation = df.iloc[i - 1]['Inflation Rate'] / 100
            df.iloc[i, df.columns.get_loc('Poverty Line')] = df.iloc[i - 1]['Poverty Line'] * (
                    1 + previous_year_inflation)

    # Project the poverty line to the next year after the last year in the data, if specified
    if project_next_year:
        last_year_inflation = df.iloc[-1]['Inflation Rate'] / 100
        next_year_poverty_line = df.iloc[-1]['Poverty Line'] * (1 + last_year_inflation)
        next_year = int(df.iloc[-1]['year']) + 1
        new_row = pd.DataFrame(
            {'year': [next_year], 'Inflation Rate': [None], 'Poverty Line': [next_year_poverty_line]})
        df = pd.concat([df, new_row], ignore_index=True)

    # Ensure the 'year' column is of integer type
    df['year'] = df['year'].astype(int)

    # Return the DataFrame filtered from the start year onwards
    return df[df['year'] >= start_year]


def preprocess_mdm_data(file_path: str) -> pd.DataFrame:
    """
    Load, preprocess, and rename columns of MDM data from an Excel file.

    :param file_path: str - Path to the Excel file containing MDM data.
    :return: pd.DataFrame - A preprocessed DataFrame with renamed columns.

    """
    # Assume file_path is a DataFrame for the purpose of the doctest
    mdm_data = file_path if isinstance(file_path, pd.DataFrame) else pd.read_excel(file_path, header=2, skipfooter=2)

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


def metrics_comparision(data: pd.DataFrame, region: str, metrics: list) -> None:
    """
    Create a line chart showing average values of various metrics over time for a specified region.

    :param data: pd.DataFrame - DataFrame containing the data.
    :param region: str - The region for which to create the chart.
    :param metrics: list - List of metrics to include in the chart.

    """
    # Group by 'Region' and 'year', then calculate the median for specified metrics
    average_by_region_and_year = data.groupby(['Region', 'year'])[metrics].median().reset_index()

    # Filter data for the specified region
    filtered_data = average_by_region_and_year[average_by_region_and_year['Region'] == region]
    # Determine the range of years for the title
    min_year = filtered_data['year'].min()
    max_year = filtered_data['year'].max()

    # Create a line chart with the filtered data
    fig = px.line(filtered_data,
                  x='year',
                  y=metrics,
                  title=f'Average Metrics for {region} from {min_year} to {max_year}')

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Values',
        legend_title='Metrics'
    )

    # Show the plot
    fig.show()
