# Weather Data Widget

The OWWeatherData widget allows users to download and visualize weather data from the Open-Meteo API for a specified city and date range. It supports both hourly and daily weather data, and the data is converted into an Orange `Timeseries` object for further analysis.

<figure>
  <img src="images/sankarsh-widgets/apiData/api2.png" alt="api" width="200"/>
  <figcaption>Widget Interface</figcaption>
</figure>

## Parameters
### Date Range
- **From**: The start date for the data retrieval. Users can select the start date using a date picker.
- **To**: The end date for the data retrieval. Users can select the end date using a date picker.

### City
- **City**: Dropdown menu to select the city for which the weather data will be fetched. The city is selected from a predefined list of cities with their coordinates.

### Use Hourly Data
- **Use Hourly Data**: Checkbox to select whether hourly weather data should be used instead of daily data.

## Inputs
None

## Outputs
- **Time Series**: A `Timeseries` object containing the weather data.

![](../images/sankarsh-widgets/apiData/api1.png)

