# OWCOVIDData Widget Documentation

The OWCOVIDData widget is designed to fetch COVID-19 data from a public API for various countries. Users can specify the date range and frequency of the data to obtain. The widget processes this data and provides it as an Orange `Timeseries` object for further analysis.

<figure>
  <img src="images/sankarsh-widgets/apiData/api4.png" alt="api" width="200"/>
  <figcaption>Widget Interface</figcaption>
</figure>


## Parameters

### Date Range
- **From**: Start date for the data retrieval. Users can select the start date using a date picker. Min date - 22/1/2020
- **To**: End date for the data retrieval. Users can select the end date using a date picker. Max date - 9/3/2023

### Country
- **Country**: Dropdown menu to select the country for which the COVID-19 data will be fetched. The list includes various countries with their respective codes used by the API.

### Frequency
- **Frequency**: Dropdown menu to choose the data aggregation frequency. Options include:
  - **Daily**: Data is aggregated daily.
  - **Weekly**: Data is aggregated weekly.
  - **Monthly**: Data is aggregated monthly.
  - **Quarterly**: Data is aggregated quarterly.

## Inputs
None

## Outputs
- **Time Series**: A `Timeseries` object containing the COVID-19 data.

![](../images/sankarsh-widgets/apiData/api3.png)