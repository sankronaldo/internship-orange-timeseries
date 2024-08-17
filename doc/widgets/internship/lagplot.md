# Lag Plot

## Overview
The **Lag Plot** widget visualizes lag plots of time series data to help assess the relationship between the current value of the series and its lagged values. This is useful for identifying patterns or dependencies in time series data.

![Lag Plot](../images/sankarsh-widgets/lag_plot/lagplot1.png)

## Parameters
- **max_lags**: The maximum number of lags for which the lag plots will be generated. Default is 12.
- **variable_index**: The selected target variable for which lag plots are generated.

## Inputs
- **Time series data**: The input time series data (Orange.data.Table).

## Outputs
- **Selected Data**: The corresponding lag plot

