# Periodogram

## Overview
The **Periodogram** widget computes and displays the periodogram (power spectrum) of a time series. It allows users to visualize the raw and smoothed periodogram and provides options to configure the view, including log scaling and smoothing window adjustments. This widget is useful for frequency domain analysis of time series data.

![Periodogram](../images/sankarsh-widgets/peridodgram/periodogram1.png)

## Parameters
- **target_variable**: The selected target variable for which the periodogram is calculated.
- **show_raw**: Option to display the raw periodogram. Enabled by default.
- **show_smoothed**: Option to display the smoothed periodogram using the Savitzky-Golay filter. Enabled by default.
- **log_scale**: Option to use a logarithmic scale for the y-axis of the periodogram. Disabled by default.
- **smoothing_window**: The length of the smoothing window for the Savitzky-Golay filter, with a minimum of 5.

## Inputs
- **Time series**: The input time series data (Orange.data.Table).

## Outputs
- **Periodogram Data**: A plot showing the raw and smoothed periodogram for the selected target variable.
