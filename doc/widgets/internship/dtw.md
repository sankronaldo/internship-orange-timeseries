# Dynamic Time Warping Widget

## Overview
The **Dynamic Time Warping** widget computes and visualizes the similarity between two time series using the Dynamic Time Warping (DTW) algorithm. It calculates the DTW distance and visualizes the alignment between the time series, including the warping path and scaled series comparisons.

![](../images/sankarsh-widgets/dtw/dtw.png)


## Inputs
- **data_a**: A table of time series data for Series A (Orange.data.Table). The table should contain time series data that will be compared.
- **data_b**: A table of time series data for Series B (Orange.data.Table). The table should contain time series data that will be compared.

## Outputs
- **dtw_score**: The DTW similarity score (float). This score represents the similarity between the two time series, where a lower score indicates higher similarity.
