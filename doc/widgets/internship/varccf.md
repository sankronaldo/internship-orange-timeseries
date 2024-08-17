# Multivariate CCF Widget

## Overview
The **Multivariate CCF** widget computes and visualizes the cross-correlation function (CCF) between the residuals of a vector autoregressive (VAR) model. This widget is useful for examining the relationships between multiple variables over different time lags.

![](../images/sankarsh-widgets/varccf/varccf.png)

## Parameters
- **max_lags**: The maximum number of lags to include in the CCF plot. This parameter controls how many lagged correlations are computed and visualized.

## Inputs
- **var_residuals**: A table of residuals from a VAR model (Orange.data.Table). This input should contain the residuals for multiple variables.

## Outputs
- **Plot**: The widget visualizes CCF plots for the residuals of each variable pair.

