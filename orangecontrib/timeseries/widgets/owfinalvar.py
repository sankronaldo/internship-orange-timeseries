import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table, Domain, TimeVariable, ContinuousVariable, StringVariable
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QFont
import traceback

class VARWidget(widget.OWWidget):
    name = "Vector Autoregression"
    description = "VAR modeling and forecasting"
    icon = "icons/ow_var.svg"
    priority = 10

    class Inputs:
        data = widget.Input("Time series data", Table)

    class Outputs:
        forecast = widget.Output("Forecast", Table)
        residuals = widget.Output("Residuals", Table)
        model_summary = widget.Output("Model Summary", str)

    # Widget parameters
    max_lags = settings.Setting(8)
    lag_order = settings.Setting(1)
    forecast_steps = settings.Setting(10)
    apply_log = settings.Setting(False)
    diff_order = settings.Setting(0)
    seasonal_diff = settings.Setting(False)
    seasonal_periods = settings.Setting(12)

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "VAR Parameters")
        gui.spin(box, self, "max_lags", 1, 20, label="Max Lags")
        gui.spin(box, self, "lag_order", 1, 20, label="Lag Order")
        gui.spin(box, self, "forecast_steps", 1, 100, label="Forecast Steps")
        gui.checkBox(box, self, "apply_log", "Apply Log Transformation")

        diff_box = gui.widgetBox(box, "Differencing")
        gui.comboBox(diff_box, self, "diff_order", label="Differencing Order",
                     items=["No Differencing", "1st Order", "2nd Order"])
        gui.checkBox(diff_box, self, "seasonal_diff", "Apply Seasonal Differencing")
        gui.spin(diff_box, self, "seasonal_periods", 2, 365, label="Seasonal Periods")

        self.apply_button = gui.button(self.controlArea, self, "Apply", callback=self.apply)

        # Text area for displaying model information
        self.info_box = gui.widgetBox(self.mainArea, "Model Information")
        self.text_output = QTextEdit(readOnly=True)
        self.text_output.setFont(QFont("Courier"))
        self.info_box.layout().addWidget(self.text_output)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if self.data is not None:
            self.apply()

    def apply(self):
        if self.data is None:
            return

        try:
            # Convert Orange Table to pandas DataFrame
            df = pd.DataFrame(self.data.X, columns=[var.name for var in self.data.domain.attributes])

            transformation_text = ""

            # Apply log transformation if selected
            if self.apply_log:
                df = np.log(df)
                transformation_text += "Log transformation applied.\n"

            # Apply differencing if selected
            if self.diff_order > 0:
                df = df.diff(periods=self.diff_order).dropna()
                transformation_text += f"{self.diff_order}{'st' if self.diff_order == 1 else 'nd'} order differencing applied.\n"

            # Apply seasonal differencing if selected
            if self.seasonal_diff:
                df = df.diff(periods=self.seasonal_periods).dropna()
                transformation_text += f"Seasonal differencing applied (period: {self.seasonal_periods}).\n"

            # Fit the VAR model
            model = VAR(df)
            lag_order_selection = model.select_order(maxlags=self.max_lags)
            optimal_lag_order = lag_order_selection.aic
            results = model.fit(self.lag_order)  # Use user-specified lag order

            # Generate forecast
            forecast = results.forecast(df.values[-self.lag_order:], steps=self.forecast_steps)

            # Get residuals
            residuals = results.resid

            # Prepare output
            output_text = transformation_text + f"Optimal lag order (AIC): {optimal_lag_order}\n"
            output_text += f"User-specified lag order: {self.lag_order}\n\n"
            output_text += str(results.summary()) + "\n\n"

            # Add VAR equations
            output_text += "VAR Equations:\n"
            for i, equation in enumerate(results.params.index):
                eq = f"{equation} = "
                for coef in results.params.columns:
                    if coef != 'const':
                        eq += f"{results.params.loc[equation, coef]:.3f} * {coef} + "
                if 'const' in results.params.columns:
                    eq += f"{results.params.loc[equation, 'const']:.3f} (constant)"
                else:
                    eq = eq.rstrip(' + ')
                output_text += eq + "\n"

            # Add optimal lag information
            output_text += "\nOptimal Lag Information:\n"
            output_text += f"AIC: {lag_order_selection.aic}\n"
            output_text += f"BIC: {lag_order_selection.bic}\n"
            output_text += f"FPE: {lag_order_selection.fpe}\n"
            output_text += f"HQIC: {lag_order_selection.hqic}\n"

            # Add stationarity test results
            output_text += "\nStationarity Test Results:\n"
            for column in df.columns:
                result = adfuller(df[column].dropna())
                output_text += f"{column}:\n"
                output_text += f"ADF Statistic: {result[0]:.4f}\n"
                output_text += f"p-value: {result[1]:.4f}\n"
                output_text += f"Critical Values:\n"
                for key, value in result[4].items():
                    output_text += f"\t{key}: {value:.4f}\n"
                output_text += "\n"

            # Display output in the text area
            self.text_output.setPlainText(output_text)

            # Create output tables
            domain_forecast = Domain([ContinuousVariable(name) for name in df.columns])
            domain_residuals = Domain([ContinuousVariable(name) for name in df.columns])

            forecast_table = Table.from_numpy(domain_forecast, forecast)
            residuals_table = Table.from_numpy(domain_residuals, residuals)

            # Send outputs
            self.Outputs.forecast.send(forecast_table)
            self.Outputs.residuals.send(residuals_table)
            self.Outputs.model_summary.send(output_text)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.text_output.setPlainText(error_message)
            self.Outputs.forecast.send(None)
            self.Outputs.residuals.send(None)
            self.Outputs.model_summary.send(error_message)


if __name__ == "__main__":
    WidgetPreview(VARWidget).run(Table("iris"))









# import Orange
# from Orange.widgets import widget, gui, settings
# from Orange.widgets.utils.widgetpreview import WidgetPreview
# from Orange.data import Table, Domain, TimeVariable, ContinuousVariable, StringVariable
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.api import VAR
# from PyQt5.QtWidgets import QTextEdit
# from PyQt5.QtGui import QFont
# from statsmodels.tsa.stattools import adfuller
#
#
# class VARWidget(widget.OWWidget):
#     name = "Vector Autoregression"
#     description = "VAR modeling and forecasting"
#     icon = "icons/var.svg"
#     priority = 10
#
#     class Inputs:
#         data = widget.Input("Time series data", Table)
#
#     class Outputs:
#         forecast = widget.Output("Forecast", Table)
#         residuals = widget.Output("Residuals", Table)
#         model_summary = widget.Output("Model Summary", str)
#
#     # Widget parameters
#     max_lags = settings.Setting(8)
#     forecast_steps = settings.Setting(10)
#     apply_log = settings.Setting(False)
#     apply_diff = settings.Setting(False)
#
#     def __init__(self):
#         super().__init__()
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "VAR Parameters")
#         gui.spin(box, self, "max_lags", 1, 20, label="Max Lags")
#         gui.spin(box, self, "forecast_steps", 1, 100, label="Forecast Steps")
#         gui.checkBox(box, self, "apply_log", "Apply Log Transformation")
#         gui.checkBox(box, self, "apply_diff", "Apply First Difference")
#
#         self.apply_button = gui.button(self.controlArea, self, "Apply", callback=self.apply)
#
#         # Text area for displaying model information
#         self.info_box = gui.widgetBox(self.mainArea, "Model Information")
#         self.text_output = QTextEdit(readOnly=True)
#         self.text_output.setFont(QFont("Courier"))
#         self.info_box.layout().addWidget(self.text_output)
#
#     @Inputs.data
#     def set_data(self, data):
#         self.data = data
#         if self.data is not None:
#             self.apply()
#
#     def apply(self):
#         if self.data is None:
#             return
#
#         # Convert Orange Table to pandas DataFrame
#         df = pd.DataFrame(self.data.X, columns=[var.name for var in self.data.domain.attributes])
#
#         # Apply transformations if selected
#         if self.apply_log:
#             df = np.log(df)
#             transformation_text = "Log transformation applied.\n"
#         else:
#             transformation_text = ""
#
#         if self.apply_diff:
#             df = df.diff().dropna()
#             transformation_text += "First difference applied.\n"
#
#         # Fit the VAR model
#         model = VAR(df)
#         lag_order_selection = model.select_order(maxlags=self.max_lags)
#         optimal_lag_order = lag_order_selection.aic
#         results = model.fit(optimal_lag_order)
#
#         # Generate forecast
#         forecast = results.forecast(df.values[-optimal_lag_order:], steps=self.forecast_steps)
#
#         # Get residuals
#         residuals = results.resid
#
#         # Prepare output
#         output_text = transformation_text + f"Optimal lag order (AIC): {optimal_lag_order}\n\n"
#         output_text += str(results.summary()) + "\n\n"
#
#         # Add VAR equations
#         output_text += "VAR Equations:\n"
#         for i, equation in enumerate(results.params.index):
#             eq = f"{equation} = "
#             for coef in results.params.columns:
#                 if coef != 'const':
#                     eq += f"{results.params.loc[equation, coef]:.3f} * {coef} + "
#             if 'const' in results.params.columns:
#                 eq += f"{results.params.loc[equation, 'const']:.3f} (constant)"
#             else:
#                 eq = eq.rstrip(' + ')
#             output_text += eq + "\n"
#
#         # Add optimal lag calculation proof
#         output_text += "\nOptimal Lag Calculation Proof:\n"
#         output_text += "The optimal lag is selected using the Akaike Information Criterion (AIC).\n"
#         output_text += "AIC = 2k - 2ln(L)\n"
#         output_text += "where k is the number of parameters and L is the likelihood of the model.\n"
#         output_text += "The lag order with the lowest AIC value is chosen as optimal.\n\n"
#         output_text += "AIC value for the optimal lag:\n"
#         output_text += f"Lag {optimal_lag_order}: AIC = {lag_order_selection.aic}\n"
#
#         # Add stationarity test results
#         output_text += "\nStationarity Test Results:\n"
#         for column in df.columns:
#             result = adfuller(df[column].dropna())
#             output_text += f"{column}:\n"
#             output_text += f"ADF Statistic: {result[0]:.4f}\n"
#             output_text += f"p-value: {result[1]:.4f}\n"
#             output_text += f"Critical Values:\n"
#             for key, value in result[4].items():
#                 output_text += f"\t{key}: {value:.4f}\n"
#             output_text += "\n"
#
#         # Display output in the text area
#         self.text_output.setPlainText(output_text)
#
#         # Create output tables
#         domain_forecast = Domain([ContinuousVariable(name) for name in df.columns])
#         domain_residuals = Domain([ContinuousVariable(name) for name in df.columns])
#
#         forecast_table = Table.from_numpy(domain_forecast, forecast)
#         residuals_table = Table.from_numpy(domain_residuals, residuals)
#
#         # Send outputs
#         self.Outputs.forecast.send(forecast_table)
#         self.Outputs.residuals.send(residuals_table)
#         self.Outputs.model_summary.send(output_text)
#
#
# if __name__ == "__main__":
#     WidgetPreview(VARWidget).run(Table("iris"))
