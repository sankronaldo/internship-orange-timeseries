import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table, Domain, TimeVariable, ContinuousVariable, StringVariable
import numpy as np
from pmdarima import auto_arima
from PyQt5.QtWidgets import QTextEdit, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QFont
import pandas as pd


class AutoARIMAWidget(widget.OWWidget):
    name = "Auto ARIMA"
    description = "Automatic ARIMA modeling and forecasting"
    icon = "icons/ow_autoarima.svg"
    priority = 10

    class Inputs:
        data = widget.Input("Time series data", Table)

    class Outputs:
        forecast = widget.Output("Forecast", Table)
        fitted_values = widget.Output("Fitted Values", Table)
        residuals = widget.Output("Residuals", Table)
        model_summary = widget.Output("Model Summary", str)
        model = widget.Output("Model", Table)

    # Widget parameters
    steps = settings.Setting(10)
    max_p = settings.Setting(5)
    max_d = settings.Setting(2)
    max_q = settings.Setting(5)
    seasonal = settings.Setting(True)
    max_P = settings.Setting(2)
    max_D = settings.Setting(1)
    max_Q = settings.Setting(2)
    m = settings.Setting(12)  # Seasonal period

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "ARIMA Parameters")
        gui.spin(box, self, "steps", 1, 100, label="Forecast Steps")
        gui.spin(box, self, "max_p", 0, 10, label="Max p")
        gui.spin(box, self, "max_d", 0, 10, label="Max d")
        gui.spin(box, self, "max_q", 0, 10, label="Max q")
        gui.checkBox(box, self, "seasonal", "Seasonal")
        gui.spin(box, self, "max_P", 0, 10, label="Max P")
        gui.spin(box, self, "max_D", 0, 10, label="Max D")
        gui.spin(box, self, "max_Q", 0, 10, label="Max Q")
        gui.spin(box, self, "m", 1, 365, label="Seasonal Period")

        self.apply_button = gui.button(self.controlArea, self, "Apply", callback=self.apply)

        # Table for displaying model information
        self.info_box = gui.widgetBox(self.mainArea, "Model Information")
        self.table_output = QTableWidget()
        self.info_box.layout().addWidget(self.table_output)

        # Text area for displaying model summary
        self.summary_box = gui.widgetBox(self.mainArea, "Model Summary")
        self.text_output = QTextEdit(readOnly=True)
        self.text_output.setFont(QFont("Courier"))
        self.summary_box.layout().addWidget(self.text_output)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if self.data is not None:
            self.apply()

    def apply(self):
        if self.data is None:
            return

        # Assume the last column is the target variable
        y = self.data.Y.ravel()

        # Fit auto_arima model
        model = auto_arima(y, start_p=0, start_q=0, max_p=self.max_p, max_d=self.max_d, max_q=self.max_q,
                           start_P=0, start_Q=0, max_P=self.max_P, max_D=self.max_D, max_Q=self.max_Q,
                           m=self.m, seasonal=self.seasonal, error_action='ignore', suppress_warnings=True,
                           stepwise=True)

        # Generate forecast
        forecast = model.predict(n_periods=self.steps)

        # Get fitted values and residuals
        fitted_values = model.predict_in_sample()
        residuals = y - fitted_values

        # Create output tables
        domain_forecast = Domain([ContinuousVariable("Forecast")])
        domain_fitted = Domain([ContinuousVariable("Fitted Values")])
        domain_residuals = Domain([ContinuousVariable("Residuals")])

        forecast_table = Table.from_numpy(domain_forecast, forecast.reshape(-1, 1))
        fitted_table = Table.from_numpy(domain_fitted, fitted_values.reshape(-1, 1))
        residuals_table = Table.from_numpy(domain_residuals, residuals.reshape(-1, 1))

        # Get model summary and parameters
        summary = model.summary()
        order = model.order
        seasonal_order = model.seasonal_order
        aic = model.aic()

        # Calculate AICc manually
        n = len(y)
        k = len(model.params())
        aicc_value = aic + (2 * k ** 2 + 2 * k) / (n - k - 1)

        me = np.mean(residuals)
        mse_value = np.mean((y - fitted_values) ** 2)
        mae_value = np.mean(np.abs(y - fitted_values))
        mape = np.mean(np.abs((y - fitted_values) / y)) * 100

        # Calculate MASE (Mean Absolute Scaled Error)
        diff = np.abs(np.diff(y))
        mae_naive = np.mean(diff)
        mase = np.mean(np.abs(residuals)) / mae_naive

        # Calculate ACF1 (Autocorrelation at lag 1)
        acf1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

        # Prepare data for the table
        data = [
            ("Model Order", f"ARIMA{order}"),
            ("Seasonal Order", f"{seasonal_order}" if self.seasonal else "N/A"),
            ("AIC", f"{aic:.4f}"),
            ("AICc", f"{aicc_value:.4f}"),
            ("ME", f"{me:.4f}"),
            ("MSE", f"{mse_value:.4f}"),
            ("MAE", f"{mae_value:.4f}"),
            ("MASE", f"{mase:.4f}"),
            ("MAPE", f"{mape:.4f}"),
            ("ACF1", f"{acf1:.4f}")
        ]

        # Set up the table
        self.table_output.setColumnCount(2)
        self.table_output.setRowCount(len(data))
        self.table_output.setHorizontalHeaderLabels(["Metric", "Value"])

        # Populate the table
        for i, (metric, value) in enumerate(data):
            self.table_output.setItem(i, 0, QTableWidgetItem(metric))
            self.table_output.setItem(i, 1, QTableWidgetItem(value))

        # Adjust table appearance
        self.table_output.resizeColumnsToContents()
        self.table_output.resizeRowsToContents()

        # Display model summary in the text area
        self.text_output.setPlainText(str(summary))

        # Prepare model output as Orange.data.Table
        param_names = model.arima_res_.param_names
        param_values = model.params()
        model_data = pd.DataFrame({
            'Parameter': param_names,
            'Value': param_values
        })
        model_domain = Domain([ContinuousVariable("Value")], metas=[StringVariable("Parameter")])
        model_table = Table.from_numpy(
            model_domain,
            X=model_data['Value'].values.reshape(-1, 1),
            metas=model_data['Parameter'].values.reshape(-1, 1)
        )

        # Send outputs
        self.Outputs.forecast.send(forecast_table)
        self.Outputs.fitted_values.send(fitted_table)
        self.Outputs.residuals.send(residuals_table)
        self.Outputs.model_summary.send(str(summary))
        self.Outputs.model.send(model_table)


if __name__ == "__main__":
    WidgetPreview(AutoARIMAWidget).run(Table("iris"))




# with table
# import Orange
# from Orange.widgets import widget, gui, settings
# from Orange.widgets.utils.widgetpreview import WidgetPreview
# from Orange.data import Table, Domain, TimeVariable, ContinuousVariable
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from pmdarima import auto_arima
# from PyQt5.QtWidgets import QTextEdit, QTableWidget, QTableWidgetItem
# from PyQt5.QtGui import QFont
# from Orange.evaluation.scoring import MSE, MAE
# from statsmodels.tools.eval_measures import mse, aic, aicc
# from sklearn.metrics import mean_absolute_percentage_error
# import pandas as pd
#
#
# class AutoARIMAWidget(widget.OWWidget):
#     name = "Auto ARIMA"
#     description = "Automatic ARIMA modeling and forecasting"
#     icon = "icons/autoarima.svg"
#     priority = 10
#
#     class Inputs:
#         data = widget.Input("Time series data", Table)
#
#     class Outputs:
#         forecast = widget.Output("Forecast", Table)
#         fitted_values = widget.Output("Fitted Values", Table)
#         residuals = widget.Output("Residuals", Table)
#         model_summary = widget.Output("Model Summary", str)
#         model = widget.Output("Model", object)  # New output for the model
#
#     # Widget parameters
#     steps = settings.Setting(10)
#     max_p = settings.Setting(5)
#     max_d = settings.Setting(2)
#     max_q = settings.Setting(5)
#     seasonal = settings.Setting(True)
#     max_P = settings.Setting(2)
#     max_D = settings.Setting(1)
#     max_Q = settings.Setting(2)
#     m = settings.Setting(12)  # Seasonal period
#
#     def __init__(self):
#         super().__init__()
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "ARIMA Parameters")
#         gui.spin(box, self, "steps", 1, 100, label="Forecast Steps")
#         gui.spin(box, self, "max_p", 0, 10, label="Max p")
#         gui.spin(box, self, "max_d", 0, 10, label="Max d")
#         gui.spin(box, self, "max_q", 0, 10, label="Max q")
#         gui.checkBox(box, self, "seasonal", "Seasonal")
#         gui.spin(box, self, "max_P", 0, 10, label="Max P")
#         gui.spin(box, self, "max_D", 0, 10, label="Max D")
#         gui.spin(box, self, "max_Q", 0, 10, label="Max Q")
#         gui.spin(box, self, "m", 1, 365, label="Seasonal Period")
#
#         self.apply_button = gui.button(self.controlArea, self, "Apply", callback=self.apply)
#
#         # Table for displaying model information
#         self.info_box = gui.widgetBox(self.mainArea, "Model Information")
#         self.table_output = QTableWidget()
#         self.info_box.layout().addWidget(self.table_output)
#
#         # Text area for displaying model summary
#         self.summary_box = gui.widgetBox(self.mainArea, "Model Summary")
#         self.text_output = QTextEdit(readOnly=True)
#         self.summary_box.layout().addWidget(self.text_output)
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
#         # Assume the last column is the target variable
#         y = self.data.Y.ravel()
#
#         # Fit auto_arima model
#         model = auto_arima(y, start_p=0, start_q=0, max_p=self.max_p, max_d=self.max_d, max_q=self.max_q,
#                            start_P=0, start_Q=0, max_P=self.max_P, max_D=self.max_D, max_Q=self.max_Q,
#                            m=self.m, seasonal=self.seasonal, error_action='ignore', suppress_warnings=True,
#                            stepwise=True)
#
#         # Generate forecast
#         forecast = model.predict(n_periods=self.steps)
#
#         # Get fitted values and residuals
#         fitted_values = model.predict_in_sample()
#         residuals = y - fitted_values
#
#         # Create output tables
#         domain_forecast = Domain([ContinuousVariable("Forecast")])
#         domain_fitted = Domain([ContinuousVariable("Fitted Values")])
#         domain_residuals = Domain([ContinuousVariable("Residuals")])
#
#         forecast_table = Table.from_numpy(domain_forecast, forecast.reshape(-1, 1))
#         fitted_table = Table.from_numpy(domain_fitted, fitted_values.reshape(-1, 1))
#         residuals_table = Table.from_numpy(domain_residuals, residuals.reshape(-1, 1))
#
#         # Get model summary and parameters
#         summary = str(model.summary())
#         order = model.order
#         seasonal_order = model.seasonal_order
#         aic = model.aic()
#
#         # Calculate AICc manually
#         n = len(y)
#         k = len(model.params())
#         aicc_value = aic + (2 * k ** 2 + 2 * k) / (n - k - 1)
#
#         me = np.mean(residuals)
#         mse_value = np.mean((y - fitted_values) ** 2)
#         mae_value = np.mean(np.abs(y - fitted_values))
#         mape = np.mean(np.abs((y - fitted_values) / y)) * 100
#
#         # Calculate MASE (Mean Absolute Scaled Error)
#         diff = np.abs(np.diff(y))
#         mae_naive = np.mean(diff)
#         mase = np.mean(np.abs(residuals)) / mae_naive
#
#         # Calculate ACF1 (Autocorrelation at lag 1)
#         acf1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
#
#         # Prepare data for the table
#         data = [
#             ("Model Order", f"ARIMA{order}"),
#             ("Seasonal Order", f"{seasonal_order}" if self.seasonal else "N/A"),
#             ("AIC", f"{aic:.4f}"),
#             ("AICc", f"{aicc_value:.4f}"),
#             ("ME", f"{me:.4f}"),
#             ("MSE", f"{mse_value:.4f}"),
#             ("MAE", f"{mae_value:.4f}"),
#             ("MASE", f"{mase:.4f}"),
#             ("MAPE", f"{mape:.4f}"),
#             ("ACF1", f"{acf1:.4f}")
#         ]
#
#         # Set up the table
#         self.table_output.setColumnCount(2)
#         self.table_output.setRowCount(len(data))
#         self.table_output.setHorizontalHeaderLabels(["Metric", "Value"])
#
#         # Populate the table
#         for i, (metric, value) in enumerate(data):
#             self.table_output.setItem(i, 0, QTableWidgetItem(metric))
#             self.table_output.setItem(i, 1, QTableWidgetItem(value))
#
#         # Adjust table appearance
#         self.table_output.resizeColumnsToContents()
#         self.table_output.resizeRowsToContents()
#
#         # Display model summary in the text area
#         self.text_output.setPlainText(summary)
#
#         # Send outputs
#         self.Outputs.forecast.send(forecast_table)
#         self.Outputs.fitted_values.send(fitted_table)
#         self.Outputs.residuals.send(residuals_table)
#         self.Outputs.model_summary.send(summary)
#         self.Outputs.model.send(model)  # Send the model as output
#
#
# if __name__ == "__main__":
#     WidgetPreview(AutoARIMAWidget).run(Table("iris"))






# original
# import Orange
# from Orange.widgets import widget, gui, settings
# from Orange.widgets.utils.widgetpreview import WidgetPreview
# from Orange.data import Table, Domain, TimeVariable, ContinuousVariable
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from pmdarima import auto_arima
# from PyQt5.QtWidgets import QTextEdit
#
#
# class AutoARIMAWidget(widget.OWWidget):
#     name = "Auto ARIMA"
#     description = "Automatic ARIMA modeling and forecasting"
#     icon = "icons/autoarima.svg"
#     priority = 10
#
#     class Inputs:
#         data = widget.Input("Time series data", Table)
#
#     class Outputs:
#         forecast = widget.Output("Forecast", Table)
#         fitted_values = widget.Output("Fitted Values", Table)
#         residuals = widget.Output("Residuals", Table)
#         model_summary = widget.Output("Model Summary", str)
#
#     # Widget parameters
#     steps = settings.Setting(10)
#     max_p = settings.Setting(5)
#     max_d = settings.Setting(2)
#     max_q = settings.Setting(5)
#     seasonal = settings.Setting(True)
#     max_P = settings.Setting(2)
#     max_D = settings.Setting(1)
#     max_Q = settings.Setting(2)
#     m = settings.Setting(12)  # Seasonal period
#
#     def __init__(self):
#         super().__init__()
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "ARIMA Parameters")
#         gui.spin(box, self, "steps", 1, 100, label="Forecast Steps")
#         gui.spin(box, self, "max_p", 0, 10, label="Max p")
#         gui.spin(box, self, "max_d", 0, 10, label="Max d")
#         gui.spin(box, self, "max_q", 0, 10, label="Max q")
#         gui.checkBox(box, self, "seasonal", "Seasonal")
#         gui.spin(box, self, "max_P", 0, 10, label="Max P")
#         gui.spin(box, self, "max_D", 0, 10, label="Max D")
#         gui.spin(box, self, "max_Q", 0, 10, label="Max Q")
#         gui.spin(box, self, "m", 1, 365, label="Seasonal Period")
#
#         self.apply_button = gui.button(self.controlArea, self, "Apply", callback=self.apply)
#
#         # Text area for displaying model information
#         self.info_box = gui.widgetBox(self.mainArea, "Model Information")
#         self.text_output = QTextEdit(readOnly=True)
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
#         # Assume the last column is the target variable
#         y = self.data.Y.ravel()
#
#         # Fit auto_arima model
#         model = auto_arima(y, start_p=0, start_q=0, max_p=self.max_p, max_d=self.max_d, max_q=self.max_q,
#                            start_P=0, start_Q=0, max_P=self.max_P, max_D=self.max_D, max_Q=self.max_Q,
#                            m=self.m, seasonal=self.seasonal, error_action='ignore', suppress_warnings=True,
#                            stepwise=True)
#
#         # Generate forecast
#         forecast = model.predict(n_periods=self.steps)
#
#         # Get fitted values and residuals
#         fitted_values = model.predict_in_sample()
#         residuals = y - fitted_values
#
#         # Create output tables
#         domain_forecast = Domain([ContinuousVariable("Forecast")])
#         domain_fitted = Domain([ContinuousVariable("Fitted Values")])
#         domain_residuals = Domain([ContinuousVariable("Residuals")])
#
#         forecast_table = Table.from_numpy(domain_forecast, forecast.reshape(-1, 1))
#         fitted_table = Table.from_numpy(domain_fitted, fitted_values.reshape(-1, 1))
#         residuals_table = Table.from_numpy(domain_residuals, residuals.reshape(-1, 1))
#
#         # Get model summary and parameters
#         summary = str(model.summary())
#         order = model.order
#         seasonal_order = model.seasonal_order
#         aic = model.aic()
#         coefficients = model.params()
#
#         # Display model information in the text area
#         info_text = f"Model Order: ARIMA{order}\n"
#         if self.seasonal:
#             info_text += f"Seasonal Order: {seasonal_order}\n"
#         info_text += f"AIC: {aic:.2f}\n\n"
#         info_text += "Coefficients:\n"
#
#         # Handle coefficients correctly
#         coef_names = ['ar', 'ma', 'sar', 'sma']
#         for i, coef in enumerate(coefficients):
#             if i < len(order):
#                 name = f"{coef_names[0]}.L{i + 1}"
#             elif i < len(order) + len(seasonal_order):
#                 name = f"{coef_names[2]}.S{i - len(order) + 1}"
#             else:
#                 name = f"exog{i - len(order) - len(seasonal_order)}"
#             info_text += f"{name}: {coef:.4f}\n"
#
#         info_text += "\nModel Summary:\n" + summary
#
#         self.text_output.setPlainText(info_text)
#
#         # Send outputs
#         self.Outputs.forecast.send(forecast_table)
#         self.Outputs.fitted_values.send(fitted_table)
#         self.Outputs.residuals.send(residuals_table)
#         self.Outputs.model_summary.send(summary)
#
#
# if __name__ == "__main__":
#     WidgetPreview(AutoARIMAWidget).run(Table("iris"))
