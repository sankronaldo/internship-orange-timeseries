import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class OWARIMASARIMAModel(widget.OWWidget):
    name = "ARIMA & SARIMA"
    description = "Fit ARIMA or SARIMA model and visualize results"
    icon = "icons/ow_arimasarima.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        residuals = Output("Residuals", Table)
        forecast = Output("Forecast", Table)
        fitted_values = Output("Fitted Values", Table)

    want_main_area = True

    # Settings
    model_type = Setting(0)  # 0: ARIMA, 1: SARIMA
    p = Setting(1)
    d = Setting(1)
    q = Setting(1)
    P = Setting(1)
    D = Setting(1)
    Q = Setting(1)
    S = Setting(12)
    forecast_steps = Setting(10)
    confidence_interval = Setting(0.95)
    target_variable = Setting("")
    plot_type = Setting(0)  # 0: Forecast, 1: Fitted Values

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.model = None
        self.results = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Model type selection
        model_box = gui.widgetBox(self.controlArea, "Model Selection")
        gui.comboBox(model_box, self, "model_type", items=["ARIMA", "SARIMA"],
                     label="Model Type:", orientation="horizontal", callback=self.on_model_type_changed)

        # ARIMA parameters
        arima_box = gui.widgetBox(self.controlArea, "ARIMA Parameters")
        gui.spin(arima_box, self, "p", 0, 10, label="p (AR order):", callback=self.on_param_changed)
        gui.spin(arima_box, self, "d", 0, 2, label="d (Differencing):", callback=self.on_param_changed)
        gui.spin(arima_box, self, "q", 0, 10, label="q (MA order):", callback=self.on_param_changed)

        # SARIMA parameters
        self.sarima_box = gui.widgetBox(self.controlArea, "SARIMA Parameters")
        gui.spin(self.sarima_box, self, "P", 0, 10, label="P (Seasonal AR):", callback=self.on_param_changed)
        gui.spin(self.sarima_box, self, "D", 0, 2, label="D (Seasonal Diff):", callback=self.on_param_changed)
        gui.spin(self.sarima_box, self, "Q", 0, 10, label="Q (Seasonal MA):", callback=self.on_param_changed)
        gui.spin(self.sarima_box, self, "S", 1, 365, label="S (Seasonal Period):", callback=self.on_param_changed)

        # Forecast settings
        forecast_box = gui.widgetBox(self.controlArea, "Forecast Settings")
        gui.spin(forecast_box, self, "forecast_steps", 1, 100, label="Forecast Steps:", callback=self.on_param_changed)
        gui.doubleSpin(forecast_box, self, "confidence_interval", 0.5, 0.99, 0.01, label="Confidence Interval:",
                       callback=self.on_param_changed)

        # Plot type selection
        plot_box = gui.widgetBox(self.controlArea, "Plot Selection")
        gui.comboBox(plot_box, self, "plot_type", items=["Forecast", "Fitted Values"],
                     label="Plot Type:", orientation="horizontal", callback=self.on_plot_type_changed)

        # Fit button
        self.fit_button = gui.button(self.controlArea, self, "Fit Model", callback=self.fit_model)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

        self.on_model_type_changed()

    @Inputs.time_series
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = getattr(data, 'time_variable', None)

            # Update target variable combo box options
            self.target_combo.clear()
            self.target_combo.addItem("")
            for var in data.domain.variables:
                if var.is_continuous:
                    self.target_combo.addItem(var.name)

            # Set initial target variable if previously selected
            if self.target_variable in data.domain:
                self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))

            self.on_target_variable_changed()
        else:
            self.data = None
            self.time_variable = None
            self.info_label.setText("No data on input.")
            self.clear_plot()

    def on_target_variable_changed(self):
        self.target_variable = self.target_combo.currentText()
        self.fit_model()

    def on_model_type_changed(self):
        self.sarima_box.setVisible(self.model_type == 1)
        self.fit_model()

    def on_param_changed(self):
        self.fit_model()

    def on_plot_type_changed(self):
        self.update_plot()

    def fit_model(self):
        if self.data is None or not self.target_variable:
            return

        value_var = self.data.domain[self.target_variable]
        y = self.data.get_column(value_var)

        if self.model_type == 0:  # ARIMA
            self.model = ARIMA(y, order=(self.p, self.d, self.q))
        else:  # SARIMA
            self.model = SARIMAX(y, order=(self.p, self.d, self.q),
                                 seasonal_order=(self.P, self.D, self.Q, self.S))

        self.results = self.model.fit()
        self.update_plot()
        self.update_model_info()
        self.output_residuals()
        self.output_forecast()
        self.output_fitted_values()

    def update_model_info(self):
        if self.results is None:
            return

        params = self.results.params
        stderr = self.results.bse

        param_text = "\n".join(f"{param}: {value:.4f} (s.e. {stderr[i]:.4f})"
                               for i, (param, value) in enumerate(zip(self.results.model.param_names, params)))

        metrics = {
            "AIC": self.results.aic,
            "AICc": self.results.aicc if hasattr(self.results, 'aicc') else "N/A",
            "BIC": self.results.bic,
            "Log Likelihood": self.results.llf,
            "MSE": np.mean(self.results.resid**2),
            "MAE": np.mean(np.abs(self.results.resid)),
            "ME": np.mean(self.results.resid),
            "MAPE": np.mean(np.abs(self.results.resid / self.results.model.endog)) * 100,
            "MASE": np.mean(np.abs(self.results.resid / np.mean(np.abs(np.diff(self.results.model.endog, n=1)))))
        }

        metrics_text = "\n".join(f"{key}: {value:.4f}" for key, value in metrics.items())

        self.info_label.setText(f"Model Parameters:\n{param_text}\n\nMetrics:\n{metrics_text}")

    def update_plot(self):
        self.plot_widget.clear()

        if self.results is None:
            return

        if self.plot_type == 0:  # Forecast
            self.plot_forecast()
        elif self.plot_type == 1:  # Fitted Values
            self.plot_fitted_values()

    def plot_forecast(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        forecast = self.results.forecast(steps=self.forecast_steps)
        ci = self.results.get_forecast(steps=self.forecast_steps).conf_int(alpha=1 - self.confidence_interval)

        nobs = self.model.nobs
        observed_x = np.arange(nobs)
        forecast_x = np.arange(nobs, nobs + len(forecast))

        observed_data = np.ravel(self.model.endog)

        observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        forecast_plot = self.plot_widget.plot(forecast_x, forecast, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        ci_lower = self.plot_widget.plot(forecast_x, ci[:, 0], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Lower')
        ci_upper = self.plot_widget.plot(forecast_x, ci[:, 1], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Upper')
        legend.addItem(ci_lower, 'CI Lower')
        legend.addItem(ci_upper, 'CI Upper')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Forecast')

    def plot_fitted_values(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = self.model.nobs
        observed_x = np.arange(nobs)

        observed_data = np.ravel(self.model.endog)

        observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        fitted_plot = self.plot_widget.plot(observed_x, self.results.fittedvalues, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')
        legend.addItem(fitted_plot, 'Fitted')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Fitted Values')

    def clear_plot(self):
        self.plot_widget.clear()

    def output_residuals(self):
        if self.results is None:
            self.Outputs.residuals.send(None)
            return

        residuals = self.results.resid
        domain = Domain([ContinuousVariable('Residuals')],
                        metas=[StringVariable('Time')])
        time_values = self.data.get_column(self.time_variable) if self.time_variable else np.arange(len(residuals))
        residuals_table = Table(domain, np.atleast_2d(residuals).T,
                                metas=np.atleast_2d(time_values).T)
        self.Outputs.residuals.send(residuals_table)

    def output_forecast(self):
        if self.results is None:
            self.Outputs.forecast.send(None)
            return

        forecast = self.results.forecast(steps=self.forecast_steps)
        ci = self.results.get_forecast(steps=self.forecast_steps).conf_int(alpha=1 - self.confidence_interval)

        domain = Domain([ContinuousVariable('Forecast'),
                         ContinuousVariable('Lower CI'),
                         ContinuousVariable('Upper CI')],
                        metas=[StringVariable('Time')])

        nobs = self.model.nobs
        forecast_time = np.arange(nobs, nobs + len(forecast))

        forecast_table = Table(domain,
                               np.column_stack((forecast, ci)),
                               metas=np.atleast_2d(forecast_time).T)
        self.Outputs.forecast.send(forecast_table)

    def output_fitted_values(self):
        if self.results is None:
            self.Outputs.fitted_values.send(None)
            return

        fitted_values = self.results.fittedvalues
        domain = Domain([ContinuousVariable('Fitted Values')],
                        metas=[StringVariable('Time')])

        time_values = self.data.get_column(self.time_variable) if self.time_variable else np.arange(len(fitted_values))
        fitted_values_table = Table(domain, np.atleast_2d(fitted_values).T,
                                    metas=np.atleast_2d(time_values).T)
        self.Outputs.fitted_values.send(fitted_values_table)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWARIMASARIMAModel).run()







#
# import numpy as np
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, Domain, ContinuousVariable, StringVariable
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# import pyqtgraph as pg
# from PyQt5.QtCore import Qt
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
#
# class OWARIMASARIMAModel(widget.OWWidget):
#     name = "ARIMA & SARIMA"
#     description = "Fit ARIMA or SARIMA model and visualize results"
#     icon = "icons/ow_arimasarima.svg"
#     priority = 10
#
#     class Inputs:
#         time_series = Input("Time series", Table)
#
#     class Outputs:
#         residuals = Output("Residuals", Table)
#
#     want_main_area = True
#
#     # Settings
#     model_type = Setting(0)  # 0: ARIMA, 1: SARIMA
#     p = Setting(1)
#     d = Setting(1)
#     q = Setting(1)
#     P = Setting(1)
#     D = Setting(1)
#     Q = Setting(1)
#     S = Setting(12)
#     forecast_steps = Setting(10)
#     confidence_interval = Setting(0.95)
#     target_variable = Setting("")
#     plot_type = Setting(0)  # 0: Forecast, 1: Fitted Values
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#         self.time_variable = None
#         self.model = None
#         self.results = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(box, "No data on input.")
#
#         # Target variable selection
#         self.target_combo = gui.comboBox(
#             box, self, "target_variable", label="Target Variable:",
#             orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)
#
#         # Model type selection
#         model_box = gui.widgetBox(self.controlArea, "Model Selection")
#         gui.comboBox(model_box, self, "model_type", items=["ARIMA", "SARIMA"],
#                      label="Model Type:", orientation="horizontal", callback=self.on_model_type_changed)
#
#         # ARIMA parameters
#         arima_box = gui.widgetBox(self.controlArea, "ARIMA Parameters")
#         gui.spin(arima_box, self, "p", 0, 10, label="p (AR order):", callback=self.on_param_changed)
#         gui.spin(arima_box, self, "d", 0, 2, label="d (Differencing):", callback=self.on_param_changed)
#         gui.spin(arima_box, self, "q", 0, 10, label="q (MA order):", callback=self.on_param_changed)
#
#         # SARIMA parameters
#         self.sarima_box = gui.widgetBox(self.controlArea, "SARIMA Parameters")
#         gui.spin(self.sarima_box, self, "P", 0, 10, label="P (Seasonal AR):", callback=self.on_param_changed)
#         gui.spin(self.sarima_box, self, "D", 0, 2, label="D (Seasonal Diff):", callback=self.on_param_changed)
#         gui.spin(self.sarima_box, self, "Q", 0, 10, label="Q (Seasonal MA):", callback=self.on_param_changed)
#         gui.spin(self.sarima_box, self, "S", 1, 365, label="S (Seasonal Period):", callback=self.on_param_changed)
#
#         # Forecast settings
#         forecast_box = gui.widgetBox(self.controlArea, "Forecast Settings")
#         gui.spin(forecast_box, self, "forecast_steps", 1, 100, label="Forecast Steps:", callback=self.on_param_changed)
#         gui.doubleSpin(forecast_box, self, "confidence_interval", 0.5, 0.99, 0.01, label="Confidence Interval:",
#                        callback=self.on_param_changed)
#
#         # Plot type selection
#         plot_box = gui.widgetBox(self.controlArea, "Plot Selection")
#         gui.comboBox(plot_box, self, "plot_type", items=["Forecast", "Fitted Values"],
#                      label="Plot Type:", orientation="horizontal", callback=self.on_plot_type_changed)
#
#         # Fit button
#         self.fit_button = gui.button(self.controlArea, self, "Fit Model", callback=self.fit_model)
#
#         # Set up the main area with plot widget
#         self.plot_widget = PlotWidget(background="w")
#         self.mainArea.layout().addWidget(self.plot_widget)
#
#         self.on_model_type_changed()
#
#     @Inputs.time_series
#     def set_data(self, data):
#         if data is not None:
#             self.data = data
#             self.info_label.setText(f"{len(data)} instances on input.")
#             self.time_variable = getattr(data, 'time_variable', None)
#
#             # Update target variable combo box options
#             self.target_combo.clear()
#             self.target_combo.addItem("")
#             for var in data.domain.variables:
#                 if var.is_continuous:
#                     self.target_combo.addItem(var.name)
#
#             # Set initial target variable if previously selected
#             if self.target_variable in data.domain:
#                 self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))
#
#             self.on_target_variable_changed()
#         else:
#             self.data = None
#             self.time_variable = None
#             self.info_label.setText("No data on input.")
#             self.clear_plot()
#
#     def on_target_variable_changed(self):
#         self.target_variable = self.target_combo.currentText()
#         self.fit_model()
#
#     def on_model_type_changed(self):
#         self.sarima_box.setVisible(self.model_type == 1)
#         self.fit_model()
#
#     def on_param_changed(self):
#         self.fit_model()
#
#     def on_plot_type_changed(self):
#         self.update_plot()
#
#     def fit_model(self):
#         if self.data is None or not self.target_variable:
#             return
#
#         value_var = self.data.domain[self.target_variable]
#         y = self.data.get_column(value_var)
#
#         if self.model_type == 0:  # ARIMA
#             self.model = ARIMA(y, order=(self.p, self.d, self.q))
#         else:  # SARIMA
#             self.model = SARIMAX(y, order=(self.p, self.d, self.q),
#                                  seasonal_order=(self.P, self.D, self.Q, self.S))
#
#         self.results = self.model.fit()
#         self.update_plot()
#         self.update_model_info()
#         self.output_residuals()
#
#     def update_model_info(self):
#         if self.results is None:
#             return
#
#         params = self.results.params
#         stderr = self.results.bse
#
#         param_text = "\n".join(f"{param}: {value:.4f} (s.e. {stderr[i]:.4f})"
#                                for i, (param, value) in enumerate(zip(self.results.model.param_names, params)))
#
#         metrics = {
#             "AIC": self.results.aic,
#             "AICc": self.results.aicc if hasattr(self.results, 'aicc') else "N/A",
#             "BIC": self.results.bic,
#             "Log Likelihood": self.results.llf,
#             # "Sigma^2": self.results.sigma2,
#             "MSE": np.mean(self.results.resid**2),
#             "MAE": np.mean(np.abs(self.results.resid)),
#             "ME": np.mean(self.results.resid),
#             "MAPE": np.mean(np.abs(self.results.resid / self.results.model.endog)) * 100,
#             "MASE": np.mean(np.abs(self.results.resid / np.mean(np.abs(np.diff(self.results.model.endog, n=1)))))
#         }
#
#         metrics_text = "\n".join(f"{key}: {value:.4f}" for key, value in metrics.items())
#
#         self.info_label.setText(f"Model Parameters:\n{param_text}\n\nMetrics:\n{metrics_text}")
#
#     def update_plot(self):
#         self.plot_widget.clear()
#
#         if self.results is None:
#             return
#
#         if self.plot_type == 0:  # Forecast
#             self.plot_forecast()
#         elif self.plot_type == 1:  # Fitted Values
#             self.plot_fitted_values()
#
#     def plot_forecast(self):
#         self.plot_widget.clear()
#         legend = pg.LegendItem(offset=(50, 30))  # Create a legend and position it at the bottom-right corner
#         legend.setParentItem(self.plot_widget.graphicsItem())
#
#         forecast = self.results.forecast(steps=self.forecast_steps)
#         ci = self.results.get_forecast(steps=self.forecast_steps).conf_int(alpha=1 - self.confidence_interval)
#
#         nobs = self.model.nobs
#         observed_x = np.arange(nobs)
#         forecast_x = np.arange(nobs, nobs + len(forecast))
#
#         # Ensure data is 1D
#         observed_data = np.ravel(self.model.endog)
#
#         # Plot observed data
#         observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
#         legend.addItem(observed_plot, 'Observed')
#
#         # Plot forecast
#         forecast_plot = self.plot_widget.plot(forecast_x, forecast, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
#         legend.addItem(forecast_plot, 'Forecast')
#
#         # Plot confidence intervals
#         ci_lower = self.plot_widget.plot(forecast_x, ci[:, 0], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Lower')
#         ci_upper = self.plot_widget.plot(forecast_x, ci[:, 1], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Upper')
#         legend.addItem(ci_lower, 'CI Lower')
#         legend.addItem(ci_upper, 'CI Upper')
#
#         self.plot_widget.setLabel('left', self.target_variable)
#         self.plot_widget.setLabel('bottom', 'Time')
#         self.plot_widget.setTitle('Forecast')
#
#     def plot_fitted_values(self):
#         self.plot_widget.clear()
#         legend = pg.LegendItem(offset=(50, 30))  # Create a legend and position it at the bottom-right corner
#         legend.setParentItem(self.plot_widget.graphicsItem())
#
#         nobs = self.model.nobs
#         observed_x = np.arange(nobs)
#
#         # Ensure data is 1D
#         observed_data = np.ravel(self.model.endog)
#
#         observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
#         legend.addItem(observed_plot, 'Observed')
#
#         fitted_plot = self.plot_widget.plot(observed_x, self.results.fittedvalues, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')
#         legend.addItem(fitted_plot, 'Fitted')
#
#         self.plot_widget.setLabel('left', self.target_variable)
#         self.plot_widget.setLabel('bottom', 'Time')
#         self.plot_widget.setTitle('Fitted Values')
#
#     def clear_plot(self):
#         self.plot_widget.clear()
#
#     def output_residuals(self):
#         if self.results is None:
#             self.Outputs.residuals.send(None)
#             return
#
#         residuals = self.results.resid
#         domain = Domain([ContinuousVariable('Residuals')],
#                         metas=[StringVariable('Time')])
#         time_values = self.data.get_column(self.time_variable) if self.time_variable else np.arange(len(residuals))
#         residuals_table = Table(domain, np.atleast_2d(residuals).T,
#                                 metas=np.atleast_2d(time_values).T)
#         self.Outputs.residuals.send(residuals_table)
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(OWARIMASARIMAModel).run()
#