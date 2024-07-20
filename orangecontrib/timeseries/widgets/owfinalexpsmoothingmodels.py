import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from scipy import stats


class OWExponentialSmoothing(widget.OWWidget):
    name = "Exponential Smoothing"
    description = "Fit Exponential Smoothing models and visualize results"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        forecast = Output("Forecast", Table)
        fitted_values = Output("Fitted Values", Table)
        residuals = Output("Residuals", Table)

    want_main_area = True

    # Settings
    model_type = Setting(0)  # 0: Simple, 1: Holt, 2: Holt-Winters Additive, 3: Holt-Winters Multiplicative
    seasonality = Setting(12)
    forecast_steps = Setting(10)
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
        gui.comboBox(model_box, self, "model_type",
                     items=["Simple", "Holt", "Holt-Winters Additive", "Holt-Winters Multiplicative"],
                     label="Model Type:", orientation="horizontal", callback=self.on_model_type_changed)

        # Seasonality
        self.seasonality_spin = gui.spin(
            model_box, self, "seasonality", 2, 365,
            label="Seasonality Period:", callback=self.on_seasonality_changed)

        # Forecast settings
        forecast_box = gui.widgetBox(self.controlArea, "Forecast Settings")
        gui.spin(forecast_box, self, "forecast_steps", 1, 100, label="Forecast Steps:", callback=self.on_param_changed)

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
        self.seasonality_spin.setEnabled(self.model_type >= 2)
        self.fit_model()

    def on_seasonality_changed(self):
        self.fit_model()

    def on_param_changed(self):
        self.fit_model()

    def on_bypass_stationarity_changed(self):
        self.check_stationarity()
        self.fit_model()

    def on_plot_type_changed(self):
        self.update_plot()

    def check_stationarity(self):
        if self.data is None or not self.target_variable or self.bypass_stationarity:
            self.is_stationary = True
            return

        y = self.data.get_column(self.target_variable)
        _, p_value, _, _ = kpss(y)
        self.is_stationary = p_value > 0.05

        if not self.is_stationary:
            self.info_label.setText("Time series is not stationary. Consider differencing or using ARIMA models.")
        else:
            self.info_label.setText("Time series is stationary.")

    def fit_model(self):
        self.results = None
        self.model = None
        if self.data is None or not self.target_variable:
            return

        y = self.data.get_column(self.target_variable)

        if self.model_type == 0:  # Simple Exponential Smoothing
            self.model = SimpleExpSmoothing(y)
        elif self.model_type == 1:  # Holt
            self.model = Holt(y)
        else:  # Holt-Winters
            seasonal = 'add' if self.model_type == 2 else 'mul'
            self.model = ExponentialSmoothing(y, seasonal_periods=self.seasonality, seasonal=seasonal)

        self.results = self.model.fit()
        self.update_plot()
        self.update_model_info()
        self.output_results()

    def update_model_info(self):
        if self.results is None:
            return

        info_text = []

        # Optimal Smoothing parameters
        info_text.append("Optimal Smoothing Parameters:")
        if hasattr(self.results, 'params'):
            params = self.results.params
            if 'smoothing_level' in params:
                info_text.append(f"α (alpha): {params['smoothing_level']:.4f}")
            if 'smoothing_trend' in params and self.model_type >= 1:
                info_text.append(f"β (beta): {params['smoothing_trend']:.4f}")
            if 'smoothing_seasonal' in params and self.model_type >= 2:
                info_text.append(f"γ (gamma): {params['smoothing_seasonal']:.4f}")
        else:
            info_text.append("Optimal parameters not available.")

        # # Model-specific parameters
        # info_text.append("\nModel Parameters:")
        # if self.model_type == 0:  # Simple Exponential Smoothing
        #     info_text.append(f"alpha: {self.results.params['smoothing_level']:.7f}")
        # elif self.model_type == 1:  # Holt's Exponential Smoothing
        #     info_text.append(f"alpha: {self.results.params['smoothing_level']:.7f}")
        #     info_text.append(f"beta: {self.results.params['smoothing_trend']:.7f}")
        # else:  # Holt-Winters
        #     info_text.append(f"alpha: {self.results.params['smoothing_level']:.7f}")
        #     info_text.append(f"beta: {self.results.params['smoothing_trend']:.7f}")
        #     info_text.append(f"gamma: {self.results.params['smoothing_seasonal']:.7f}")

        # Coefficients
        info_text.append("\nCoefficients:")

        # Initial level (a)
        if hasattr(self.results, 'level'):
            level = self.results.level[0]
            if level == 0:
                info_text.append("a (initial level): 0 (Note: This might indicate issues with level estimation)")
            else:
                info_text.append(f"a (initial level): {level:.6e}")

        # Initial trend (b)
        if self.model_type >= 1 and hasattr(self.results, 'trend') and len(self.results.trend) > 0:
            trend = self.results.trend[0]
            if np.isnan(trend):
                info_text.append("b (initial trend): NaN (Note: This might indicate issues with trend estimation)")
            else:
                info_text.append(f"b (initial trend): {trend:.6e}")
        elif self.model_type == 0:
            info_text.append("b (initial trend): Not applicable for Simple Exponential Smoothing")

        # Seasonal components (s1, s2, ...)
        if self.model_type >= 2 and hasattr(self.results, 'season'):
            for i, s in enumerate(self.results.season[:self.seasonality], 1):
                info_text.append(f"s{i}: {s:.6e}")
        elif self.model_type < 2:
            info_text.append("Seasonal components: Not applicable for this model")

        # Information criteria
        info_text.append("\nInformation Criteria:")
        for criterion in ['aic', 'bic', 'aicc']:
            if hasattr(self.results, criterion):
                value = getattr(self.results, criterion)
                info_text.append(f"{criterion.upper()}: {value:.4f}")

        # Calculate additional metrics
        y = self.model.endog
        y_hat = self.results.fittedvalues
        resid = self.results.resid

        me = np.mean(resid)
        rmse = np.sqrt(np.mean(resid ** 2))
        mae = np.mean(np.abs(resid))
        mape = np.mean(np.abs(resid / y)) * 100

        # Calculate MASE
        diff = np.abs(np.diff(y))
        mae_naive = np.mean(diff)
        mase = mae / mae_naive

        # Calculate ACF1
        acf1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]

        info_text.append("\nMetrics:")
        metrics = [
            f"ME: {me:.4f}",
            f"RMSE: {rmse:.4f}",
            f"MAE: {mae:.4f}",
            f"MAPE: {mape:.4f}",
            f"MASE: {mase:.4f}",
            f"ACF1: {acf1:.4f}"
        ]
        info_text.extend(metrics)

        self.info_label.setText("\n".join(info_text))

    # def update_model_info(self):
    #     if self.results is None:
    #         return
    #
    #     info_text = []
    #
    #     # Optimal Smoothing parameters
    #     info_text.append("Optimal Smoothing Parameters:")
    #     if hasattr(self.results, 'params'):
    #         params = self.results.params
    #         if 'smoothing_level' in params:
    #             info_text.append(f"α (alpha): {params['smoothing_level']:.4f}")
    #         if 'smoothing_trend' in params:
    #             info_text.append(f"β (beta): {params['smoothing_trend']:.4f}")
    #         if 'smoothing_seasonal' in params:
    #             info_text.append(f"γ (gamma): {params['smoothing_seasonal']:.4f}")
    #     else:
    #         info_text.append("Optimal parameters not available.")
    #
    #     # Existing Smoothing parameters
    #     info_text.append("\nSmoothing parameters:")
    #     for param in ['alpha', 'beta', 'gamma', 'phi']:
    #         if hasattr(self.results, param):
    #             value = getattr(self.results, param)
    #             info_text.append(f"{param}: {value:.7f}")
    #
    #     # Coefficients
    #     info_text.append("\nCoefficients:")
    #
    #     # Initial level (a)
    #     if hasattr(self.results, 'level'):
    #         info_text.append(f"a: {self.results.level[0]:.6e}")
    #
    #     # Initial trend (b)
    #     if hasattr(self.results, 'trend') and len(self.results.trend) > 0:
    #         info_text.append(f"b: {self.results.trend[0]:.6e}")
    #
    #     # Seasonal components (s1, s2, ...)
    #     if hasattr(self.results, 'season'):
    #         for i, s in enumerate(self.results.season[:self.seasonality], 1):
    #             info_text.append(f"s{i}: {s:.6e}")
    #
    #     # Information criteria
    #     info_text.append("\nInformation Criteria:")
    #     for criterion in ['aic', 'bic', 'aicc']:
    #         if hasattr(self.results, criterion):
    #             value = getattr(self.results, criterion)
    #             info_text.append(f"{criterion.upper()}: {value:.4f}")
    #
    #     # Calculate additional metrics
    #     y = self.model.endog
    #     y_hat = self.results.fittedvalues
    #     resid = self.results.resid
    #
    #     me = np.mean(resid)
    #     rmse = np.sqrt(np.mean(resid ** 2))
    #     mae = np.mean(np.abs(resid))
    #     mape = np.mean(np.abs(resid / y)) * 100
    #
    #     # Calculate MASE
    #     diff = np.abs(np.diff(y))
    #     mae_naive = np.mean(diff)
    #     mase = mae / mae_naive
    #
    #     # Calculate ACF1
    #     acf1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    #
    #     info_text.append("\nMetrics:")
    #     metrics = [
    #         f"ME: {me:.4f}",
    #         f"RMSE: {rmse:.4f}",
    #         f"MAE: {mae:.4f}",
    #         f"MAPE: {mape:.4f}",
    #         f"MASE: {mase:.4f}",
    #         f"ACF1: {acf1:.4f}"
    #     ]
    #     info_text.extend(metrics)
    #
    #     self.info_label.setText("\n".join(info_text))

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

        # Get the observed data
        observed_data = np.ravel(self.model.endog)
        nobs = len(observed_data)
        observed_x = np.arange(nobs)

        # Generate forecast
        forecast = self.results.forecast(steps=self.forecast_steps)
        forecast_x = np.arange(nobs, nobs + len(forecast))

        # Calculate confidence intervals manually
        resid = self.results.resid
        mse = np.mean(resid ** 2)
        confidence = 0.95
        degrees_of_freedom = len(resid) - 1
        t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)

        forecast_std_error = np.sqrt(mse * (1 + np.arange(1, len(forecast) + 1)))
        ci_lower = forecast - t_value * forecast_std_error
        ci_upper = forecast + t_value * forecast_std_error

        # Plot observed data
        observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2),
                                              name='Observed')
        legend.addItem(observed_plot, 'Observed')

        # Plot forecast
        forecast_plot = self.plot_widget.plot(forecast_x, forecast, pen=pg.mkPen(color=(255, 0, 0), width=2),
                                              name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        # Plot confidence intervals
        ci_lower_plot = self.plot_widget.plot(forecast_x, ci_lower, pen=pg.mkPen(color=(200, 200, 200), width=2),
                                              name='CI Lower')
        ci_upper_plot = self.plot_widget.plot(forecast_x, ci_upper, pen=pg.mkPen(color=(200, 200, 200), width=2),
                                              name='CI Upper')
        legend.addItem(ci_lower_plot, 'CI Lower')
        legend.addItem(ci_upper_plot, 'CI Upper')

        # Fill the area between confidence intervals
        ci_fill = pg.FillBetweenItem(ci_upper_plot, ci_lower_plot, brush=pg.mkBrush(200, 200, 200, 100))
        self.plot_widget.addItem(ci_fill)

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Forecast')

    def plot_fitted_values(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = len(self.model.endog)
        observed_x = np.arange(nobs)

        observed_data = np.ravel(self.model.endog)

        observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2),
                                              name='Observed')
        legend.addItem(observed_plot, 'Observed')

        fitted_plot = self.plot_widget.plot(observed_x, self.results.fittedvalues,
                                            pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')
        legend.addItem(fitted_plot, 'Fitted')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Fitted Values')

    def clear_plot(self):
        self.plot_widget.clear()

    def output_results(self):
        if self.results is None:
            self.Outputs.forecast.send(None)
            self.Outputs.fitted_values.send(None)
            self.Outputs.residuals.send(None)
            return

        # Forecast output
        forecast = self.results.forecast(steps=self.forecast_steps)
        forecast_domain = Domain([ContinuousVariable('Forecast')],
                                 metas=[StringVariable('Time')])
        forecast_time = np.arange(len(self.model.endog), len(self.model.endog) + len(forecast))
        forecast_table = Table(forecast_domain, np.atleast_2d(forecast).T,
                               metas=np.atleast_2d(forecast_time).T)
        self.Outputs.forecast.send(forecast_table)

        # Fitted values output
        fitted_domain = Domain([ContinuousVariable('Fitted Values')],
                               metas=[StringVariable('Time')])
        fitted_time = np.arange(len(self.model.endog))
        fitted_table = Table(fitted_domain, np.atleast_2d(self.results.fittedvalues).T,
                             metas=np.atleast_2d(fitted_time).T)
        self.Outputs.fitted_values.send(fitted_table)

        # Residuals output
        residuals_domain = Domain([ContinuousVariable('Residuals')],
                                  metas=[StringVariable('Time')])
        residuals_table = Table(residuals_domain, np.atleast_2d(self.results.resid).T,
                                metas=np.atleast_2d(fitted_time).T)
        self.Outputs.residuals.send(residuals_table)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWExponentialSmoothing).run()




# import Orange
# from Orange.widgets import widget, gui, settings
# from Orange.widgets.utils.widgetpreview import WidgetPreview
# from Orange.data import Table, Domain, ContinuousVariable
# import numpy as np
# from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
# from statsmodels.tsa.stattools import kpss
# from PyQt5.QtWidgets import QTextEdit
#
#
# class ExpSmoothingWidget(widget.OWWidget):
#     name = "Exponential Smoothing"
#     description = "Exponential Smoothing models for time series forecasting"
#     icon = "icons/expsmoothing.svg"
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
#     model_type = settings.Setting(0)
#     seasonal_periods = settings.Setting(12)
#     bypass_stationarity = settings.Setting(False)
#
#     def __init__(self):
#         super().__init__()
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Exponential Smoothing Parameters")
#         gui.spin(box, self, "steps", 1, 100, label="Forecast Steps")
#         gui.comboBox(box, self, "model_type", label="Model Type", items=[
#             "Simple Exponential Smoothing",
#             "Holt's Exponential Smoothing",
#             "Holt-Winters Additive",
#             "Holt-Winters Multiplicative"
#         ])
#         gui.spin(box, self, "seasonal_periods", 2, 365, label="Seasonal Periods")
#         gui.checkBox(box, self, "bypass_stationarity", "Bypass Stationarity Check")
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
#     def check_stationarity(self, y):
#         result = kpss(y, regression='c', nlags="auto")
#         return result[1] > 0.05, result  # return both the decision and full results
#
#     def apply(self):
#         if self.data is None:
#             return
#
#         # Assume the last column is the target variable
#         y = self.data.Y.ravel()
#
#         # Check stationarity
#         is_stationary, kpss_results = self.check_stationarity(y)
#
#         info_text = "Stationarity Check (KPSS Test):\n"
#         info_text += f"KPSS Statistic: {kpss_results[0]:.4f}\n"
#         info_text += f"p-value: {kpss_results[1]:.4f}\n"
#         info_text += f"Lags Used: {kpss_results[2]}\n"
#         info_text += f"Critical Values:\n"
#         for key, value in kpss_results[3].items():
#             info_text += f"  {key}: {value:.4f}\n"
#         info_text += f"\nConclusion: The time series is {'stationary' if is_stationary else 'not stationary'}.\n\n"
#
#         if not is_stationary and not self.bypass_stationarity:
#             info_text += "The model was not fitted due to non-stationarity. "
#             info_text += "You can either difference/transform your data or check 'Bypass Stationarity Check' to proceed anyway.\n"
#             self.text_output.setPlainText(info_text)
#             return
#
#         # Fit the selected model
#         if self.model_type == 0:  # Simple Exponential Smoothing
#             model = SimpleExpSmoothing(y)
#             fit = model.fit()
#         else:
#             trend = 'add' if self.model_type in [1, 2] else None
#             seasonal = 'add' if self.model_type == 2 else ('mul' if self.model_type == 3 else None)
#             model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal, seasonal_periods=self.seasonal_periods)
#             fit = model.fit()
#
#         # Generate forecast
#         forecast = fit.forecast(self.steps)
#
#         # Get fitted values and residuals
#         fitted_values = fit.fittedvalues
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
#         # Display model parameters in the text area
#         info_text += "Model Parameters:\n"
#         if self.model_type == 0:
#             info_text += f"Alpha (level): {fit.params['smoothing_level']:.4f}\n"
#         elif self.model_type == 1:
#             info_text += f"Alpha (level): {fit.params['smoothing_level']:.4f}\n"
#             info_text += f"Beta (trend): {fit.params['smoothing_trend']:.4f}\n"
#         else:
#             info_text += f"Alpha (level): {fit.params['smoothing_level']:.4f}\n"
#             info_text += f"Beta (trend): {fit.params['smoothing_trend']:.4f}\n"
#             info_text += f"Gamma (seasonal): {fit.params['smoothing_seasonal']:.4f}\n"
#
#         info_text += f"\nAIC: {fit.aic:.2f}\n"
#         info_text += f"BIC: {fit.bic:.2f}\n"
#         info_text += f"\nModel Summary:\n{str(fit.summary())}"
#
#         self.text_output.setPlainText(info_text)
#
#         # Send outputs
#         self.Outputs.forecast.send(forecast_table)
#         self.Outputs.fitted_values.send(fitted_table)
#         self.Outputs.residuals.send(residuals_table)
#         self.Outputs.model_summary.send(str(fit.summary()))
#
#
# if __name__ == "__main__":
#     WidgetPreview(ExpSmoothingWidget).run(Table("iris"))
