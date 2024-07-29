import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from scipy import stats

class OWETS(widget.OWWidget):
    name = "ETS Model"
    description = "Fit ETS (Error, Trend, Seasonal) models and visualize results"
    icon = "icons/ets.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        forecast = Output("Forecast", Table)
        fitted_values = Output("Fitted Values", Table)
        residuals = Output("Residuals", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    error_type = Setting("add")
    trend_type = Setting("add")
    seasonal_type = Setting("add")
    seasonal_period = Setting(12)
    forecast_steps = Setting(10)
    plot_type = Setting(0)  # 0: Forecast, 1: Fitted Values
    damped_trend = Setting(False)

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
            orientation="horizontal", callback=self.on_target_variable_changed)

        # Model components selection
        components_box = gui.widgetBox(self.controlArea, "Model Components")
        self.error_combo = gui.comboBox(components_box, self, "error_type",
                     items=["add", "mul"],
                     label="Error Type:", orientation="horizontal",
                     callback=self.on_component_changed)
        self.trend_combo = gui.comboBox(components_box, self, "trend_type",
                     items=["add", "mul", "none"],
                     label="Trend Type:", orientation="horizontal",
                     callback=self.on_component_changed)
        self.seasonal_combo = gui.comboBox(components_box, self, "seasonal_type",
                     items=["add", "mul", "none"],
                     label="Seasonal Type:", orientation="horizontal",
                     callback=self.on_component_changed)

        # Damped trend checkbox
        self.damped_check = gui.checkBox(components_box, self, "damped_trend",
                                         label="Damped Trend",
                                         callback=self.on_component_changed)

        # Seasonality
        self.seasonality_spin = gui.spin(
            components_box, self, "seasonal_period", 2, 365,
            label="Seasonal Period:", callback=self.on_seasonality_changed)

        # Forecast settings
        forecast_box = gui.widgetBox(self.controlArea, "Forecast Settings")
        gui.spin(forecast_box, self, "forecast_steps", 1, 100,
                 label="Forecast Steps:", callback=self.on_param_changed)

        # Plot type selection
        plot_box = gui.widgetBox(self.controlArea, "Plot Selection")
        gui.comboBox(plot_box, self, "plot_type",
                     items=["Forecast", "Fitted Values"],
                     label="Plot Type:", orientation="horizontal",
                     callback=self.on_plot_type_changed)

        # Fit button
        self.fit_button = gui.button(self.controlArea, self, "Fit Model",
                                     callback=self.fit_model)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

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

    def on_component_changed(self):
        self.error_type = self.error_combo.currentText()
        self.trend_type = self.trend_combo.currentText()
        self.seasonal_type = self.seasonal_combo.currentText()
        self.fit_model()

    def on_seasonality_changed(self):
        self.fit_model()

    def on_param_changed(self):
        self.fit_model()

    def on_plot_type_changed(self):
        self.update_plot()

    def fit_model(self):
        self.results = None
        self.model = None
        if self.data is None or not self.target_variable:
            return

        y = self.data.get_column(self.target_variable)

        # Set up the model
        error = str(self.error_type)
        trend = None if self.trend_type == "none" else str(self.trend_type)
        seasonal = None if self.seasonal_type == "none" else str(self.seasonal_type)

        try:
            self.info_label.setText(f"Fitting model with error={error}, trend={trend}, seasonal={seasonal}, "
                                    f"seasonal_periods={self.seasonal_period}, damped_trend={self.damped_trend}")
            self.model = ETSModel(
                y,
                error=error,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=self.seasonal_period if seasonal else None,
                damped_trend=self.damped_trend
            )
            self.results = self.model.fit()
            self.update_plot()
            self.update_model_info()
            self.output_results()
        except Exception as e:
            self.info_label.setText(f"Error fitting model: {str(e)}\n"
                                    f"Parameters: error={error}, trend={trend}, seasonal={seasonal}, "
                                    f"seasonal_periods={self.seasonal_period}, damped_trend={self.damped_trend}")

    def update_model_info(self):
        if self.results is None:
            return

        info_text = []

        # Model parameters
        info_text.append("Model Parameters:")
        info_text.append(f"Error type: {self.error_type}")
        info_text.append(f"Trend type: {self.trend_type}")
        if self.trend_type != "none":
            info_text.append(f"Damped trend: {'Yes' if self.damped_trend else 'No'}")
        info_text.append(f"Seasonal type: {self.seasonal_type}")
        if self.seasonal_type != "none":
            info_text.append(f"Seasonal period: {self.seasonal_period}")

        # Smoothing parameters
        info_text.append("\nSmoothing parameters:")
        params = self.results.params
        param_names = ['alpha', 'beta', 'gamma', 'phi']
        for name in param_names:
            if name in params:
                info_text.append(f"    {name:5} = {params[name]:.4f}")

        # Initial states
        info_text.append("Initial states:")
        initial_states = self.results.initial_state
        info_text.append(f"    l = {initial_states[0]:.4f}")
        if len(initial_states) > 1:
            info_text.append(f"    b = {initial_states[1]:.4f}")
        if len(initial_states) > 2:
            s_values = initial_states[2:]
            info_text.append("    s =")
            for i, v in enumerate(s_values):
                info_text.append(f"        s_{i + 1} = {v:.4f}")

        # Sigma
        # info_text.append(f"sigma:  {self.results.sigma2:.4f}")

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

        info_text.append("\nMetrics:")
        metrics = [
            f"ME: {me:.4f}",
            f"RMSE: {rmse:.4f}",
            f"MAE: {mae:.4f}",
            f"MAPE: {mape:.4f}"
        ]
        info_text.extend(metrics)

        self.info_label.setText("\n".join(info_text))

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
        observed_data = self.model.endog
        nobs = len(observed_data)
        observed_x = np.arange(nobs)

        # Generate forecast
        forecast = self.results.forecast(steps=self.forecast_steps)
        forecast_x = np.arange(nobs, nobs + len(forecast))

        # Plot observed data
        observed_plot = self.plot_widget.plot(observed_x, observed_data,
                                              pen=pg.mkPen(color=(0, 0, 255), width=2),
                                              name='Observed')
        legend.addItem(observed_plot, 'Observed')

        # Plot forecast
        forecast_plot = self.plot_widget.plot(forecast_x, forecast,
                                              pen=pg.mkPen(color=(255, 0, 0), width=2),
                                              name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Forecast')

    def plot_fitted_values(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = len(self.model.endog)
        observed_x = np.arange(nobs)

        observed_data = self.model.endog

        observed_plot = self.plot_widget.plot(observed_x, observed_data,
                                              pen=pg.mkPen(color=(0, 0, 255), width=2),
                                              name='Observed')
        legend.addItem(observed_plot, 'Observed')

        fitted_plot = self.plot_widget.plot(observed_x, self.results.fittedvalues,
                                            pen=pg.mkPen(color=(255, 0, 0), width=2),
                                            name='Fitted')
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
        forecast_time = np.arange(len(self.model.endog),
                                  len(self.model.endog) + len(forecast))
        forecast_table = Table(forecast_domain, np.atleast_2d(forecast).T,
                               metas=np.atleast_2d(forecast_time).T)
        self.Outputs.forecast.send(forecast_table)

        # Fitted values output
        fitted_domain = Domain([ContinuousVariable('Fitted Values')],
                               metas=[StringVariable('Time')])
        fitted_time = np.arange(len(self.model.endog))
        fitted_table = Table(fitted_domain,
                             np.atleast_2d(self.results.fittedvalues).T,
                             metas=np.atleast_2d(fitted_time).T)
        self.Outputs.fitted_values.send(fitted_table)

        # Residuals output
        residuals_domain = Domain([ContinuousVariable('Residuals')],
                                  metas=[StringVariable('Time')])
        residuals_table = Table(residuals_domain,
                                np.atleast_2d(self.results.resid).T,
                                metas=np.atleast_2d(fitted_time).T)
        self.Outputs.residuals.send(residuals_table)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWETS).run()







# import numpy as np
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, Domain, ContinuousVariable, StringVariable
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# import pyqtgraph as pg
# from PyQt5.QtCore import Qt
# from statsmodels.tsa.exponential_smoothing.ets import ETSModel
# from scipy import stats
#
# class OWETS(widget.OWWidget):
#     name = "ETS Model"
#     description = "Fit ETS (Error, Trend, Seasonal) models and visualize results"
#     icon = "icons/ets.svg"
#     priority = 10
#
#     class Inputs:
#         time_series = Input("Time series", Table)
#
#     class Outputs:
#         forecast = Output("Forecast", Table)
#         fitted_values = Output("Fitted Values", Table)
#         residuals = Output("Residuals", Table)
#
#     want_main_area = True
#
#     # Settings
#     target_variable = Setting("")
#     error_type = Setting("add")
#     trend_type = Setting("add")
#     seasonal_type = Setting("add")
#     seasonal_period = Setting(12)
#     forecast_steps = Setting(10)
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
#             orientation="horizontal", callback=self.on_target_variable_changed)
#
#         # Model components selection
#         components_box = gui.widgetBox(self.controlArea, "Model Components")
#         self.error_combo = gui.comboBox(components_box, self, "error_type",
#                      items=["add", "mul"],
#                      label="Error Type:", orientation="horizontal",
#                      callback=self.on_component_changed)
#         self.trend_combo = gui.comboBox(components_box, self, "trend_type",
#                      items=["add", "mul", "none"],
#                      label="Trend Type:", orientation="horizontal",
#                      callback=self.on_component_changed)
#         self.seasonal_combo = gui.comboBox(components_box, self, "seasonal_type",
#                      items=["add", "mul", "none"],
#                      label="Seasonal Type:", orientation="horizontal",
#                      callback=self.on_component_changed)
#
#         # Seasonality
#         self.seasonality_spin = gui.spin(
#             components_box, self, "seasonal_period", 2, 365,
#             label="Seasonal Period:", callback=self.on_seasonality_changed)
#
#         # Forecast settings
#         forecast_box = gui.widgetBox(self.controlArea, "Forecast Settings")
#         gui.spin(forecast_box, self, "forecast_steps", 1, 100,
#                  label="Forecast Steps:", callback=self.on_param_changed)
#
#         # Plot type selection
#         plot_box = gui.widgetBox(self.controlArea, "Plot Selection")
#         gui.comboBox(plot_box, self, "plot_type",
#                      items=["Forecast", "Fitted Values"],
#                      label="Plot Type:", orientation="horizontal",
#                      callback=self.on_plot_type_changed)
#
#         # Fit button
#         self.fit_button = gui.button(self.controlArea, self, "Fit Model",
#                                      callback=self.fit_model)
#
#         # Set up the main area with plot widget
#         self.plot_widget = PlotWidget(background="w")
#         self.mainArea.layout().addWidget(self.plot_widget)
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
#     def on_component_changed(self):
#         self.error_type = self.error_combo.currentText()
#         self.trend_type = self.trend_combo.currentText()
#         self.seasonal_type = self.seasonal_combo.currentText()
#         self.fit_model()
#
#     def on_seasonality_changed(self):
#         self.fit_model()
#
#     def on_param_changed(self):
#         self.fit_model()
#
#     def on_plot_type_changed(self):
#         self.update_plot()
#
#     def fit_model(self):
#         self.results = None
#         self.model = None
#         if self.data is None or not self.target_variable:
#             return
#
#         y = self.data.get_column(self.target_variable)
#
#         # Set up the model
#         error = str(self.error_type)
#         trend = None if self.trend_type == "none" else str(self.trend_type)
#         seasonal = None if self.seasonal_type == "none" else str(self.seasonal_type)
#
#         try:
#             self.info_label.setText(f"Fitting model with error={error}, trend={trend}, seasonal={seasonal}, seasonal_periods={self.seasonal_period}")
#             self.model = ETSModel(
#                 y,
#                 error=error,
#                 trend=trend,
#                 seasonal=seasonal,
#                 seasonal_periods=self.seasonal_period if seasonal else None
#             )
#             self.results = self.model.fit()
#             self.update_plot()
#             self.update_model_info()
#             self.output_results()
#         except Exception as e:
#             self.info_label.setText(f"Error fitting model: {str(e)}\n"
#                                     f"Parameters: error={error}, trend={trend}, seasonal={seasonal}, seasonal_periods={self.seasonal_period}")
#
#     def update_model_info(self):
#         if self.results is None:
#             return
#
#         info_text = []
#
#         # Model parameters
#         info_text.append("Model Parameters:")
#         info_text.append(f"Error type: {self.error_type}")
#         info_text.append(f"Trend type: {self.trend_type}")
#         info_text.append(f"Seasonal type: {self.seasonal_type}")
#         if self.seasonal_type != "none":
#             info_text.append(f"Seasonal period: {self.seasonal_period}")
#
#         # Smoothing parameters
#         info_text.append("\nSmoothing Parameters:")
#         params = self.results.params
#         param_names = ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal']
#         for i, param in enumerate(params):
#             if i < len(param_names):
#                 info_text.append(f"{param_names[i]}: {param:.4f}")
#
#         # Information criteria
#         info_text.append("\nInformation Criteria:")
#         for criterion in ['aic', 'bic', 'aicc']:
#             if hasattr(self.results, criterion):
#                 value = getattr(self.results, criterion)
#                 info_text.append(f"{criterion.upper()}: {value:.4f}")
#
#         # Calculate additional metrics
#         y = self.model.endog
#         y_hat = self.results.fittedvalues
#         resid = self.results.resid
#
#         me = np.mean(resid)
#         rmse = np.sqrt(np.mean(resid ** 2))
#         mae = np.mean(np.abs(resid))
#         mape = np.mean(np.abs(resid / y)) * 100
#
#         info_text.append("\nMetrics:")
#         metrics = [
#             f"ME: {me:.4f}",
#             f"RMSE: {rmse:.4f}",
#             f"MAE: {mae:.4f}",
#             f"MAPE: {mape:.4f}"
#         ]
#         info_text.extend(metrics)
#
#         self.info_label.setText("\n".join(info_text))
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
#         legend = pg.LegendItem(offset=(50, 30))
#         legend.setParentItem(self.plot_widget.graphicsItem())
#
#         # Get the observed data
#         observed_data = self.model.endog
#         nobs = len(observed_data)
#         observed_x = np.arange(nobs)
#
#         # Generate forecast
#         forecast = self.results.forecast(steps=self.forecast_steps)
#         forecast_x = np.arange(nobs, nobs + len(forecast))
#
#         # Plot observed data
#         observed_plot = self.plot_widget.plot(observed_x, observed_data,
#                                               pen=pg.mkPen(color=(0, 0, 255), width=2),
#                                               name='Observed')
#         legend.addItem(observed_plot, 'Observed')
#
#         # Plot forecast
#         forecast_plot = self.plot_widget.plot(forecast_x, forecast,
#                                               pen=pg.mkPen(color=(255, 0, 0), width=2),
#                                               name='Forecast')
#         legend.addItem(forecast_plot, 'Forecast')
#
#         self.plot_widget.setLabel('left', self.target_variable)
#         self.plot_widget.setLabel('bottom', 'Time')
#         self.plot_widget.setTitle('Forecast')
#
#     def plot_fitted_values(self):
#         self.plot_widget.clear()
#         legend = pg.LegendItem(offset=(50, 30))
#         legend.setParentItem(self.plot_widget.graphicsItem())
#
#         nobs = len(self.model.endog)
#         observed_x = np.arange(nobs)
#
#         observed_data = self.model.endog
#
#         observed_plot = self.plot_widget.plot(observed_x, observed_data,
#                                               pen=pg.mkPen(color=(0, 0, 255), width=2),
#                                               name='Observed')
#         legend.addItem(observed_plot, 'Observed')
#
#         fitted_plot = self.plot_widget.plot(observed_x, self.results.fittedvalues,
#                                             pen=pg.mkPen(color=(255, 0, 0), width=2),
#                                             name='Fitted')
#         legend.addItem(fitted_plot, 'Fitted')
#
#         self.plot_widget.setLabel('left', self.target_variable)
#         self.plot_widget.setLabel('bottom', 'Time')
#         self.plot_widget.setTitle('Fitted Values')
#
#     def clear_plot(self):
#         self.plot_widget.clear()
#
#     def output_results(self):
#         if self.results is None:
#             self.Outputs.forecast.send(None)
#             self.Outputs.fitted_values.send(None)
#             self.Outputs.residuals.send(None)
#             return
#
#         # Forecast output
#         forecast = self.results.forecast(steps=self.forecast_steps)
#         forecast_domain = Domain([ContinuousVariable('Forecast')],
#                                  metas=[StringVariable('Time')])
#         forecast_time = np.arange(len(self.model.endog),
#                                   len(self.model.endog) + len(forecast))
#         forecast_table = Table(forecast_domain, np.atleast_2d(forecast).T,
#                                metas=np.atleast_2d(forecast_time).T)
#         self.Outputs.forecast.send(forecast_table)
#
#         # Fitted values output
#         fitted_domain = Domain([ContinuousVariable('Fitted Values')],
#                                metas=[StringVariable('Time')])
#         fitted_time = np.arange(len(self.model.endog))
#         fitted_table = Table(fitted_domain,
#                              np.atleast_2d(self.results.fittedvalues).T,
#                              metas=np.atleast_2d(fitted_time).T)
#         self.Outputs.fitted_values.send(fitted_table)
#
#         # Residuals output
#         residuals_domain = Domain([ContinuousVariable('Residuals')],
#                                   metas=[StringVariable('Time')])
#         residuals_table = Table(residuals_domain,
#                                 np.atleast_2d(self.results.resid).T,
#                                 metas=np.atleast_2d(fitted_time).T)
#         self.Outputs.residuals.send(residuals_table)
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(OWETS).run()