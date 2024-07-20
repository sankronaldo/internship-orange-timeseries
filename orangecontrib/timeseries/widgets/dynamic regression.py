import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.signals import Input, Output
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable, StringVariable
from PyQt5.QtWidgets import QGridLayout, QLabel, QSpinBox
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt

class SARIMAXForecaster(OWWidget):
    name = "Dynamic Regression"
    description = "Forecast time series data using multiple exogenous variable"
    icon = "icons/sarimax.svg"
    priority = 20

    class Inputs:
        target = Input("Target Variable", Table)
        exogenous = Input("Exogenous Variables", Table)

    class Outputs:
        forecast = Output("Forecast", Table)
        model = Output("SARIMAX Model", object)
        residuals = Output("Residuals", Table)

    # Widget parameters
    p = settings.Setting(1)
    d = settings.Setting(1)
    q = settings.Setting(1)
    P = settings.Setting(1)
    D = settings.Setting(1)
    Q = settings.Setting(1)
    s = settings.Setting(12)  # Default seasonal period (e.g., 12 for monthly data)
    forecast_steps = settings.Setting(10)
    confidence_interval = settings.Setting(0.95)
    plot_type = settings.Setting(0)  # 0: Forecast, 1: Fitted Values

    want_main_area = True

    class Error(OWWidget.Error):
        fitting_failed = Msg("Failed to fit the SARIMAX model. Check your data and parameters.")
        forecasting_failed = Msg("Failed to generate forecast. Check your exogenous data.")

    def __init__(self):
        super().__init__()
        self.target_data = None
        self.exog_data = None
        self.model = None
        self.results = None
        self.endog = None
        self.exog = None

        # GUI
        box = gui.widgetBox(self.controlArea, "SARIMAX Parameters")
        grid = QGridLayout()
        box.layout().addLayout(grid)

        parameters = [
            ("p", "AR order:"),
            ("d", "Differencing:"),
            ("q", "MA order:"),
            ("P", "Seasonal AR order:"),
            ("D", "Seasonal Differencing:"),
            ("Q", "Seasonal MA order:"),
            ("s", "Seasonal Period:"),
        ]

        for row, (param, label) in enumerate(parameters):
            grid.addWidget(QLabel(label), row, 0)
            sp = QSpinBox(minimum=0, maximum=365, value=getattr(self, param))
            sp.valueChanged.connect(lambda value, param=param: setattr(self, param, value))
            grid.addWidget(sp, row, 1)

        gui.spin(box, self, "forecast_steps", minv=1, maxv=1000, label="Forecast Steps:")
        gui.doubleSpin(box, self, "confidence_interval", 0.5, 0.99, 0.01, label="Confidence Interval:")

        plot_box = gui.widgetBox(self.controlArea, "Plot Selection")
        gui.comboBox(plot_box, self, "plot_type", items=["Forecast", "Fitted Values"],
                     label="Plot Type:", orientation="horizontal", callback=self.update_plot)

        self.forecast_button = gui.button(self.controlArea, self, "Fit and Forecast", callback=self.fit_and_forecast)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

        # Info box for model coefficients and metrics
        self.info_box = gui.widgetBox(self.controlArea, "Model Information")
        self.info_label = gui.widgetLabel(self.info_box, "")

    @Inputs.target
    def set_target(self, data):
        self.target_data = data
        if data is not None:
            self.endog = data.X.ravel()
        else:
            self.endog = None

    @Inputs.exogenous
    def set_exogenous(self, data):
        self.exog_data = data
        if data is not None:
            self.exog = data.X
        else:
            self.exog = None

    def fit_and_forecast(self):
        self.Error.clear()
        if self.endog is None:
            self.Error.fitting_failed("No target data provided.")
            return

        try:
            model = SARIMAX(self.endog, exog=self.exog,
                            order=(self.p, self.d, self.q),
                            seasonal_order=(self.P, self.D, self.Q, self.s))
            self.results = model.fit(disp=False)
            self.model = self.results

            # Prepare exog data for forecasting
            if self.exog is not None:
                forecast_exog = self.exog[-self.forecast_steps:]
                if len(forecast_exog) < self.forecast_steps:
                    self.Error.forecasting_failed("Insufficient exogenous data for forecasting.")
                    return
            else:
                forecast_exog = None

            # Forecast
            forecast = self.results.forecast(steps=self.forecast_steps, exog=forecast_exog)

            # Create Orange Table for forecast output
            domain = Domain([ContinuousVariable("Forecast")])
            forecast_table = Table.from_numpy(domain, forecast.reshape(-1, 1))
            self.Outputs.forecast.send(forecast_table)
            self.Outputs.model.send(self.model)

            self.output_residuals()
            self.update_model_info()
            self.update_plot()

        except Exception as e:
            self.Error.fitting_failed(str(e))
            print(f"SARIMAX fitting failed: {str(e)}")
            print(f"Endog shape: {self.endog.shape}")
            print(f"Exog shape: {self.exog.shape if self.exog is not None else 'None'}")
            print(f"Model parameters: p={self.p}, d={self.d}, q={self.q}, P={self.P}, D={self.D}, Q={self.Q}, s={self.s}")

    def update_model_info(self):
        if self.results is None:
            return

        params = self.results.params
        stderr = self.results.bse

        param_text = "\n".join(f"{param}: {value:.4f} (s.e. {stderr[i]:.4f})"
                               for i, (param, value) in enumerate(zip(self.results.model.param_names, params)))

        metrics = {
            "AIC": self.results.aic,
            "BIC": self.results.bic,
            "Log Likelihood": self.results.llf,
            "MSE": np.mean(self.results.resid**2),
            "MAE": np.mean(np.abs(self.results.resid)),
            "MAPE": np.mean(np.abs(self.results.resid / self.endog)) * 100 if np.all(self.endog != 0) else np.nan,
        }

        metrics_text = "\n".join(f"{key}: {value:.4f}" for key, value in metrics.items())

        self.info_label.setText(f"Model Coefficients:\n{param_text}\n\nMetrics:\n{metrics_text}")

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

        # Prepare exog data for forecasting
        if self.exog is not None:
            forecast_exog = self.exog[-self.forecast_steps:]
            if len(forecast_exog) < self.forecast_steps:
                self.Error.forecasting_failed("Insufficient exogenous data for forecasting.")
                return
        else:
            forecast_exog = None

        forecast = self.results.forecast(steps=self.forecast_steps, exog=forecast_exog)
        ci = self.results.get_forecast(steps=self.forecast_steps, exog=forecast_exog).conf_int(alpha=1 - self.confidence_interval)

        nobs = len(self.endog)
        observed_x = np.arange(nobs)
        forecast_x = np.arange(nobs, nobs + len(forecast))

        observed_plot = self.plot_widget.plot(observed_x, self.endog, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        forecast_plot = self.plot_widget.plot(forecast_x, forecast, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        ci_lower = self.plot_widget.plot(forecast_x, ci[:, 0], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Lower')
        ci_upper = self.plot_widget.plot(forecast_x, ci[:, 1], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Upper')
        legend.addItem(ci_lower, 'CI Lower')
        legend.addItem(ci_upper, 'CI Upper')

        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Forecast')

    def plot_fitted_values(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = len(self.endog)
        observed_x = np.arange(nobs)

        observed_plot = self.plot_widget.plot(observed_x, self.endog, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        fitted_plot = self.plot_widget.plot(observed_x, self.results.fittedvalues, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')
        legend.addItem(fitted_plot, 'Fitted')

        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Fitted Values')

    def output_residuals(self):
        if self.results is None:
            self.Outputs.residuals.send(None)
            return

        residuals = self.results.resid
        domain = Domain([ContinuousVariable('Residuals')],
                        metas=[StringVariable('Time')])
        time_values = np.arange(len(residuals))
        residuals_table = Table(domain, np.atleast_2d(residuals).T,
                                metas=np.atleast_2d(time_values).T)
        self.Outputs.residuals.send(residuals_table)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(SARIMAXForecaster).run()







# import numpy as np
# import pandas as pd
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from Orange.widgets import widget, gui, settings
# from Orange.widgets.utils.signals import Input, Output
# from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
# from PyQt5.QtWidgets import QGridLayout, QLabel, QSpinBox
# from Orange.widgets.widget import OWWidget, Input, Output, Msg
#
#
# class SARIMAXForecaster(OWWidget):
#     name = "SARIMAX Forecaster"
#     description = "Forecast time series data using SARIMAX model"
#     icon = "icons/sarimax.svg"
#     priority = 20
#
#     class Inputs:
#         target = Input("Target Variable", Table)
#         exogenous = Input("Exogenous Variables", Table)
#
#     class Outputs:
#         forecast = Output("Forecast", Table)
#         model = Output("SARIMAX Model", object)
#
#     # Widget parameters
#     p = settings.Setting(1)
#     d = settings.Setting(1)
#     q = settings.Setting(1)
#     P = settings.Setting(1)
#     D = settings.Setting(1)
#     Q = settings.Setting(1)
#     s = settings.Setting(12)  # Default seasonal period (e.g., 12 for monthly data)
#     forecast_steps = settings.Setting(10)
#
#     want_main_area = False
#
#     class Error(OWWidget.Error):
#         fitting_failed = Msg("Failed to fit the SARIMAX model. Check your data and parameters.")
#
#     def __init__(self):
#         super().__init__()
#
#         self.target_data = None
#         self.exog_data = None
#         self.model = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "SARIMAX Parameters")
#         grid = QGridLayout()
#         box.layout().addLayout(grid)
#
#         parameters = [
#             ("p", "AR order:"),
#             ("d", "Differencing:"),
#             ("q", "MA order:"),
#             ("P", "Seasonal AR order:"),
#             ("D", "Seasonal Differencing:"),
#             ("Q", "Seasonal MA order:"),
#             ("s", "Seasonal Period:"),
#         ]
#
#         for row, (param, label) in enumerate(parameters):
#             grid.addWidget(QLabel(label), row, 0)
#             sp = QSpinBox(minimum=0, maximum=365, value=getattr(self, param))
#             sp.valueChanged.connect(lambda value, param=param: setattr(self, param, value))
#             grid.addWidget(sp, row, 1)
#
#         gui.spin(box, self, "forecast_steps", minv=1, maxv=1000, label="Forecast Steps:")
#
#         self.forecast_button = gui.button(self.controlArea, self, "Forecast", callback=self.fit_and_forecast)
#
#     @Inputs.target
#     def set_target(self, data):
#         self.target_data = data
#
#     @Inputs.exogenous
#     def set_exogenous(self, data):
#         self.exog_data = data
#
#     def fit_and_forecast(self):
#         self.Error.fitting_failed.clear()
#         if self.target_data is None:
#             return
#
#         target = self.target_data.X.ravel()
#         exog = self.exog_data.X if self.exog_data is not None else None
#
#         try:
#             model = SARIMAX(target, exog=exog,
#                             order=(self.p, self.d, self.q),
#                             seasonal_order=(self.P, self.D, self.Q, self.s))
#             self.model = model.fit(disp=False)
#
#             # Forecast
#             forecast = self.model.forecast(steps=self.forecast_steps,
#                                            exog=exog[-self.forecast_steps:] if exog is not None else None)
#
#             # Create Orange Table for forecast output
#             domain = Domain([ContinuousVariable("Forecast")])
#             forecast_table = Table.from_numpy(domain, forecast.reshape(-1, 1))
#
#             self.Outputs.forecast.send(forecast_table)
#             self.Outputs.model.send(self.model)
#
#         except Exception as e:
#             self.Error.fitting_failed()
#             print(f"SARIMAX fitting failed: {str(e)}")
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#     WidgetPreview(SARIMAXForecaster).run()