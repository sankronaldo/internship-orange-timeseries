import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtWidgets import QTabWidget
from PyQt5.QtCore import Qt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class OWARIMASARIMAModel(widget.OWWidget):
    name = "UnitRoot"
    description = "Fit ARIMA or SARIMA model and visualize results"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        residuals = Output("Residuals", Table)

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

        # Fit button
        self.fit_button = gui.button(self.controlArea, self, "Fit Model", callback=self.fit_model)

        # Set up the main area with tab widget
        self.tab_widget = QTabWidget()
        self.mainArea.layout().addWidget(self.tab_widget)

        # Forecast Plot
        self.forecast_plot_widget = PlotWidget(background="w")
        self.tab_widget.addTab(self.forecast_plot_widget, "Forecast")

        # Fitted Values Plot
        self.fitted_plot_widget = PlotWidget(background="w")
        self.tab_widget.addTab(self.fitted_plot_widget, "Fitted Values")

        # Inverse Roots Plot
        self.roots_plot_widget = PlotWidget(background="w")
        self.tab_widget.addTab(self.roots_plot_widget, "Inverse Roots")

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
            self.clear_plots()

    def on_target_variable_changed(self):
        self.target_variable = self.target_combo.currentText()
        self.fit_model()

    def on_model_type_changed(self):
        self.sarima_box.setVisible(self.model_type == 1)
        self.fit_model()

    def on_param_changed(self):
        self.fit_model()

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

        try:
            self.results = self.model.fit()
            self.update_plot()
            self.update_model_info()
            self.output_residuals()
        except Exception as e:
            self.info_label.setText(f"Error fitting model: {str(e)}")
            self.clear_plots()

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
        self.clear_plots()

        if self.results is None:
            return

        self.plot_forecast()
        self.plot_fitted_values()
        self.plot_inverse_roots()

    def plot_forecast(self):
        forecast = self.results.forecast(steps=self.forecast_steps)
        ci = self.results.get_forecast(steps=self.forecast_steps).conf_int(alpha=1 - self.confidence_interval)

        nobs = self.model.nobs
        observed_x = np.arange(nobs)
        forecast_x = np.arange(nobs, nobs + len(forecast))

        observed_data = np.ravel(self.model.endog)

        self.forecast_plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        self.forecast_plot_widget.plot(forecast_x, forecast, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
        self.forecast_plot_widget.plot(forecast_x, ci[:, 0], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Lower')
        self.forecast_plot_widget.plot(forecast_x, ci[:, 1], pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Upper')

        self.forecast_plot_widget.setLabel('left', self.target_variable)
        self.forecast_plot_widget.setLabel('bottom', 'Time')
        self.forecast_plot_widget.setTitle('Forecast')
        self.add_legend(self.forecast_plot_widget)

    def plot_fitted_values(self):
        nobs = self.model.nobs
        observed_x = np.arange(nobs)
        observed_data = np.ravel(self.model.endog)

        self.fitted_plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        self.fitted_plot_widget.plot(observed_x, self.results.fittedvalues, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')

        self.fitted_plot_widget.setLabel('left', self.target_variable)
        self.fitted_plot_widget.setLabel('bottom', 'Time')
        self.fitted_plot_widget.setTitle('Fitted Values')
        self.add_legend(self.fitted_plot_widget)

    def plot_inverse_roots(self):
        self.roots_plot_widget.clear()

        # Extract AR, MA, SAR, and SMA coefficients
        ar_params = self.results.arparams if hasattr(self.results, 'arparams') else []
        ma_params = self.results.maparams if hasattr(self.results, 'maparams') else []
        sar_params = self.results.seasonalarparams if hasattr(self.results, 'seasonalarparams') else []
        sma_params = self.results.seasonalmaparams if hasattr(self.results, 'seasonalmaparams') else []

        # Compute inverse roots
        ar_roots = np.roots(np.r_[1, -ar_params]) if len(ar_params) > 0 else []
        ma_roots = np.roots(np.r_[1, ma_params]) if len(ma_params) > 0 else []
        sar_roots = np.roots(np.r_[1, -sar_params]) if len(sar_params) > 0 else []
        sma_roots = np.roots(np.r_[1, sma_params]) if len(sma_params) > 0 else []

        # Plot unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        self.roots_plot_widget.plot(np.cos(theta), np.sin(theta), pen=pg.mkPen(color=(200, 200, 200), width=2))

        # Plot x and y axes
        self.roots_plot_widget.plot([-1.1, 1.1], [0, 0], pen=pg.mkPen(color=(100, 100, 100), width=1))
        self.roots_plot_widget.plot([0, 0], [-1.1, 1.1], pen=pg.mkPen(color=(100, 100, 100), width=1))

        # Plot AR roots
        if len(ar_roots) > 0:
            self.roots_plot_widget.plot(ar_roots.real, ar_roots.imag, pen=None, symbol='o',
                                        symbolPen=None, symbolBrush=(255, 0, 0, 120), symbolSize=10, name='AR Roots')

        # Plot MA roots
        if len(ma_roots) > 0:
            self.roots_plot_widget.plot(ma_roots.real, ma_roots.imag, pen=None, symbol='s',
                                        symbolPen=None, symbolBrush=(0, 0, 255, 120), symbolSize=10, name='MA Roots')

        # Plot SAR roots
        if len(sar_roots) > 0:
            self.roots_plot_widget.plot(sar_roots.real, sar_roots.imag, pen=None, symbol='t',
                                        symbolPen=None, symbolBrush=(0, 255, 0, 120), symbolSize=10, name='SAR Roots')

        # Plot SMA roots
        if len(sma_roots) > 0:
            self.roots_plot_widget.plot(sma_roots.real, sma_roots.imag, pen=None, symbol='d',
                                        symbolPen=None, symbolBrush=(255, 0, 255, 120), symbolSize=10, name='SMA Roots')

        # Add legend items
        legend = self.roots_plot_widget.addLegend()
        legend.addItem(pg.ScatterPlotItem(symbol='o', size=10, brush=(255, 0, 0, 120)), 'AR Roots')
        legend.addItem(pg.ScatterPlotItem(symbol='s', size=10, brush=(0, 0, 255, 120)), 'MA Roots')
        legend.addItem(pg.ScatterPlotItem(symbol='t', size=10, brush=(0, 255, 0, 120)), 'SAR Roots')
        legend.addItem(pg.ScatterPlotItem(symbol='d', size=10, brush=(255, 0, 255, 120)), 'SMA Roots')

        self.roots_plot_widget.setAspectLocked(True)
        self.roots_plot_widget.setXRange(-1.1, 1.1)
        self.roots_plot_widget.setYRange(-1.1, 1.1)
        self.roots_plot_widget.setLabel('left', 'Imaginary')
        self.roots_plot_widget.setLabel('bottom', 'Real')
        self.roots_plot_widget.setTitle('Inverse Roots')

        # Add grid
        self.roots_plot_widget.showGrid(x=True, y=True, alpha=0.3)

    def add_legend(self, plot_widget):
        if plot_widget.plotItem.legend is None:
            plot_widget.addLegend()

    def clear_plots(self):
        self.forecast_plot_widget.clear()
        self.fitted_plot_widget.clear()
        self.roots_plot_widget.clear()

        for plot_widget in [self.forecast_plot_widget, self.fitted_plot_widget, self.roots_plot_widget]:
            if plot_widget.plotItem.legend is not None:
                plot_widget.plotItem.legend.clear()

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

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWARIMASARIMAModel).run()