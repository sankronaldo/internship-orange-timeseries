import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class OWNNAR(widget.OWWidget):
    name = "NNAR Model"
    description = "Fit Neural Network AutoRegression model and visualize results"
    icon = "icons/ow_nnar.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        forecast = Output("Forecast", Table)
        fitted_values = Output("Fitted Values", Table)
        residuals = Output("Residuals", Table)

    want_main_area = True

    # Settings
    p = Setting(1)  # Number of lagged inputs (non-seasonal)
    P = Setting(0)  # Number of seasonal lagged inputs
    k = Setting(1)  # Number of nodes in the hidden layer
    forecast_steps = Setting(10)
    random_seed = Setting(42)
    seasonal_period = Setting(12)  # Default to monthly data
    target_variable = Setting("")
    plot_type = Setting(0)  # 0: Forecast, 1: Fitted Values

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.model = None
        self.scaler = None
        self.results = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        seed_box = gui.widgetBox(self.controlArea, "Random Seed")
        gui.spin(seed_box, self, "random_seed", 0, 1000000, label="Random Seed:", callback=self.on_param_changed)

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # NNAR parameters
        nnar_box = gui.widgetBox(self.controlArea, "NNAR Parameters")
        gui.spin(nnar_box, self, "p", 1, 20, label="p (Non-seasonal lags):", callback=self.on_param_changed)
        gui.spin(nnar_box, self, "P", 0, 20, label="P (Seasonal lags):", callback=self.on_param_changed)
        gui.spin(nnar_box, self, "k", 1, 20, label="k (Hidden Nodes):", callback=self.on_param_changed)
        gui.spin(nnar_box, self, "seasonal_period", 1, 365, label="Seasonal Period:", callback=self.on_param_changed)

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

    def on_param_changed(self):
        self.fit_model()

    def on_plot_type_changed(self):
        self.update_plot()

    def prepare_data(self, y):
        max_lag = max(self.p, self.P * self.seasonal_period)
        X = []
        for i in range(len(y) - max_lag):
            row = []
            for j in range(1, self.p + 1):
                row.append(y[i + max_lag - j])
            for j in range(1, self.P + 1):
                row.append(y[i + max_lag - j * self.seasonal_period])
            X.append(row)
        return np.array(X), y[max_lag:]

    def fit_model(self):
        if self.data is None or not self.target_variable:
            return

        value_var = self.data.domain[self.target_variable]
        y = self.data.get_column(value_var)

        # Prepare data
        X, y_train = self.prepare_data(y)

        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit NNAR model
        self.model = MLPRegressor(hidden_layer_sizes=(self.k,), max_iter=10000000000,random_state=self.random_seed)
        self.model.fit(X_scaled, y_train)

        # Generate fitted values and residuals
        fitted_scaled = self.model.predict(X_scaled)
        self.fitted_values = np.full(len(y), np.nan)
        self.fitted_values[-len(fitted_scaled):] = fitted_scaled
        self.residuals = y - self.fitted_values

        # Generate forecast
        last_window = X[-1:].copy()
        self.forecast = []
        for _ in range(self.forecast_steps):
            last_window_scaled = self.scaler.transform(last_window)
            next_pred = self.model.predict(last_window_scaled)[0]
            self.forecast.append(next_pred)
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1] = next_pred
            if self.P > 0:  # Update seasonal lags if present
                seasonal_index = self.p + self.P - 1
                last_window[0, seasonal_index] = last_window[0, self.p - 1]

        self.forecast = np.array(self.forecast)

        self.update_plot()
        self.update_model_info()
        self.output_results()

    def update_model_info(self):
        if self.model is None:
            return

        mse = np.mean(self.residuals[~np.isnan(self.residuals)]**2)
        mae = np.mean(np.abs(self.residuals[~np.isnan(self.residuals)]))
        mape = np.mean(np.abs(self.residuals[~np.isnan(self.residuals)] / self.fitted_values[~np.isnan(self.fitted_values)])) * 100

        metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.4f}%"
        self.info_label.setText(f"NNAR({self.p},{self.P},{self.k}) Model\nSeasonal Period: {self.seasonal_period}\n\nMetrics:\n{metrics_text}")

    def update_plot(self):
        self.plot_widget.clear()

        if self.model is None:
            return

        if self.plot_type == 0:  # Forecast
            self.plot_forecast()
        elif self.plot_type == 1:  # Fitted Values
            self.plot_fitted_values()

    def plot_forecast(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = len(self.data)
        observed_x = np.arange(nobs)
        forecast_x = np.arange(nobs, nobs + len(self.forecast))

        # Plot observed data
        observed_data = self.data.get_column(self.target_variable)
        observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        # Plot forecast
        forecast_plot = self.plot_widget.plot(forecast_x, self.forecast, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Forecast')

    def plot_fitted_values(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = len(self.data)
        observed_x = np.arange(nobs)

        # Plot observed data
        observed_data = self.data.get_column(self.target_variable)
        observed_plot = self.plot_widget.plot(observed_x, observed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        # Plot fitted values
        fitted_plot = self.plot_widget.plot(observed_x, self.fitted_values, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')
        legend.addItem(fitted_plot, 'Fitted')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Fitted Values')

    def clear_plot(self):
        self.plot_widget.clear()

    def output_results(self):
        if self.model is None:
            self.Outputs.forecast.send(None)
            self.Outputs.fitted_values.send(None)
            self.Outputs.residuals.send(None)
            return

        # Output forecast
        forecast_domain = Domain([ContinuousVariable('Forecast')],
                                 metas=[StringVariable('Time')])
        forecast_time = np.arange(len(self.data), len(self.data) + len(self.forecast))
        forecast_table = Table(forecast_domain, self.forecast.reshape(-1, 1),
                               metas=forecast_time.reshape(-1, 1))
        self.Outputs.forecast.send(forecast_table)

        # Output fitted values
        fitted_domain = Domain([ContinuousVariable('Fitted')],
                               metas=[StringVariable('Time')])
        fitted_time = np.arange(len(self.fitted_values))
        fitted_table = Table(fitted_domain, self.fitted_values.reshape(-1, 1),
                             metas=fitted_time.reshape(-1, 1))
        self.Outputs.fitted_values.send(fitted_table)

        # Output residuals
        residuals_domain = Domain([ContinuousVariable('Residuals')],
                                  metas=[StringVariable('Time')])
        residuals_table = Table(residuals_domain, self.residuals.reshape(-1, 1),
                                metas=fitted_time.reshape(-1, 1))
        self.Outputs.residuals.send(residuals_table)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWNNAR).run()
