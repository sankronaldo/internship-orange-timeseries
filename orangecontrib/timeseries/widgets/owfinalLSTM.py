import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


class OWLSTM(widget.OWWidget):
    name = "LSTM Forecaster"
    description = "Train LSTM model for time series forecasting"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        residuals = Output("Residuals", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    lookback = Setting(3)
    lstm_units = Setting(50)
    epochs = Setting(100)
    batch_size = Setting(32)
    learning_rate = Setting(0.001)
    forecast_steps = Setting(10)
    plot_type = Setting(0)  # 0: Forecast, 1: Fitted Values

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = None
        self.predictions = None
        self.forecast = None
        self.original_values = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", callback=self.on_target_variable_changed)

        # LSTM parameters
        lstm_box = gui.widgetBox(self.controlArea, "LSTM Parameters")
        gui.spin(lstm_box, self, "lookback", 1, 50, label="Lookback:", callback=self.on_param_changed)
        gui.spin(lstm_box, self, "lstm_units", 1, 500, label="LSTM Units:", callback=self.on_param_changed)
        gui.spin(lstm_box, self, "epochs", 1, 1000, label="Epochs:", callback=self.on_param_changed)
        gui.spin(lstm_box, self, "batch_size", 1, 128, label="Batch Size:", callback=self.on_param_changed)
        gui.doubleSpin(lstm_box, self, "learning_rate", 0.0001, 0.1, 0.0001, label="Learning Rate:",
                       callback=self.on_param_changed)

        # Forecast settings
        forecast_box = gui.widgetBox(self.controlArea, "Forecast Settings")
        gui.spin(forecast_box, self, "forecast_steps", 1, 100, label="Forecast Steps:", callback=self.on_param_changed)

        # Plot type selection
        plot_box = gui.widgetBox(self.controlArea, "Plot Selection")
        gui.comboBox(plot_box, self, "plot_type", items=["Forecast", "Fitted Values"],
                     label="Plot Type:", orientation="horizontal", callback=self.on_plot_type_changed)

        # Train button
        self.train_button = gui.button(self.controlArea, self, "Train Model", callback=self.train_model)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

    @Inputs.time_series
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = data.time_variable if hasattr(data, 'time_variable') else None

            # Update target variable combo box options
            self.target_combo.clear()
            self.target_combo.addItem("")
            for var in data.domain.variables:
                if var.is_continuous and not isinstance(var, TimeVariable):
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
        self.prepare_data()

    def on_param_changed(self):
        if self.data is not None and self.target_variable:
            self.train_model()

    def on_plot_type_changed(self):
        self.update_plot()

    def prepare_data(self):
        if self.data is None or not self.target_variable:
            return

        value_var = self.data.domain[self.target_variable]
        self.original_values = self.data.get_column(value_var)
        self.scaled_data = self.scaler.fit_transform(self.original_values.reshape(-1, 1)).flatten()

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def train_model(self):
        if self.scaled_data is None:
            return

        X, y = self.create_sequences(self.scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.lookback, 1), return_sequences=True),
            LSTM(self.lstm_units),
            Dense(32),
            Dense(64),
            Dense(1)
        ])

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self.predictions = self.model.predict(X)
        self.predictions = self.scaler.inverse_transform(self.predictions).flatten()

        self.forecast = self.generate_forecast()

        self.update_plot()
        self.update_model_info()
        self.output_residuals()

    def generate_forecast(self):
        last_sequence = self.scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
        forecast = []

        for _ in range(self.forecast_steps):
            next_pred = self.model.predict(last_sequence)
            forecast.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred[0, 0]

        return self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    def update_plot(self):
        self.plot_widget.clear()
        if self.predictions is None:
            return

        if self.plot_type == 0:  # Forecast
            self.plot_forecast()
        elif self.plot_type == 1:  # Fitted Values
            self.plot_fitted_values()

    def plot_forecast(self):
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = len(self.original_values)
        observed_x = np.arange(nobs)
        forecast_x = np.arange(nobs, nobs + len(self.forecast))

        # Plot observed data
        observed_plot = self.plot_widget.plot(observed_x, self.original_values,
                                              pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        # Plot forecast
        forecast_plot = self.plot_widget.plot(forecast_x, self.forecast, pen=pg.mkPen(color=(255, 0, 0), width=2),
                                              name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Forecast')

    def plot_fitted_values(self):
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        nobs = len(self.original_values)
        observed_x = np.arange(nobs)

        # Plot observed data
        observed_plot = self.plot_widget.plot(observed_x[self.lookback:], self.original_values[self.lookback:],
                                              pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        # Plot fitted values
        fitted_plot = self.plot_widget.plot(observed_x[self.lookback:], self.predictions,
                                            pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')
        legend.addItem(fitted_plot, 'Fitted')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Fitted Values')

    def update_model_info(self):
        if self.model is None:
            return

        mse = mean_squared_error(self.original_values[self.lookback:], self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.original_values[self.lookback:], self.predictions)

        model_info = (
            f"LSTM Model:\n"
            f"  Input shape: ({self.lookback}, 1)\n"
            f"  LSTM units: {self.lstm_units}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Learning rate: {self.learning_rate}\n\n"
            f"Performance Metrics:\n"
            f"  MSE: {mse:.4f}\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  MAE: {mae:.4f}"
        )

        self.info_label.setText(model_info)

    def output_residuals(self):
        if self.predictions is None:
            self.Outputs.residuals.send(None)
            return

        residuals = self.original_values[self.lookback:] - self.predictions

        # Create a domain with only the 'Residuals' variable
        domain = Domain([ContinuousVariable('Residuals')])

        # Create the Table without any meta attributes
        residuals_table = Table.from_numpy(domain, residuals.reshape(-1, 1))

        self.Outputs.residuals.send(residuals_table)

    def clear_plot(self):
        self.plot_widget.clear()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLSTM).run()