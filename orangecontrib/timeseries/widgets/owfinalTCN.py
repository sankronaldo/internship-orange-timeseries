import numpy as np
import pandas as pd
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler

class OWTCN(widget.OWWidget):
    name = "TCN-Temporal Convolutional Network"
    description = "Train TCN model for time series forecasting using Darts"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        residuals = Output("Residuals", Table)
        forecast = Output("Forecast", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    input_chunk_length = Setting(50)
    output_chunk_length = Setting(12)
    num_filters = Setting(64)
    kernel_size = Setting(3)
    num_layers = Setting(3)
    dilation_base = Setting(2)
    dropout = Setting(0.1)
    epochs = Setting(100)
    batch_size = Setting(32)
    learning_rate = Setting(1e-3)
    forecast_steps = Setting(24)
    plot_type = Setting(0)  # 0: Forecast, 1: Fitted Values

    def __init__(self):
        super().__init__()

        self.data = None
        self.model = None
        self.scaler = Scaler()
        self.scaled_series = None
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

        # TCN parameters
        tcn_box = gui.widgetBox(self.controlArea, "TCN Parameters")
        gui.spin(tcn_box, self, "input_chunk_length", 1, 100, label="Input Chunk Length:", callback=self.on_param_changed)
        gui.spin(tcn_box, self, "output_chunk_length", 1, 50, label="Output Chunk Length:", callback=self.on_param_changed)
        gui.spin(tcn_box, self, "num_filters", 1, 256, label="Number of Filters:", callback=self.on_param_changed)
        gui.spin(tcn_box, self, "kernel_size", 1, 10, label="Kernel Size:", callback=self.on_param_changed)
        gui.spin(tcn_box, self, "num_layers", 1, 10, label="Number of Layers:", callback=self.on_param_changed)
        gui.spin(tcn_box, self, "dilation_base", 1, 4, label="Dilation Base:", callback=self.on_param_changed)
        gui.doubleSpin(tcn_box, self, "dropout", 0, 0.5, 0.01, label="Dropout:", callback=self.on_param_changed)
        gui.spin(tcn_box, self, "epochs", 1, 1000, label="Epochs:", callback=self.on_param_changed)
        gui.spin(tcn_box, self, "batch_size", 1, 128, label="Batch Size:", callback=self.on_param_changed)
        gui.doubleSpin(tcn_box, self, "learning_rate", 1e-5, 1e-1, 1e-5, label="Learning Rate:", callback=self.on_param_changed)

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

        # Create a Darts TimeSeries with a simple numeric index
        self.series = TimeSeries.from_values(self.original_values)
        self.scaled_series = self.scaler.fit_transform(self.series)

        # Update info label
        self.info_label.setText(f"{len(self.data)} instances on input.")

    def train_model(self):
        if self.scaled_series is None:
            return

        self.model = TCNModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            dilation_base=self.dilation_base,
            dropout=self.dropout,
            batch_size=self.batch_size,
            n_epochs=self.epochs,
            optimizer_kwargs={"lr": self.learning_rate},
            pl_trainer_kwargs={"accelerator": "cpu"}  # Use CPU instead of GPU
        )

        self.model.fit(self.scaled_series)

        # Generate predictions (fitted values)
        self.predictions = self.model.historical_forecasts(
            self.scaled_series,
            start=self.input_chunk_length,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False
        )
        self.predictions = self.scaler.inverse_transform(self.predictions)

        # Generate forecast
        self.forecast = self.model.predict(n=self.forecast_steps)
        self.forecast = self.scaler.inverse_transform(self.forecast)

        self.update_plot()
        self.update_model_info()
        self.output_results()

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
        forecast_plot = self.plot_widget.plot(forecast_x, self.forecast.values().flatten(),
                                              pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
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
        observed_plot = self.plot_widget.plot(observed_x, self.original_values,
                                              pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        # Plot fitted values
        fitted_plot = self.plot_widget.plot(observed_x[self.input_chunk_length:], self.predictions.values().flatten(),
                                            pen=pg.mkPen(color=(255, 0, 0), width=2), name='Fitted')
        legend.addItem(fitted_plot, 'Fitted')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Fitted Values')

    def update_model_info(self):
        if self.model is None:
            return

        mse = mean_squared_error(self.original_values[self.input_chunk_length:], self.predictions.values().flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.original_values[self.input_chunk_length:], self.predictions.values().flatten())

        model_info = (
            f"TCN Model:\n"
            f"  Input chunk length: {self.input_chunk_length}\n"
            f"  Output chunk length: {self.output_chunk_length}\n"
            f"  Number of filters: {self.num_filters}\n"
            f"  Kernel size: {self.kernel_size}\n"
            f"  Number of layers: {self.num_layers}\n"
            f"  Dilation base: {self.dilation_base}\n"
            f"  Dropout: {self.dropout}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Learning rate: {self.learning_rate}\n\n"
            f"Performance Metrics:\n"
            f"  MSE: {mse:.4f}\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  MAE: {mae:.4f}"
        )

        self.info_label.setText(model_info)

    def output_results(self):
        self.output_residuals()
        self.output_forecast()

    def output_residuals(self):
        if self.predictions is None:
            self.Outputs.residuals.send(None)
            return

        residuals = self.original_values[self.input_chunk_length:] - self.predictions.values().flatten()

        # Create a domain with only the 'Residuals' variable
        domain = Domain([ContinuousVariable('Residuals')])

        # Create the Table without any meta attributes
        residuals_table = Table.from_numpy(domain, residuals.reshape(-1, 1))

        self.Outputs.residuals.send(residuals_table)

    def output_forecast(self):
        if self.forecast is None:
            self.Outputs.forecast.send(None)
            return

        # Create a domain with only the 'Forecast' variable
        domain = Domain([ContinuousVariable('Forecast')])

        # Create the Table without any meta attributes
        forecast_table = Table.from_numpy(domain, self.forecast.values().reshape(-1, 1))

        self.Outputs.forecast.send(forecast_table)

    def clear_plot(self):
        self.plot_widget.clear()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWTCN).run()