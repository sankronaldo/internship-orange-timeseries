import numpy as np
import pandas as pd
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from darts import TimeSeries
from darts.models import TiDEModel, TSMixerModel
from darts.dataprocessing.transformers import Scaler

class OWAdvancedTimeSeriesModels(widget.OWWidget):
    name = "Advanced MLP models"
    description = "Train TiDE or TSMixer model for time series forecasting using Darts"
    icon = "icons/ow_advancemlp.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        residuals = Output("Residuals", Table)
        forecast = Output("Forecast", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    model_type = Setting(0)  # 0: TiDE, 1: TSMixer
    input_chunk_length = Setting(50)
    output_chunk_length = Setting(12)

    # TiDE specific
    num_encoder_layers = Setting(1)
    num_decoder_layers = Setting(1)
    decoder_output_dim = Setting(16)
    hidden_size = Setting(128)
    temporal_width_past = Setting(4)
    temporal_width_future = Setting(4)
    temporal_decoder_hidden = Setting(32)
    use_layer_norm = Setting(False)

    # TSMixer specific
    hidden_size_mixer = Setting(64)
    ff_size = Setting(64)
    num_blocks = Setting(2)
    normalize_before = Setting(False)

    # Common
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

        # Model selection
        model_box = gui.widgetBox(self.controlArea, "Model Selection")
        gui.comboBox(model_box, self, "model_type", items=["TiDE", "TSMixer"],
                     label="Model Type:", orientation="horizontal", callback=self.on_model_type_changed)

        # Model parameters
        self.params_box = gui.widgetBox(self.controlArea, "Model Parameters")
        self.setup_model_params()

        # Common parameters
        common_box = gui.widgetBox(self.controlArea, "Common Parameters")
        gui.spin(common_box, self, "input_chunk_length", 1, 100, label="Input Chunk Length:",
                 callback=self.on_param_changed)
        gui.spin(common_box, self, "output_chunk_length", 1, 50, label="Output Chunk Length:",
                 callback=self.on_param_changed)
        gui.doubleSpin(common_box, self, "dropout", 0, 0.5, 0.01, label="Dropout:", callback=self.on_param_changed)
        gui.spin(common_box, self, "epochs", 1, 1000, label="Epochs:", callback=self.on_param_changed)
        gui.spin(common_box, self, "batch_size", 1, 128, label="Batch Size:", callback=self.on_param_changed)
        gui.doubleSpin(common_box, self, "learning_rate", 1e-5, 1e-1, 1e-5, label="Learning Rate:",
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

    def setup_model_params(self):
        for i in reversed(range(self.params_box.layout().count())):
            self.params_box.layout().itemAt(i).widget().setParent(None)

        if self.model_type == 0:  # TiDE
            gui.spin(self.params_box, self, "num_encoder_layers", 1, 5, label="Num Encoder Layers:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "num_decoder_layers", 1, 5, label="Num Decoder Layers:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "decoder_output_dim", 1, 64, label="Decoder Output Dim:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "hidden_size", 1, 256, label="Hidden Size:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "temporal_width_past", 1, 10, label="Temporal Width Past:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "temporal_width_future", 1, 10, label="Temporal Width Future:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "temporal_decoder_hidden", 1, 64, label="Temporal Decoder Hidden:",
                     callback=self.on_param_changed)
            gui.checkBox(self.params_box, self, "use_layer_norm", label="Use Layer Norm:",
                         callback=self.on_param_changed)
        elif self.model_type == 1:  # TSMixer
            gui.spin(self.params_box, self, "hidden_size_mixer", 1, 256, label="Hidden Size:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "ff_size", 1, 256, label="FF Size:",
                     callback=self.on_param_changed)
            gui.spin(self.params_box, self, "num_blocks", 1, 10, label="Number of Blocks:",
                     callback=self.on_param_changed)
            gui.checkBox(self.params_box, self, "normalize_before", label="Normalize Before:",
                         callback=self.on_param_changed)

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

    def on_model_type_changed(self):
        self.setup_model_params()

    def on_param_changed(self):
        pass  # We'll only train when the user clicks the train button

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

        common_params = {
            "input_chunk_length": self.input_chunk_length,
            "output_chunk_length": self.output_chunk_length,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "n_epochs": self.epochs,
            "optimizer_kwargs": {"lr": self.learning_rate},
            "pl_trainer_kwargs": {"accelerator": "cpu"}
        }

        if self.model_type == 0:  # TiDE
            self.model = TiDEModel(
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                decoder_output_dim=self.decoder_output_dim,
                hidden_size=self.hidden_size,
                temporal_width_past=self.temporal_width_past,
                temporal_width_future=self.temporal_width_future,
                temporal_decoder_hidden=self.temporal_decoder_hidden,
                use_layer_norm=self.use_layer_norm,
                **common_params
            )
        elif self.model_type == 1:  # TSMixer
            self.model = TSMixerModel(
                hidden_size=self.hidden_size_mixer,
                ff_size=self.ff_size,
                num_blocks=self.num_blocks,
                activation="ReLU",  # Fixed to ReLU
                normalize_before=self.normalize_before,
                **common_params
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

        model_type = ["TiDE", "TSMixer"][self.model_type]
        model_info = f"{model_type} Model:\n"

        if self.model_type == 0:  # TiDE
            model_info += (
                f"  Num Encoder Layers: {self.num_encoder_layers}\n"
                f"  Num Decoder Layers: {self.num_decoder_layers}\n"
                f"  Decoder Output Dim: {self.decoder_output_dim}\n"
                f"  Hidden Size: {self.hidden_size}\n"
                f"  Temporal Width Past: {self.temporal_width_past}\n"
                f"  Temporal Width Future: {self.temporal_width_future}\n"
                f"  Temporal Decoder Hidden: {self.temporal_decoder_hidden}\n"
                f"  Use Layer Norm: {self.use_layer_norm}\n"
            )
        elif self.model_type == 1:  # TSMixer
            model_info += (
                f"  Hidden Size: {self.hidden_size_mixer}\n"
                f"  FF Size: {self.ff_size}\n"
                f"  Number of Blocks: {self.num_blocks}\n"
                # f"  Activation: {self.activation}\n"
                # f"  Norm Type: {self.norm_type}\n"
                f"  Normalize Before: {self.normalize_before}\n"
            )

        model_info += (
            f"  Input Chunk Length: {self.input_chunk_length}\n"
            f"  Output Chunk Length: {self.output_chunk_length}\n"
            f"  Dropout: {self.dropout}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Learning Rate: {self.learning_rate}\n\n"
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
    WidgetPreview(OWAdvancedTimeSeriesModels).run()