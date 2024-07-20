import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table, Domain, ContinuousVariable
from PyQt5.QtWidgets import QTextEdit


class RNNWidget(widget.OWWidget):
    name = "RNN"
    description = "RNN/LSTM modeling and forecasting"
    icon = "icons/rnn.svg"
    priority = 10

    class Inputs:
        data = widget.Input("Time series data", Table)

    class Outputs:
        forecast = widget.Output("Forecast", Table)
        fitted_values = widget.Output("Fitted Values", Table)
        residuals = widget.Output("Residuals", Table)
        model_summary = widget.Output("Model Summary", str)

    # Widget parameters
    steps = settings.Setting(10)
    seq_length = settings.Setting(10)
    epochs = settings.Setting(50)
    batch_size = settings.Setting(32)

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "RNN Parameters")
        gui.spin(box, self, "steps", 1, 100, label="Forecast Steps")
        gui.spin(box, self, "seq_length", 1, 100, label="Sequence Length")
        gui.spin(box, self, "epochs", 1, 100, label="Epochs")
        gui.spin(box, self, "batch_size", 1, 512, label="Batch Size")

        self.apply_button = gui.button(self.controlArea, self, "Apply", callback=self.apply)

        # Text area for displaying model information
        self.info_box = gui.widgetBox(self.mainArea, "Model Information")
        self.text_output = QTextEdit(readOnly=True)
        self.info_box.layout().addWidget(self.text_output)

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

        # Prepare data for LSTM
        X, y = self.create_sequences(y, self.seq_length)

        # Define LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train model
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Forecast and compute fitted values and residuals
        forecast = self.forecast(model, X[-1], self.steps)
        fitted_values = model.predict(X).flatten()
        residuals = y - fitted_values

        # Create output tables
        domain_forecast = Domain([ContinuousVariable("Forecast")])
        domain_fitted = Domain([ContinuousVariable("Fitted Values")])
        domain_residuals = Domain([ContinuousVariable("Residuals")])

        forecast_table = Table.from_numpy(domain_forecast, forecast.reshape(-1, 1))
        fitted_table = Table.from_numpy(domain_fitted, fitted_values.reshape(-1, 1))
        residuals_table = Table.from_numpy(domain_residuals, residuals.reshape(-1, 1))

        # Display model information in the text area
        info_text = f"Model Summary:\n"
        info_text += f"Sequence Length: {self.seq_length}\n"
        info_text += f"Epochs: {self.epochs}\n"
        info_text += f"Batch Size: {self.batch_size}\n"

        self.text_output.setPlainText(info_text)

        # Send outputs
        self.Outputs.forecast.send(forecast_table)
        self.Outputs.fitted_values.send(fitted_table)
        self.Outputs.residuals.send(residuals_table)
        self.Outputs.model_summary.send(info_text)

    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data)-seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs).reshape(-1, seq_length, 1), np.array(ys)

    def forecast(self, model, last_sequence, steps):
        forecast = []
        current_seq = last_sequence.reshape(1, last_sequence.shape[0], 1)
        for _ in range(steps):
            pred = model.predict(current_seq)
            forecast.append(pred[0, 0])
            current_seq = np.append(current_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
        return np.array(forecast)


if __name__ == "__main__":
    WidgetPreview(RNNWidget).run(Table("iris"))
