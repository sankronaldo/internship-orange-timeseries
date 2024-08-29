import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime


class OWProphetForecaster(widget.OWWidget):
    name = "Prophet Forecaster"
    description = "Forecast time series using Facebook Prophet"
    icon = "icons/ow_prophet.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        forecast = Output("Forecast", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    start_date = Setting("")
    frequency = Setting(0)
    forecast_periods = Setting(30)

    # Prophet parameters
    changepoint_prior_scale = Setting(0.05)
    seasonality_prior_scale = Setting(10.0)
    holidays_prior_scale = Setting(10.0)
    seasonality_mode = Setting(0)  # 0: additive, 1: multiplicative

    FREQ_OPTIONS = [
        ("Daily", "D"),
        ("Weekly", "W"),
        ("Monthly", "M"),
        ("Quarterly", "Q"),
        ("Yearly", "Y")
    ]

    SEASONALITY_MODES = ["additive", "multiplicative"]

    def __init__(self):
        super().__init__()

        self.data = None
        self.model = None
        self.forecast = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Data Settings")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            callback=self.on_target_variable_changed)

        gui.lineEdit(box, self, "start_date", label="Start Date (YYYY-MM-DD):",
                     callback=self.on_start_date_changed)

        gui.comboBox(box, self, "frequency", label="Frequency:",
                     items=[freq[0] for freq in self.FREQ_OPTIONS],
                     callback=self.on_frequency_changed)

        gui.spin(box, self, "forecast_periods", 1, 1000, label="Forecast Periods:",
                 callback=self.on_forecast_periods_changed)

        # Prophet parameters box
        prophet_box = gui.widgetBox(self.controlArea, "Prophet Parameters")
        gui.doubleSpin(prophet_box, self, "changepoint_prior_scale", 0.001, 1.0, 0.001,
                       label="Changepoint Prior Scale:", decimals=3,
                       callback=self.on_prophet_param_changed)
        gui.doubleSpin(prophet_box, self, "seasonality_prior_scale", 0.01, 100.0, 0.1,
                       label="Seasonality Prior Scale:", decimals=2,
                       callback=self.on_prophet_param_changed)
        gui.doubleSpin(prophet_box, self, "holidays_prior_scale", 0.01, 100.0, 0.1,
                       label="Holidays Prior Scale:", decimals=2,
                       callback=self.on_prophet_param_changed)
        gui.comboBox(prophet_box, self, "seasonality_mode", label="Seasonality Mode:",
                     items=self.SEASONALITY_MODES,
                     callback=self.on_prophet_param_changed)

        self.forecast_button = gui.button(self.controlArea, self, "Forecast", callback=self.run_forecast)

        # Info box for metrics and parameters
        self.info_box = gui.widgetBox(self.controlArea, "Model Info")
        self.info_text = gui.widgetLabel(self.info_box, "")

        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

    @Inputs.time_series
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")

            self.target_combo.clear()
            self.target_combo.addItems(
                [var.name for var in data.domain.variables if var.is_continuous]
            )
            if self.target_combo.count() > 0:
                self.target_variable = self.target_combo.itemText(0)
        else:
            self.data = None
            self.info_label.setText("No data on input.")
            self.clear_plot()

    def on_target_variable_changed(self):
        self.run_forecast()

    def on_start_date_changed(self):
        self.run_forecast()

    def on_frequency_changed(self):
        self.run_forecast()

    def on_forecast_periods_changed(self):
        self.run_forecast()

    def on_prophet_param_changed(self):
        self.run_forecast()

    def run_forecast(self):
        if self.data is None or not self.target_variable or not self.start_date:
            self.info_label.setText("Please select a target variable and set a start date.")
            return

        target_var = self.data.domain[self.target_variable]
        freq = self.FREQ_OPTIONS[self.frequency][1]

        try:
            start_date = pd.to_datetime(self.start_date)
            end_date = start_date + pd.Timedelta(days=len(self.data) - 1)
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

            if len(date_range) != len(self.data):
                date_range = pd.date_range(start=start_date, periods=len(self.data), freq=freq)

            df = pd.DataFrame({
                'ds': date_range,
                'y': self.data.get_column(target_var)
            })

            self.model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                seasonality_mode=self.SEASONALITY_MODES[self.seasonality_mode]
            )
            self.model.fit(df)

            future = self.model.make_future_dataframe(periods=self.forecast_periods, freq=freq)
            self.forecast = self.model.predict(future)

            self.update_plot()
            self.output_forecast()
            self.update_info_box()
        except Exception as e:
            self.info_label.setText(f"Error during forecasting: {str(e)}")

    def update_plot(self):
        self.plot_widget.clear()
        if self.forecast is None:
            return

        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        timestamps = self.forecast['ds'].astype(int) / 10 ** 9

        observed_data = pd.merge(self.forecast[['ds']], self.model.history[['ds', 'y']], on='ds', how='left')
        observed_plot = self.plot_widget.plot(observed_data['ds'].astype(int) / 10 ** 9, observed_data['y'],
                                              pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        forecast_plot = self.plot_widget.plot(timestamps, self.forecast['yhat'],
                                              pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        ci_plot = pg.FillBetweenItem(
            pg.PlotDataItem(timestamps, self.forecast['yhat_lower']),
            pg.PlotDataItem(timestamps, self.forecast['yhat_upper']),
            brush=pg.mkBrush(color=(200, 200, 255, 100))
        )
        self.plot_widget.addItem(ci_plot)

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Date')
        self.plot_widget.setTitle('Prophet Forecast')

        axis = self.plot_widget.getAxis('bottom')
        axis.setScale(1)
        axis.tickFormatter = lambda x, pos: pd.to_datetime(x, unit='s').strftime('%Y-%m-%d')

    def output_forecast(self):
        if self.forecast is None:
            self.Outputs.forecast.send(None)
            return

        domain = Domain([
            TimeVariable('Date'),
            ContinuousVariable('y'),
            ContinuousVariable('yhat'),
            ContinuousVariable('yhat_lower'),
            ContinuousVariable('yhat_upper')
        ])

        observed_y = pd.merge(self.forecast[['ds']], self.model.history[['ds', 'y']], on='ds', how='left')['y']

        forecast_array = np.column_stack((
            self.forecast['ds'].values.astype(float),
            observed_y.values,
            self.forecast['yhat'].values,
            self.forecast['yhat_lower'].values,
            self.forecast['yhat_upper'].values
        ))

        forecast_table = Table.from_numpy(domain, forecast_array)
        self.Outputs.forecast.send(forecast_table)

    def update_info_box(self):
        if self.model is None or self.forecast is None:
            return

        # Calculate metrics
        observed = self.model.history['y']
        predicted = self.forecast['yhat'][:len(observed)]

        mae = mean_absolute_error(observed, predicted)
        mse = mean_squared_error(observed, predicted)
        rmse = np.sqrt(mse)

        info_text = f"""
        Model Parameters:
        - Changepoint Prior Scale: {self.changepoint_prior_scale}
        - Seasonality Prior Scale: {self.seasonality_prior_scale}
        - Holidays Prior Scale: {self.holidays_prior_scale}
        - Seasonality Mode: {self.SEASONALITY_MODES[self.seasonality_mode]}

        Performance Metrics:
        - Mean Absolute Error (MAE): {mae:.4f}
        - Mean Squared Error (MSE): {mse:.4f}
        - Root Mean Squared Error (RMSE): {rmse:.4f}
        """

        self.info_text.setText(info_text)

    def clear_plot(self):
        self.plot_widget.clear()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWProphetForecaster).run()