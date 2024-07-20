import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, StringVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from prophet import Prophet
import pandas as pd


class OWProphetModel(widget.OWWidget):
    name = "Prophet"
    description = "Fit Prophet model and visualize results"
    icon = "icons/prophet.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        residuals = Output("Residuals", Table)
        forecast = Output("Forecast", Table)
        fitted_values = Output("Fitted Values", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    date_variable = Setting("")
    forecast_steps = Setting(30)
    seasonality_mode = Setting(0)  # 0: additive, 1: multiplicative
    yearly_seasonality = Setting(True)
    weekly_seasonality = Setting(True)
    daily_seasonality = Setting(False)
    growth = Setting(0)  # 0: linear, 1: logistic
    changepoint_prior_scale = Setting(0.05)
    seasonality_prior_scale = Setting(10.0)
    holidays_prior_scale = Setting(10.0)
    plot_type = Setting(0)  # 0: Forecast, 1: Components

    def __init__(self):
        super().__init__()

        self.data = None
        self.model = None
        self.results = None
        self.input_df = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Variable selection
        var_box = gui.widgetBox(self.controlArea, "Variable Selection")
        self.date_combo = gui.comboBox(
            var_box, self, "date_variable", label="Date Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_variable_changed)
        self.target_combo = gui.comboBox(
            var_box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_variable_changed)

        # Model parameters
        param_box = gui.widgetBox(self.controlArea, "Model Parameters")
        gui.comboBox(param_box, self, "seasonality_mode", items=["Additive", "Multiplicative"],
                     label="Seasonality Mode:", orientation="horizontal", callback=self.on_param_changed)
        gui.checkBox(param_box, self, "yearly_seasonality", label="Yearly Seasonality", callback=self.on_param_changed)
        gui.checkBox(param_box, self, "weekly_seasonality", label="Weekly Seasonality", callback=self.on_param_changed)
        gui.checkBox(param_box, self, "daily_seasonality", label="Daily Seasonality", callback=self.on_param_changed)
        gui.comboBox(param_box, self, "growth", items=["Linear", "Logistic"],
                     label="Growth:", orientation="horizontal", callback=self.on_param_changed)
        gui.doubleSpin(param_box, self, "changepoint_prior_scale", 0.001, 100, 0.001,
                       label="Changepoint Prior Scale:", callback=self.on_param_changed)
        gui.doubleSpin(param_box, self, "seasonality_prior_scale", 0.01, 100, 0.01,
                       label="Seasonality Prior Scale:", callback=self.on_param_changed)
        gui.doubleSpin(param_box, self, "holidays_prior_scale", 0.01, 100, 0.01,
                       label="Holidays Prior Scale:", callback=self.on_param_changed)

        # Forecast settings
        forecast_box = gui.widgetBox(self.controlArea, "Forecast Settings")
        gui.spin(forecast_box, self, "forecast_steps", 1, 1000, label="Forecast Steps:", callback=self.on_param_changed)

        # Plot type selection
        plot_box = gui.widgetBox(self.controlArea, "Plot Selection")
        gui.comboBox(plot_box, self, "plot_type", items=["Forecast", "Components"],
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

            # Update variable combo box options
            self.date_combo.clear()
            self.target_combo.clear()
            self.date_combo.addItem("")
            self.target_combo.addItem("")

            for var in data.domain.variables + data.domain.metas:
                if isinstance(var, (TimeVariable, StringVariable)):
                    self.date_combo.addItem(var.name)
                if isinstance(var, ContinuousVariable):
                    self.target_combo.addItem(var.name)

            # Set initial variables if previously selected
            if self.date_variable in data.domain:
                self.date_combo.setCurrentIndex(self.date_combo.findText(self.date_variable))
            if self.target_variable in data.domain:
                self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))

            self.on_variable_changed()
        else:
            self.data = None
            self.info_label.setText("No data on input.")
            self.clear_plot()

    def on_variable_changed(self):
        self.date_variable = self.date_combo.currentText()
        self.target_variable = self.target_combo.currentText()
        if self.data is not None and self.date_variable and self.target_variable:
            self.prepare_data()
            self.fit_model()

    def on_param_changed(self):
        if self.data is not None and self.date_variable and self.target_variable:
            self.fit_model()

    def on_plot_type_changed(self):
        self.update_plot()

    def prepare_data(self):
        date_var = self.data.domain[self.date_variable]
        target_var = self.data.domain[self.target_variable]

        dates = self.data.get_column(date_var)
        targets = self.data.get_column(target_var)

        if isinstance(date_var, StringVariable):
            dates = pd.to_datetime(dates)

        self.input_df = pd.DataFrame({
            'ds': dates,
            'y': targets
        })

    def fit_model(self):
        if self.input_df is None:
            return

        self.model = Prophet(
            seasonality_mode='multiplicative' if self.seasonality_mode else 'additive',
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            growth='logistic' if self.growth else 'linear',
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale
        )

        if self.growth:
            self.input_df['cap'] = self.input_df['y'].max()
            self.input_df['floor'] = self.input_df['y'].min()

        self.model.fit(self.input_df)

        future = self.model.make_future_dataframe(periods=self.forecast_steps)
        if self.growth:
            future['cap'] = self.input_df['cap'].iloc[0]
            future['floor'] = self.input_df['floor'].iloc[0]

        self.results = self.model.predict(future)

        self.update_plot()
        self.output_results()

    def update_plot(self):
        self.plot_widget.clear()

        if self.results is None:
            return

        if self.plot_type == 0:  # Forecast
            self.plot_forecast()
        elif self.plot_type == 1:  # Components
            self.plot_components()

    def plot_forecast(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        observed_x = pd.to_datetime(self.input_df['ds']).astype(int) // 10 ** 9
        forecast_x = pd.to_datetime(self.results['ds']).astype(int) // 10 ** 9

        observed_plot = self.plot_widget.plot(observed_x, self.input_df['y'],
                                              pen=pg.mkPen(color=(0, 0, 255), width=2), name='Observed')
        legend.addItem(observed_plot, 'Observed')

        forecast_plot = self.plot_widget.plot(forecast_x, self.results['yhat'],
                                              pen=pg.mkPen(color=(255, 0, 0), width=2), name='Forecast')
        legend.addItem(forecast_plot, 'Forecast')

        ci_lower = self.plot_widget.plot(forecast_x, self.results['yhat_lower'],
                                         pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Lower')
        ci_upper = self.plot_widget.plot(forecast_x, self.results['yhat_upper'],
                                         pen=pg.mkPen(color=(200, 200, 200), width=2), name='CI Upper')
        legend.addItem(ci_lower, 'CI Lower')
        legend.addItem(ci_upper, 'CI Upper')

        self.plot_widget.setLabel('left', self.target_variable)
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Forecast')

    def plot_components(self):
        self.plot_widget.clear()
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.plot_widget.graphicsItem())

        x = pd.to_datetime(self.results['ds']).astype(int) // 10 ** 9

        components = ['trend', 'yearly', 'weekly', 'daily']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for component, color in zip(components, colors):
            if component in self.results.columns:
                plot = self.plot_widget.plot(x, self.results[component],
                                             pen=pg.mkPen(color=color, width=2), name=component.capitalize())
                legend.addItem(plot, component.capitalize())

        self.plot_widget.setLabel('left', 'Component Value')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Model Components')

    def clear_plot(self):
        self.plot_widget.clear()

    def output_results(self):
        if self.results is None or self.input_df is None:
            self.Outputs.residuals.send(None)
            self.Outputs.forecast.send(None)
            self.Outputs.fitted_values.send(None)
            return

        # Residuals
        fitted_values = self.results['yhat'][:len(self.input_df)]
        residuals = self.input_df['y'] - fitted_values
        residuals_domain = Domain([ContinuousVariable('Residuals')],
                                  metas=[StringVariable('Time')])
        residuals_table = Table(residuals_domain,
                                np.atleast_2d(residuals).T,
                                metas=np.atleast_2d(self.input_df['ds']).T)
        self.Outputs.residuals.send(residuals_table)

        # Forecast
        forecast_domain = Domain([ContinuousVariable('yhat'),
                                  ContinuousVariable('yhat_lower'),
                                  ContinuousVariable('yhat_upper')],
                                 metas=[StringVariable('Time')])
        forecast_table = Table(forecast_domain,
                               self.results[['yhat', 'yhat_lower', 'yhat_upper']].values,
                               metas=np.atleast_2d(self.results['ds']).T)
        self.Outputs.forecast.send(forecast_table)

        # Fitted Values
        fitted_domain = Domain([ContinuousVariable('Fitted')],
                               metas=[StringVariable('Time')])
        fitted_table = Table(fitted_domain,
                             np.atleast_2d(fitted_values).T,
                             metas=np.atleast_2d(self.input_df['ds']).T)
        self.Outputs.fitted_values.send(fitted_table)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWProphetModel).run()