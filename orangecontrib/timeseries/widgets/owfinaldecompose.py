import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from datetime import datetime, timedelta


class OWSeasonalDecomposition(widget.OWWidget):
    name = "Seasonal Decomposition"
    description = "Perform seasonal decomposition on time series data"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        decomposed_data = Output("Decomposed Data", Table)

    want_main_area = True

    seasonality = Setting(12)  # Default seasonality
    target_variable = Setting("")  # Selected target variable
    model = Setting(0)  # 0 for additive, 1 for multiplicative
    max_lags = Setting(30)  # Default max lags for ACF

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.time_values = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.apply)

        # Add seasonality control
        seasonality_box = gui.widgetBox(self.controlArea, "Decomposition Parameters")
        self.seasonality_spin = gui.spin(
            seasonality_box, self, "seasonality", minv=2, maxv=365,
            label="Seasonality:", callback=self.apply)

        # Add model selection
        self.model_combo = gui.comboBox(
            seasonality_box, self, "model", items=["Additive", "Multiplicative"],
            label="Model:", orientation="horizontal", callback=self.apply)

        # Add max lags control for ACF
        self.lags_spin = gui.spin(
            seasonality_box, self, "max_lags", minv=1, maxv=100,
            label="Max Lags for ACF:", callback=self.apply)

        # Set up the main area with four plot widgets
        self.original_plot = PlotWidget(background="w")
        self.trend_plot = PlotWidget(background="w")
        self.seasonal_plot = PlotWidget(background="w")
        self.residual_acf_plot = PlotWidget(background="w")

        gui.vBox(self.mainArea).layout().addWidget(self.original_plot)
        gui.vBox(self.mainArea).layout().addWidget(self.trend_plot)
        gui.vBox(self.mainArea).layout().addWidget(self.seasonal_plot)
        gui.vBox(self.mainArea).layout().addWidget(self.residual_acf_plot)

    @Inputs.time_series
    def set_data(self, data):
        self.clear_plots()
        self.clear_messages()  # Clear any previous error messages

        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = data.time_variable

            if self.time_variable is None:
                self.error("Input data has no time variable")
                return

            self.time_values = data.get_column_view(self.time_variable)[0]

            # Update target variable combo box options
            self.target_combo.clear()
            self.target_combo.addItem("")
            for var in data.domain.variables:
                if var.is_continuous and var != self.time_variable:
                    self.target_combo.addItem(var.name)

            # Set initial target variable if previously selected
            if self.target_variable in data.domain:
                self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))
            else:
                self.target_variable = ""  # Reset if not found in new data

            self.apply()
        else:
            self.data = None
            self.time_variable = None
            self.time_values = None
            self.info_label.setText("No data on input.")
            self.target_variable = ""
            self.target_combo.clear()

    def apply(self):
        self.clear_plots()
        self.clear_messages()  # Clear any previous error messages

        if self.data is None or not self.target_variable:
            return

        value_var = self.data.domain[self.target_variable]
        y_values = self.data.get_column(value_var)

        try:
            result = seasonal_decompose(y_values, model='additive' if self.model == 0 else 'multiplicative',
                                        period=self.seasonality)

            self.plot_decomposition(result)

            # Create output table
            domain = Domain([ContinuousVariable("Original"),
                             ContinuousVariable("Trend"),
                             ContinuousVariable("Seasonal"),
                             ContinuousVariable("Residual")])

            output_data = Table.from_numpy(domain, np.column_stack((
                y_values,
                result.trend,
                result.seasonal,
                result.resid
            )))

            self.Outputs.decomposed_data.send(output_data)

        except Exception as e:
            self.error(str(e))

    def plot_decomposition(self, result):
        self.plot_component(self.original_plot, result.observed, "Original")
        self.plot_component(self.trend_plot, result.trend, "Trend")
        self.plot_component(self.seasonal_plot, result.seasonal, "Seasonal")
        self.plot_acf(self.residual_acf_plot, result.resid, "ACF of Residuals")

    def plot_component(self, plot_widget, values, component_name):
        plot_widget.clear()
        plot_widget.getAxis('bottom').setLabel('Time')
        plot_widget.getAxis('left').setLabel('Value')

        x = self.time_values
        plot_widget.plot(x, values, pen=pg.mkPen(color=(0, 0, 255), width=2))

        self.style_plot(plot_widget, component_name)
        self.set_time_axis(plot_widget)

    def plot_acf(self, plot_widget, residuals, component_name):
        plot_widget.clear()
        plot_widget.getAxis('bottom').setLabel('Lag')
        plot_widget.getAxis('left').setLabel('ACF')

        # Remove NaN values
        residuals = residuals[~np.isnan(residuals)]

        acf_values = acf(residuals, nlags=self.max_lags)
        x = np.arange(len(acf_values))

        # Plot vertical lines (sticks)
        for i in x:
            plot_widget.plot([i, i], [0, acf_values[i]], pen=pg.mkPen(color=(0, 0, 255), width=3))

        # Plot markers at the top of each stick
        plot_widget.plot(x, acf_values, pen=None, symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=1),
                         symbolBrush=(0, 0, 255, 200), symbolSize=5)

        # Add zero line
        plot_widget.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=1, style=Qt.DashLine))

        # Add 5% significance level lines
        significance_level = 1.96 / np.sqrt(len(residuals))
        plot_widget.addLine(y=significance_level, pen=pg.mkPen(color=(255, 0, 0), width=3, style=Qt.DotLine))
        plot_widget.addLine(y=-significance_level, pen=pg.mkPen(color=(255, 0, 0), width=3, style=Qt.DotLine))

        self.style_plot(plot_widget, component_name)

        # Set y-axis range to ensure significance lines are visible
        plot_widget.setYRange(-1, 1)

    def style_plot(self, plot_widget, title):
        # Set background to white
        plot_widget.setBackground('w')

        # Increase font size and thickness of axis labels
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        plot_widget.getAxis('bottom').setTickFont(font)
        plot_widget.getAxis('left').setTickFont(font)

        # Set title with larger, bold font
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        plot_widget.setTitle(f"{title} for {self.target_variable}", color='k', size='14pt')

        # Increase axis line width
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
        plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=2))

    def set_time_axis(self, plot_widget):
        axis = plot_widget.getAxis('bottom')

        # Convert timestamps to datetime objects
        datetimes = [self.timestamp_to_datetime(ts) for ts in self.time_values]

        # Determine appropriate date format based on the time range
        date_format = self.get_date_format(datetimes[0], datetimes[-1])

        # Create tick values and labels
        ticks = []
        for i in range(0, len(datetimes), max(1, len(datetimes) // 10)):
            tick_value = self.time_values[i]
            tick_label = datetimes[i].strftime(date_format)
            ticks.append((tick_value, tick_label))

        axis.setTicks([ticks])

    def timestamp_to_datetime(self, timestamp):
        # Convert Orange's TimeVariable timestamp to Python datetime
        # Orange stores time as seconds since the epoch (1970-01-01)
        return datetime(1970, 1, 1) + timedelta(seconds=timestamp)

    def get_date_format(self, start_date, end_date):
        # Determine the appropriate date format based on the time range
        delta = end_date - start_date
        if delta.days > 365:
            return "%Y-%m"  # Year-Month for ranges over a year
        elif delta.days > 30:
            return "%Y-%m-%d"  # Year-Month-Day for ranges over a month
        elif delta.days > 1:
            return "%m-%d"  # Month-Day for ranges over a day
        else:
            return "%H:%M"  # Hour:Minute for ranges within a day

    def clear_plots(self):
        self.original_plot.clear()
        self.trend_plot.clear()
        self.seasonal_plot.clear()
        self.residual_acf_plot.clear()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSeasonalDecomposition).run()

