import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Input, Output
from statsmodels.tsa.stattools import acf, pacf
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from scipy import stats

class OWACFPACF(widget.OWWidget):
    name = "ACF & PACF"
    description = "Plot the ACF and PACF of time series data"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        acf_pacf_data = Output("ACF & PACF Data", Table)

    want_main_area = True

    max_lags = Setting(30)  # Default to 30 lags
    target_variable = Setting("")  # Selected target variable
    significance_level = Setting(0.05)  # Default significance level (5%)

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", sendSelectedValue=True, callback=self.on_target_variable_changed)

        # Add max lags control
        lags_box = gui.widgetBox(self.controlArea, "ACF & PACF Parameters")
        self.lags_spin = gui.spin(
            lags_box, self, "max_lags", minv=1, maxv=365,
            label="Max Lags:", callback=self.on_target_variable_changed)

        # Add significance level control
        self.significance_spin = gui.doubleSpin(
            lags_box, self, "significance_level", minv=0.01, maxv=0.1, step=0.01,
            label="Significance Level:", callback=self.on_target_variable_changed)

        # Set up the main area with two plot widgets
        self.acf_plot = PlotWidget(background="w")
        self.pacf_plot = PlotWidget(background="w")

        gui.vBox(self.mainArea).layout().addWidget(self.acf_plot)
        gui.vBox(self.mainArea).layout().addWidget(self.pacf_plot)

    @Inputs.time_series
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")
            # Check if the data has a time_variable attribute
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
        self.plot_results()

    def plot_results(self):
        if self.data is None or not self.target_variable:
            return

        value_var = self.data.domain[self.target_variable]
        y_values = self.data.get_column(value_var)

        # Calculate ACF
        acf_values = acf(y_values, nlags=self.max_lags)

        # Calculate PACF
        pacf_values = pacf(y_values, nlags=self.max_lags, method="ywm")

        # Remove lag 0 from ACF (it's always 1)
        acf_values = acf_values[1:]

        # Ensure PACF has the same length as ACF (remove lag 0 if present)
        if len(pacf_values) > len(acf_values):
            pacf_values = pacf_values[1:]

        # Plot ACF
        self.plot_correlation(self.acf_plot, acf_values, "Autocorrelation")

        # Plot PACF
        self.plot_correlation(self.pacf_plot, pacf_values, "Partial Autocorrelation")

        # Ensure both arrays have the same length before combining
        min_length = min(len(acf_values), len(pacf_values))
        acf_values = acf_values[:min_length]
        pacf_values = pacf_values[:min_length]

        # Send ACF and PACF data as a single output
        domain = Domain([ContinuousVariable("ACF"), ContinuousVariable("PACF")],
                        metas=[ContinuousVariable("Lag")])
        combined_data = Table.from_numpy(
            domain,
            np.column_stack((acf_values, pacf_values)),
            metas=np.arange(1, min_length + 1).reshape(-1, 1)
        )
        self.Outputs.acf_pacf_data.send(combined_data)

    def plot_correlation(self, plot_widget, values, plot_type):
        plot_widget.clear()
        plot_widget.getAxis('bottom').setLabel('Lag')
        plot_widget.getAxis('left').setLabel(plot_type)

        x = np.arange(1, len(values) + 1)

        # Plot vertical lines (sticks)
        for i in x:
            plot_widget.plot([i, i], [0, values[i - 1]], pen=pg.mkPen(color=(0, 0, 255), width=4))

        # Plot markers at the top of each stick
        plot_widget.plot(x, values, pen=None, symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=1),
                         symbolBrush=(0, 0, 255, 200), symbolSize=5)

        # Add zero line
        plot_widget.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=2, style=pg.QtCore.Qt.DashLine))

        # Add significance levels
        z_score = stats.norm.ppf(1 - self.significance_level / 2)
        significance_level = z_score / np.sqrt(len(self.data))
        plot_widget.addLine(y=significance_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))
        plot_widget.addLine(y=-significance_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))

        # Set y-axis range
        plot_widget.setYRange(-1, 1)

        # Set x-axis range
        plot_widget.setXRange(0.5, self.max_lags + 0.5)

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
        plot_widget.setTitle(f"{plot_type} for {self.target_variable}", color='k', size='14pt')

        # Increase axis line width
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
        plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=2))

    def clear_plot(self):
        self.acf_plot.clear()
        self.pacf_plot.clear()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWACFPACF).run()
