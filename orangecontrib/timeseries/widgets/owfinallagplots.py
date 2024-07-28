import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtWidgets import QScrollArea, QGridLayout, QWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class OWLagPlot(widget.OWWidget):
    name = "Lag Plot"
    description = "Visualize lag plots of time series data"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table)

    want_main_area = True

    max_lags = Setting(12)  # Default to 12 lags
    variable_index = Setting(0)

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_series = None
        self.variables = []

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Add max lags control
        lags_box = gui.widgetBox(self.controlArea, "Lag Plot Parameters")
        self.lags_spin = gui.spin(
            lags_box, self, "max_lags", minv=1, maxv=100,
            label="Max Lags:", callback=self.plot_results)

        # Variable selector
        self.var_combo = gui.comboBox(
            self.controlArea, self, "variable_index", box="Variable",
            callback=self.variable_changed)

        # Set up the main area with a scroll area for plots
        self.scroll_area = QScrollArea()
        self.mainArea.layout().addWidget(self.scroll_area)
        self.plot_widget = QWidget()
        self.plot_layout = QGridLayout()
        self.plot_widget.setLayout(self.plot_layout)
        self.scroll_area.setWidget(self.plot_widget)
        self.scroll_area.setWidgetResizable(True)

    @Inputs.time_series
    def set_data(self, data):
        self.data = data
        self.variables = []
        self.var_combo.clear()

        if data is not None:
            self.variables = [var for var in data.domain.variables if var.is_continuous]
            self.var_combo.addItems([var.name for var in self.variables])
            self.info_label.setText(f"{len(data)} instances, {len(self.variables)} numeric variables on input.")
            self.variable_index = 0
            self.variable_changed()
        else:
            self.info_label.setText("No data on input.")
            self.clear_plots()

    def variable_changed(self):
        if self.data is not None and self.variables:
            self.time_series = self.data.get_column(self.variables[self.variable_index])
            self.plot_results()

    def plot_results(self):
        if self.time_series is None:
            return

        self.clear_plots()

        n_cols = 4
        n_rows = (self.max_lags // n_cols) + (1 if self.max_lags % n_cols else 0)

        for lag in range(1, self.max_lags + 1):
            plot_widget = PlotWidget(background="w")
            self.plot_layout.addWidget(plot_widget, (lag - 1) // n_cols, (lag - 1) % n_cols)

            self.plot_lag(plot_widget, lag)

    def plot_lag(self, plot_widget, lag):
        plot_widget.clear()
        plot_widget.getAxis('bottom').setLabel('Current Value')
        plot_widget.getAxis('left').setLabel('Lagged Value')

        x = self.time_series[:-lag]
        y = self.time_series[lag:]

        # Plot scatter points
        scatter = pg.ScatterPlotItem(x=x, y=y, size=5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 120))
        plot_widget.addItem(scatter)

        # Add x=y line
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        line = pg.InfiniteLine(angle=45, pen=pg.mkPen(color=(255, 0, 0), width=1, style=Qt.DashLine))
        plot_widget.addItem(line)

        # Set axis range
        plot_widget.setXRange(min_val, max_val)
        plot_widget.setYRange(min_val, max_val)

        # Set background to white
        plot_widget.setBackground('w')

        # Increase font size and thickness of axis labels
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        plot_widget.getAxis('bottom').setTickFont(font)
        plot_widget.getAxis('left').setTickFont(font)

        # Set title with larger, bold font
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        plot_widget.setTitle(f"Lag Plot (lag={lag})", color='k', size='10pt')

        # Increase axis line width
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=1))

    def clear_plots(self):
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLagPlot).run()