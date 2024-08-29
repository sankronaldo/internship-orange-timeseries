import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table
from Orange.widgets.widget import Input
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtWidgets import QScrollArea, QGridLayout, QWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from statsmodels.tsa.stattools import ccf


class OWMultivariateCCF(widget.OWWidget):
    name = "Multivariate CCF"
    description = "Plot the residual CCF from a VAR model"
    icon = "icons/ow_mulccf.svg"
    priority = 10

    class Inputs:
        var_residuals = Input("data", Table)

    want_main_area = True

    max_lags = Setting(20)  # Default to 20 lags

    def __init__(self):
        super().__init__()

        self.residuals = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No residuals on input.")

        # Add max lags control
        lags_box = gui.widgetBox(self.controlArea, "CCF Parameters")
        self.lags_spin = gui.spin(
            lags_box, self, "max_lags", minv=1, maxv=100,
            label="Max Lags:", callback=self.plot_results)

        # Set up the main area with a scroll area for plots
        self.scroll_area = QScrollArea()
        self.mainArea.layout().addWidget(self.scroll_area)
        self.plot_widget = QWidget()
        self.plot_layout = QGridLayout()
        self.plot_widget.setLayout(self.plot_layout)
        self.scroll_area.setWidget(self.plot_widget)
        self.scroll_area.setWidgetResizable(True)

    @Inputs.var_residuals
    def set_residuals(self, residuals):
        if residuals is not None:
            self.residuals = residuals
            self.info_label.setText(
                f"{len(residuals)} instances, {len(residuals.domain.attributes)} variables on input.")
            self.plot_results()
        else:
            self.residuals = None
            self.info_label.setText("No residuals on input.")
            self.clear_plots()

    def plot_results(self):
        if self.residuals is None:
            return

        self.clear_plots()

        variables = self.residuals.domain.attributes
        n_vars = len(variables)

        for i in range(n_vars):
            for j in range(n_vars):
                plot_widget = PlotWidget(background="w")
                self.plot_layout.addWidget(plot_widget, i, j)

                var1 = variables[i]
                var2 = variables[j]

                y1 = self.residuals.get_column(var1)
                y2 = self.residuals.get_column(var2)

                ccf_values = ccf(y1, y2, adjusted=False)[:self.max_lags + 1]

                self.plot_ccf(plot_widget, ccf_values, var1.name, var2.name, len(y1))

    def plot_ccf(self, plot_widget, values, var1_name, var2_name, n_samples):
        plot_widget.clear()
        plot_widget.getAxis('bottom').setLabel('Lag')
        plot_widget.getAxis('left').setLabel('CCF')

        x = np.arange(len(values))

        # Plot vertical lines (sticks)
        for i in x:
            plot_widget.plot([i, i], [0, values[i]], pen=pg.mkPen(color=(0, 0, 255), width=2))

        # Plot markers at the top of each stick
        plot_widget.plot(x, values, pen=None, symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=1),
                         symbolBrush=(0, 0, 255, 200), symbolSize=5)

        # Add zero line
        plot_widget.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=1, style=Qt.DashLine))

        # Add 0.05 significance level lines
        sig_level = 1.96 / np.sqrt(n_samples)  # 1.96 is the z-score for 0.05 significance
        plot_widget.addLine(y=sig_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=Qt.DotLine))
        plot_widget.addLine(y=-sig_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=Qt.DotLine))

        # Set y-axis range
        plot_widget.setYRange(-1, 1)

        # Set x-axis range
        plot_widget.setXRange(-0.5, self.max_lags + 0.5)

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
        plot_widget.setTitle(f"CCF: {var1_name} vs {var2_name}", color='k', size='10pt')

        # Increase axis line width
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=1))

    def clear_plots(self):
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWMultivariateCCF).run()

