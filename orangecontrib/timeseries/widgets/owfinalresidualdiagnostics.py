from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pyqtgraph as pg
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from PyQt5.QtGui import QFont
from Orange.widgets.utils.widgetpreview import WidgetPreview

class OWResidualDiagnostics(widget.OWWidget):
    name = "Residual Diagnostics"
    description = "Perform Ljung-Box and Shapiro-Wilk tests on residuals"
    icon = "icons/ow_residual.svg"
    priority = 70

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        diagnostic_results = Output("Diagnostic Results", Table)

    want_main_area = True

    # Settings
    ljung_box_lags = Setting(10)
    target_variable = ContextSetting("")

    def __init__(self):
        super().__init__()

        self.data = None
        self.residuals = None
        self.ljung_box_result = None
        self.shapiro_wilk_result = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Ljung-Box Test")
        gui.spin(box, self, "ljung_box_lags", minv=1, maxv=100,
                 label="Number of Lags:", callback=self.update_results)

        # Add target variable selection
        self.target_combo = gui.comboBox(
            self.controlArea, self, "target_variable",
            box="Target Variable", label="Select Variable:",
            callback=self.update_results)

        self.info_box = gui.widgetBox(self.controlArea, "Test Results")
        self.info_label = gui.widgetLabel(self.info_box, "No data on input.")

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.mainArea.layout().addWidget(self.plot_widget)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.update_variable_list()
        self.update_results()

    def update_variable_list(self):
        self.target_combo.clear()
        if self.data is not None:
            for var in self.data.domain.variables:
                if var.is_continuous:
                    self.target_combo.addItem(var.name)
            if self.target_combo.count() > 0:
                self.target_variable = self.target_combo.itemText(0)
        else:
            self.info_label.setText("No data on input.")
            self.plot_widget.clear()

    def update_results(self):
        if self.data is None or self.target_variable == "":
            return

        var = self.data.domain[self.target_variable]
        self.residuals = self.data.get_column(var)

        # Perform Ljung-Box test
        self.ljung_box_result = acorr_ljungbox(self.residuals, lags=[self.ljung_box_lags])

        # Perform Shapiro-Wilk test
        self.shapiro_wilk_result = stats.shapiro(self.residuals)

        self.update_info()
        self.update_plot()
        self.send_output()

    def update_info(self):
        lb_pvalue = self.ljung_box_result.iloc[0, 1]
        sw_pvalue = self.shapiro_wilk_result.pvalue

        info_text = f"Target Variable: {self.target_variable}\n\n"
        info_text += f"Ljung-Box Test (lag={self.ljung_box_lags}):\n"
        info_text += f"p-value: {lb_pvalue:.4f}\n\n"
        info_text += "Shapiro-Wilk Test:\n"
        info_text += f"p-value: {sw_pvalue:.4f}"

        self.info_label.setText(info_text)

    def update_plot(self):
        self.plot_widget.clear()
        self.plot_widget.setBackground('w')  # Set background to white for the entire widget

        # Create two separate plot items
        acf_plot = self.plot_widget.addPlot(row=0, col=0, title="Autocorrelation Function (ACF)")
        dist_plot = self.plot_widget.addPlot(row=1, col=0, title="Residual Distribution")

        # Calculate ACF
        acf_values = acf(self.residuals, nlags=self.ljung_box_lags)
        acf_values = acf_values[1:]  # Remove lag 0
        x = np.arange(1, len(acf_values) + 1)

        # Plot ACF
        for i in x:
            acf_plot.plot([i, i], [0, acf_values[i - 1]], pen=pg.mkPen(color=(0, 0, 255), width=4))

        acf_plot.plot(x, acf_values, pen=None, symbol='o',
                      symbolPen=pg.mkPen(color=(0, 0, 255), width=1),
                      symbolBrush=(0, 0, 255, 200), symbolSize=5)

        # Add zero line
        acf_plot.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=2, style=pg.QtCore.Qt.DashLine))

        # Add significance levels
        z_score = stats.norm.ppf(1 - 0.05 / 2)  # Using 5% significance level
        significance_level = z_score / np.sqrt(len(self.residuals))
        acf_plot.addLine(y=significance_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))
        acf_plot.addLine(y=-significance_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))

        # Set y-axis range
        acf_plot.setYRange(-1, 1)

        # Set x-axis range
        acf_plot.setXRange(0.5, self.ljung_box_lags + 0.5)

        # Increase font size and thickness of axis labels
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        acf_plot.getAxis('bottom').setTickFont(font)
        acf_plot.getAxis('left').setTickFont(font)

        # Set title with larger, bold font
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        acf_plot.setTitle("Autocorrelation Function (ACF)", color='k', size='14pt')

        # Increase axis line width
        acf_plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
        acf_plot.getAxis('left').setPen(pg.mkPen(color='k', width=2))

        acf_plot.setLabel('left', 'ACF')
        acf_plot.setLabel('bottom', 'Lag')

        # Plot residual distribution
        hist, bin_edges = np.histogram(self.residuals, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_graph = pg.BarGraphItem(x=bin_centers, height=hist, width=(bin_edges[1] - bin_edges[0]),
                                    brush=(0, 0, 255, 50))
        dist_plot.addItem(bar_graph)

        # Plot normal distribution
        mean, std = np.mean(self.residuals), np.std(self.residuals)
        xmin, xmax = np.min(self.residuals), np.max(self.residuals)
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean, std)
        dist_plot.plot(x, p, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Normal Distribution')

        dist_plot.setLabel('left', 'Density')
        dist_plot.setLabel('bottom', 'Residual Value')

        # Add legend for distribution plot
        dist_plot.addLegend()

        # Apply similar styling to distribution plot
        dist_plot.getAxis('bottom').setTickFont(font)
        dist_plot.getAxis('left').setTickFont(font)
        dist_plot.setTitle("Residual Distribution", color='k', size='14pt')
        dist_plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
        dist_plot.getAxis('left').setPen(pg.mkPen(color='k', width=2))

    def send_output(self):
        lb_pvalue = self.ljung_box_result.iloc[0, 1]
        sw_pvalue = self.shapiro_wilk_result.pvalue

        domain = Domain([ContinuousVariable("Ljung-Box p-value"),
                         ContinuousVariable("Shapiro-Wilk p-value")])
        output_data = Table(domain, [[lb_pvalue, sw_pvalue]])
        self.Outputs.diagnostic_results.send(output_data)

if __name__ == "__main__":
    WidgetPreview(OWResidualDiagnostics).run()





# from statsmodels.graphics.tsaplots import plot_acf
# import numpy as np
# import pyqtgraph as pg
# from Orange.data import Table, Domain, ContinuousVariable
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# from scipy import stats
# from statsmodels.stats.diagnostic import acorr_ljungbox
# from statsmodels.tsa.stattools import acf
# from PyQt5.QtGui import QFont
#
#
# class OWResidualDiagnostics(widget.OWWidget):
#     name = "Residual Diagnostics"
#     description = "Perform Ljung-Box and Shapiro-Wilk tests on residuals"
#     icon = "icons/ow_residual.svg"
#     priority = 70
#
#     class Inputs:
#         residuals = Input("Residuals", Table)
#
#     class Outputs:
#         diagnostic_results = Output("Diagnostic Results", Table)
#
#     want_main_area = True
#
#     # Settings
#     ljung_box_lags = Setting(10)
#
#     def __init__(self):
#         super().__init__()
#
#         self.residuals = None
#         self.ljung_box_result = None
#         self.shapiro_wilk_result = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Ljung-Box Test")
#         gui.spin(box, self, "ljung_box_lags", minv=1, maxv=100,
#                  label="Number of Lags:", callback=self.update_results)
#
#         self.info_box = gui.widgetBox(self.controlArea, "Test Results")
#         self.info_label = gui.widgetLabel(self.info_box, "No data on input.")
#
#         self.plot_widget = pg.GraphicsLayoutWidget()
#         self.mainArea.layout().addWidget(self.plot_widget)
#
#     @Inputs.residuals
#     def set_data(self, data):
#         if data is not None:
#             self.residuals = data.X.ravel()
#             self.update_results()
#         else:
#             self.residuals = None
#             self.info_label.setText("No data on input.")
#             self.plot_widget.clear()
#
#     def update_results(self):
#         if self.residuals is None:
#             return
#
#         # Perform Ljung-Box test
#         self.ljung_box_result = acorr_ljungbox(self.residuals, lags=[self.ljung_box_lags])
#
#         # Perform Shapiro-Wilk test
#         self.shapiro_wilk_result = stats.shapiro(self.residuals)
#
#         self.update_info()
#         self.update_plot()
#         self.send_output()
#
#     def update_info(self):
#         lb_pvalue = self.ljung_box_result.iloc[0, 1]
#         sw_pvalue = self.shapiro_wilk_result.pvalue
#
#         info_text = f"Ljung-Box Test (lag={self.ljung_box_lags}):\n"
#         info_text += f"p-value: {lb_pvalue:.4f}\n\n"
#         info_text += "Shapiro-Wilk Test:\n"
#         info_text += f"p-value: {sw_pvalue:.4f}"
#
#         self.info_label.setText(info_text)
#
#     def update_plot(self):
#         self.plot_widget.clear()
#         self.plot_widget.setBackground('w')  # Set background to white for the entire widget
#
#         # Create two separate plot items
#         acf_plot = self.plot_widget.addPlot(row=0, col=0, title="Autocorrelation Function (ACF)")
#         dist_plot = self.plot_widget.addPlot(row=1, col=0, title="Residual Distribution")
#
#         # Calculate ACF
#         acf_values = acf(self.residuals, nlags=self.ljung_box_lags)
#         acf_values = acf_values[1:]  # Remove lag 0
#         x = np.arange(1, len(acf_values) + 1)
#
#         # Plot ACF
#         for i in x:
#             acf_plot.plot([i, i], [0, acf_values[i - 1]], pen=pg.mkPen(color=(0, 0, 255), width=4))
#
#         acf_plot.plot(x, acf_values, pen=None, symbol='o',
#                       symbolPen=pg.mkPen(color=(0, 0, 255), width=1),
#                       symbolBrush=(0, 0, 255, 200), symbolSize=5)
#
#         # Add zero line
#         acf_plot.addLine(y=0, pen=pg.mkPen(color=(0, 0, 0), width=2, style=pg.QtCore.Qt.DashLine))
#
#         # Add significance levels
#         z_score = stats.norm.ppf(1 - 0.05 / 2)  # Using 5% significance level
#         significance_level = z_score / np.sqrt(len(self.residuals))
#         acf_plot.addLine(y=significance_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))
#         acf_plot.addLine(y=-significance_level, pen=pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DotLine))
#
#         # Set y-axis range
#         acf_plot.setYRange(-1, 1)
#
#         # Set x-axis range
#         acf_plot.setXRange(0.5, self.ljung_box_lags + 0.5)
#
#         # Increase font size and thickness of axis labels
#         font = QFont()
#         font.setPointSize(12)
#         font.setBold(True)
#         acf_plot.getAxis('bottom').setTickFont(font)
#         acf_plot.getAxis('left').setTickFont(font)
#
#         # Set title with larger, bold font
#         title_font = QFont()
#         title_font.setPointSize(14)
#         title_font.setBold(True)
#         acf_plot.setTitle("Autocorrelation Function (ACF)", color='k', size='14pt')
#
#         # Increase axis line width
#         acf_plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
#         acf_plot.getAxis('left').setPen(pg.mkPen(color='k', width=2))
#
#         acf_plot.setLabel('left', 'ACF')
#         acf_plot.setLabel('bottom', 'Lag')
#
#         # Plot residual distribution
#         hist, bin_edges = np.histogram(self.residuals, bins='auto', density=True)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#         bar_graph = pg.BarGraphItem(x=bin_centers, height=hist, width=(bin_edges[1] - bin_edges[0]),
#                                     brush=(0, 0, 255, 50))
#         dist_plot.addItem(bar_graph)
#
#         # Plot normal distribution
#         mean, std = np.mean(self.residuals), np.std(self.residuals)
#         xmin, xmax = np.min(self.residuals), np.max(self.residuals)
#         x = np.linspace(xmin, xmax, 100)
#         p = stats.norm.pdf(x, mean, std)
#         dist_plot.plot(x, p, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Normal Distribution')
#
#         dist_plot.setLabel('left', 'Density')
#         dist_plot.setLabel('bottom', 'Residual Value')
#
#         # Add legend for distribution plot
#         dist_plot.addLegend()
#
#         # Apply similar styling to distribution plot
#         dist_plot.getAxis('bottom').setTickFont(font)
#         dist_plot.getAxis('left').setTickFont(font)
#         dist_plot.setTitle("Residual Distribution", color='k', size='14pt')
#         dist_plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
#         dist_plot.getAxis('left').setPen(pg.mkPen(color='k', width=2))
#
#     def send_output(self):
#         lb_pvalue = self.ljung_box_result.iloc[0, 1]
#         sw_pvalue = self.shapiro_wilk_result.pvalue
#
#         domain = Domain([ContinuousVariable("Ljung-Box p-value"),
#                          ContinuousVariable("Shapiro-Wilk p-value")])
#         output_data = Table(domain, [[lb_pvalue, sw_pvalue]])
#         self.Outputs.diagnostic_results.send(output_data)
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(OWResidualDiagnostics).run()









# from statsmodels.graphics.tsaplots import plot_acf
# import numpy as np
# import pyqtgraph as pg
# from Orange.data import Table, Domain, ContinuousVariable
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# from scipy import stats
# from statsmodels.stats.diagnostic import acorr_ljungbox
# from statsmodels.tsa.stattools import acf
#
# class OWResidualDiagnostics(widget.OWWidget):
#     name = "Residual Diagnostics"
#     description = "Perform Ljung-Box and Shapiro-Wilk tests on residuals"
#     icon = "icons/residual-diagnostics.svg"
#     priority = 70
#
#     class Inputs:
#         residuals = Input("Residuals", Table)
#
#     class Outputs:
#         diagnostic_results = Output("Diagnostic Results", Table)
#
#     want_main_area = True
#
#     # Settings
#     ljung_box_lags = Setting(10)
#
#     def __init__(self):
#         super().__init__()
#
#         self.residuals = None
#         self.ljung_box_result = None
#         self.shapiro_wilk_result = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Ljung-Box Test")
#         gui.spin(box, self, "ljung_box_lags", minv=1, maxv=100,
#                  label="Number of Lags:", callback=self.update_results)
#
#         self.info_box = gui.widgetBox(self.controlArea, "Test Results")
#         self.info_label = gui.widgetLabel(self.info_box, "No data on input.")
#
#         self.plot_widget = pg.GraphicsLayoutWidget()
#         self.mainArea.layout().addWidget(self.plot_widget)
#
#     @Inputs.residuals
#     def set_data(self, data):
#         if data is not None:
#             self.residuals = data.X.ravel()
#             self.update_results()
#         else:
#             self.residuals = None
#             self.info_label.setText("No data on input.")
#             self.plot_widget.clear()
#
#     def update_results(self):
#         if self.residuals is None:
#             return
#
#         # Perform Ljung-Box test
#         self.ljung_box_result = acorr_ljungbox(self.residuals, lags=[self.ljung_box_lags])
#
#         # Perform Shapiro-Wilk test
#         self.shapiro_wilk_result = stats.shapiro(self.residuals)
#
#         self.update_info()
#         self.update_plot()
#         self.send_output()
#
#     def update_info(self):
#         lb_pvalue = self.ljung_box_result.iloc[0, 1]
#         sw_pvalue = self.shapiro_wilk_result.pvalue
#
#         info_text = f"Ljung-Box Test (lag={self.ljung_box_lags}):\n"
#         info_text += f"p-value: {lb_pvalue:.4f}\n\n"
#         info_text += "Shapiro-Wilk Test:\n"
#         info_text += f"p-value: {sw_pvalue:.4f}"
#
#         self.info_label.setText(info_text)
#
#     def update_plot(self):
#         self.plot_widget.clear()
#
#         # Create two separate plot items
#         acf_plot = self.plot_widget.addPlot(row=0, col=0, title="ACF Plot")
#         dist_plot = self.plot_widget.addPlot(row=1, col=0, title="Residual Distribution")
#
#         # Calculate ACF
#         acf_values, confint = acf(self.residuals, nlags=self.ljung_box_lags, alpha=0.05, fft=False)
#         acf_x = np.arange(len(acf_values))
#
#         # Plot ACF
#         acf_plot.plot(acf_x, acf_values, pen=pg.mkPen(color=(0, 0, 255), width=2), name='ACF')
#
#         # Plot confidence intervals
#         lower_ci = confint[:, 0] - acf_values
#         upper_ci = confint[:, 1] - acf_values
#         err = pg.ErrorBarItem(x=acf_x, y=acf_values, top=upper_ci, bottom=lower_ci, beam=0.5)
#         acf_plot.addItem(err)
#
#         acf_plot.setLabel('left', 'ACF')
#         acf_plot.setLabel('bottom', 'Lag')
#
#         # Plot residual distribution
#         hist, bin_edges = np.histogram(self.residuals, bins='auto', density=True)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#         bar_graph = pg.BarGraphItem(x=bin_centers, height=hist, width=(bin_edges[1] - bin_edges[0]),
#                                     brush=(0, 0, 255, 50))
#         dist_plot.addItem(bar_graph)
#
#         # Plot normal distribution
#         mean, std = np.mean(self.residuals), np.std(self.residuals)
#         xmin, xmax = np.min(self.residuals), np.max(self.residuals)
#         x = np.linspace(xmin, xmax, 100)
#         p = stats.norm.pdf(x, mean, std)
#         dist_plot.plot(x, p, pen=pg.mkPen(color=(255, 0, 0), width=2), name='Normal Distribution')
#
#         dist_plot.setLabel('left', 'Density')
#         dist_plot.setLabel('bottom', 'Residual Value')
#
#         # Add legends
#         acf_plot.addLegend()
#         dist_plot.addLegend()
#
#     def send_output(self):
#         lb_pvalue = self.ljung_box_result.iloc[0, 1]
#         sw_pvalue = self.shapiro_wilk_result.pvalue
#
#         domain = Domain([ContinuousVariable("Ljung-Box p-value"),
#                          ContinuousVariable("Shapiro-Wilk p-value")])
#         output_data = Table(domain, [[lb_pvalue, sw_pvalue]])
#         self.Outputs.diagnostic_results.send(output_data)
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(OWResidualDiagnostics).run()