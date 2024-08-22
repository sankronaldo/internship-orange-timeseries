import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table, Domain, TimeVariable, ContinuousVariable, StringVariable
import numpy as np
from arch import arch_model
from PyQt5.QtWidgets import QTextEdit, QTabWidget, QVBoxLayout, QWidget, QSplitter
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

class ARCHGARCHWidget(widget.OWWidget):
    name = "ARCH/GARCH"
    description = "ARCH and GARCH modeling and forecasting"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        data = widget.Input("Time series data", Table)

    class Outputs:
        fitted_values = widget.Output("Fitted Values", Table)
        residuals = widget.Output("Residuals", Table)
        model_summary = widget.Output("Model Summary", str)
        model = widget.Output("Model", Table)

    # Widget parameters
    p = settings.Setting(1)
    q = settings.Setting(1)
    mean_type = settings.Setting(0)
    vol_type = settings.Setting(1)

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "ARCH/GARCH Parameters")
        gui.spin(box, self, "p", 0, 5, label="p (ARCH order)", callback=self.apply)
        gui.spin(box, self, "q", 0, 5, label="q (GARCH order)", callback=self.apply)
        gui.comboBox(box, self, "mean_type", label="Mean Model",
                     items=["Constant Mean", "Zero Mean", "AR", "HAR", "LS"], callback=self.apply)
        gui.comboBox(box, self, "vol_type", label="Volatility Model",
                     items=["ARCH", "GARCH"], callback=self.apply)

        # Main area layout
        splitter = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(splitter)

        # Text area for displaying model summary
        self.summary_box = QTextEdit(readOnly=True)
        self.summary_box.setFont(QFont("Courier"))
        splitter.addWidget(self.summary_box)

        # Tab widget for plots
        self.plot_tabs = QTabWidget()
        splitter.addWidget(self.plot_tabs)

        # Create tabs for different plots
        self.usual_plot = FigureCanvas(plt.Figure(figsize=(5, 4)))
        self.qq_plot = FigureCanvas(plt.Figure(figsize=(5, 4)))
        self.acf_plot = FigureCanvas(plt.Figure(figsize=(5, 4)))

        self.plot_tabs.addTab(self.usual_plot, "Usual Plot")
        self.plot_tabs.addTab(self.qq_plot, "QQ Plot")
        self.plot_tabs.addTab(self.acf_plot, "ACF of Squared Residuals")

        self.data = None

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if self.data is not None:
            self.apply()
        else:
            self.clear_outputs()

    def clear_outputs(self):
        self.summary_box.clear()
        self.clear_plot(self.usual_plot)
        self.clear_plot(self.qq_plot)
        self.clear_plot(self.acf_plot)
        self.Outputs.fitted_values.send(None)
        self.Outputs.residuals.send(None)
        self.Outputs.model_summary.send(None)
        self.Outputs.model.send(None)

    def clear_plot(self, canvas):
        canvas.figure.clear()
        canvas.draw()

    def apply(self):
        if self.data is None:
            self.clear_outputs()
            return

        # Assume the last column is the target variable
        y = self.data.Y.ravel()

        # Set up the model
        mean_types = ['Constant', 'Zero', 'AR', 'HAR', 'LS']
        vol_types = ['ARCH', 'GARCH']

        model = arch_model(y,
                           p=self.p,
                           q=self.q if self.vol_type == 1 else 0,  # q is 0 for ARCH
                           mean=mean_types[self.mean_type],
                           vol=vol_types[self.vol_type],
                           dist='normal')

        # Fit the model
        results = model.fit(disp='off')

        # Get fitted values and residuals
        fitted_values = results.conditional_volatility
        residuals = results.resid

        # Create output tables
        domain_fitted = Domain([ContinuousVariable("Fitted Values")])
        domain_residuals = Domain([ContinuousVariable("Residuals")])

        fitted_table = Table.from_numpy(domain_fitted, fitted_values.reshape(-1, 1))
        residuals_table = Table.from_numpy(domain_residuals, residuals.reshape(-1, 1))

        # Get model summary
        summary = results.summary()

        # Display model summary in the text area
        self.summary_box.setPlainText(str(summary))

        # Prepare model output as Orange.data.Table
        param_names = results.params.index
        param_values = results.params.values
        model_data = pd.DataFrame({
            'Parameter': param_names,
            'Value': param_values
        })
        model_domain = Domain([ContinuousVariable("Value")], metas=[StringVariable("Parameter")])
        model_table = Table.from_numpy(
            model_domain,
            X=model_data['Value'].values.reshape(-1, 1),
            metas=model_data['Parameter'].values.reshape(-1, 1)
        )

        # Create plots
        self.create_usual_plot(y, fitted_values)
        self.create_qq_plot(residuals)
        self.create_acf_plot(residuals)

        # Send outputs
        self.Outputs.fitted_values.send(fitted_table)
        self.Outputs.residuals.send(residuals_table)
        self.Outputs.model_summary.send(str(summary))
        self.Outputs.model.send(model_table)

    def create_usual_plot(self, y, fitted_values):
        self.clear_plot(self.usual_plot)
        ax = self.usual_plot.figure.add_subplot(111)
        ax.plot(y, label='Observed')
        ax.plot(fitted_values, label='Fitted')
        ax.legend()
        ax.set_title('Observed vs Fitted Values')
        self.usual_plot.draw()

    def create_qq_plot(self, residuals):
        self.clear_plot(self.qq_plot)
        ax = self.qq_plot.figure.add_subplot(111)
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals')
        self.qq_plot.draw()

    def create_acf_plot(self, residuals):
        self.clear_plot(self.acf_plot)
        ax = self.acf_plot.figure.add_subplot(111)
        squared_residuals = residuals**2
        plot_acf(squared_residuals, ax=ax, lags=20)
        ax.set_title('ACF of Squared Residuals')
        self.acf_plot.draw()

if __name__ == "__main__":
    WidgetPreview(ARCHGARCHWidget).run(Table("iris"))


