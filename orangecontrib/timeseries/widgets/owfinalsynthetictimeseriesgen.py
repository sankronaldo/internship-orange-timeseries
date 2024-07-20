import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg

class OWSyntheticTimeSeriesGenerator(widget.OWWidget):
    name = "Synthetic Time Series Generator"
    description = "Generate synthetic time series data using SARIMA models"
    icon = "icons/final.svg"
    priority = 10

    class Outputs:
        time_series = Output("Time series", Table)

    # Standard coefficient values
    STANDARD_COEFFS = {
        "AR": [0.3, 0.6, 0.9, -0.3, -0.6, -0.9],
        "MA": [0.3, 0.6, 0.9, -0.3, -0.6, -0.9],
        "SAR": [0.3, 0.6, 0.9],
        "SMA": [0.3, 0.6, 0.9]
    }

    # Widget parameters
    n_points = Setting(100)
    include_seasonal = Setting(True)
    seasonal_period = Setting(12)

    want_main_area = True

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "Generator Settings")
        gui.spin(box, self, "n_points", 50, 1000, label="Number of data points:", callback=self.param_changed)
        gui.checkBox(box, self, "include_seasonal", "Include Seasonal Component", callback=self.param_changed)
        self.seasonal_spin = gui.spin(box, self, "seasonal_period", 2, 365, label="Seasonal Period:", callback=self.param_changed)

        gui.button(self.controlArea, self, "Generate", callback=self.generate_data)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

        self.on_seasonal_changed()
        self.generate_data()

    def on_seasonal_changed(self):
        self.seasonal_spin.setEnabled(self.include_seasonal)

    def param_changed(self):
        self.on_seasonal_changed()
        self.generate_data()

    def generate_data(self):
        # Randomly choose model order
        p = np.random.randint(0, 3)
        d = np.random.randint(0, 2)
        q = np.random.randint(0, 3)

        # Randomly choose coefficients
        ar = np.random.choice(self.STANDARD_COEFFS["AR"], size=p, replace=False)
        ma = np.random.choice(self.STANDARD_COEFFS["MA"], size=q, replace=False)

        order = (p, d, q)
        seasonal_order = (0, 0, 0, 0)

        if self.include_seasonal:
            P = np.random.randint(0, 2)
            D = np.random.randint(0, 2)
            Q = np.random.randint(0, 2)
            s = self.seasonal_period

            sar = np.random.choice(self.STANDARD_COEFFS["SAR"], size=P, replace=False)
            sma = np.random.choice(self.STANDARD_COEFFS["SMA"], size=Q, replace=False)

            seasonal_order = (P, D, Q, s)
        else:
            sar = []
            sma = []

        model = SARIMAX(endog=np.zeros(self.n_points),
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        # Generate the synthetic data
        params = [*ar, *ma, *sar, *sma, 1]  # 1 is for sigma^2
        synthetic_data = model.simulate(params=params, nsimulations=self.n_points)

        # Plot the data
        self.plot_widget.clear()
        self.plot_widget.plot(np.arange(self.n_points), synthetic_data, pen=pg.mkPen(color=(0, 0, 255), width=2))
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle(f'Synthetic Time Series: SARIMA{order}{seasonal_order}')

        # Create Orange Table
        value_var = ContinuousVariable('Value')
        domain = Domain([value_var])
        table = Table.from_numpy(domain, synthetic_data.reshape(-1, 1))

        # Send data to output
        self.Outputs.time_series.send(table)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSyntheticTimeSeriesGenerator).run()