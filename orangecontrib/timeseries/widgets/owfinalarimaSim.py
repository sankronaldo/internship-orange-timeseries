import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.widget import Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg


class OWARIMASimulator(widget.OWWidget):
    name = "ARIMA/SARIMA Simulator"
    description = "Simulate ARIMA or SARIMA time series"
    icon = "icons/final.svg"
    priority = 10

    class Outputs:
        simulated_data = Output("Simulated Data", Table)

    want_main_area = True

    # Settings
    model_type = Setting(0)  # 0: ARIMA, 1: SARIMA
    p = Setting(1)
    d = Setting(1)
    q = Setting(1)
    P = Setting(1)
    D = Setting(1)
    Q = Setting(1)
    S = Setting(12)
    n_points = Setting(100)
    ar_coef = Setting("")
    ma_coef = Setting("")
    sar_coef = Setting("")
    sma_coef = Setting("")

    def __init__(self):
        super().__init__()

        self.simulated_data = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Model Parameters")

        # Model type selection
        gui.comboBox(box, self, "model_type", items=["ARIMA", "SARIMA"],
                     label="Model Type:", orientation="horizontal", callback=self.on_model_type_changed)

        # ARIMA parameters
        arima_box = gui.widgetBox(box, "ARIMA Parameters")
        gui.spin(arima_box, self, "p", 0, 10, label="p (AR order):", callback=self.on_param_changed)
        gui.spin(arima_box, self, "d", 0, 2, label="d (Differencing):", callback=self.on_param_changed)
        gui.spin(arima_box, self, "q", 0, 10, label="q (MA order):", callback=self.on_param_changed)

        # SARIMA parameters
        self.sarima_box = gui.widgetBox(box, "SARIMA Parameters")
        gui.spin(self.sarima_box, self, "P", 0, 10, label="P (Seasonal AR):", callback=self.on_param_changed)
        gui.spin(self.sarima_box, self, "D", 0, 2, label="D (Seasonal Diff):", callback=self.on_param_changed)
        gui.spin(self.sarima_box, self, "Q", 0, 10, label="Q (Seasonal MA):", callback=self.on_param_changed)
        gui.spin(self.sarima_box, self, "S", 1, 365, label="S (Seasonal Period):", callback=self.on_param_changed)

        # Coefficient inputs
        coef_box = gui.widgetBox(box, "Model Coefficients")
        gui.lineEdit(coef_box, self, "ar_coef", label="AR Coefficients:")
        gui.lineEdit(coef_box, self, "ma_coef", label="MA Coefficients:")
        self.sar_input = gui.lineEdit(self.sarima_box, self, "sar_coef", label="SAR Coefficients:")
        self.sma_input = gui.lineEdit(self.sarima_box, self, "sma_coef", label="SMA Coefficients:")

        # Number of points
        gui.spin(box, self, "n_points", 1, 10000, label="Number of points:")

        # Generate button
        self.generate_button = gui.button(self.controlArea, self, "Generate Series", callback=self.generate_series)

        # Set up the main area with plot widget
        self.plot_widget = PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot_widget)

        self.on_model_type_changed()

    def on_model_type_changed(self):
        self.sarima_box.setVisible(self.model_type == 1)

    def on_param_changed(self):
        pass  # You can add any necessary logic here

    def generate_series(self):
        ar = np.array([float(x) for x in self.ar_coef.split()] if self.ar_coef else [])
        ma = np.array([float(x) for x in self.ma_coef.split()] if self.ma_coef else [])
        sar = np.array([float(x) for x in self.sar_coef.split()] if self.sar_coef else [])
        sma = np.array([float(x) for x in self.sma_coef.split()] if self.sma_coef else [])

        model = {
            'order': (self.p, self.d, self.q),
            'ar': ar,
            'ma': ma
        }

        if self.model_type == 1:  # SARIMA
            model['seasonal_order'] = (self.P, self.D, self.Q, self.S)
            model['sar'] = sar
            model['sma'] = sma

        series = arima_sim(model, self.n_points)

        self.simulated_data = self.create_orange_table(series)
        self.Outputs.simulated_data.send(self.simulated_data)

        self.plot_series(series)

    def create_orange_table(self, series):
        domain = Domain([ContinuousVariable("value")])
        return Table.from_numpy(domain, series.reshape(-1, 1))

    def plot_series(self, series):
        self.plot_widget.clear()
        self.plot_widget.plot(np.arange(len(series)), series, pen=pg.mkPen(color=(0, 0, 255), width=2))
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setTitle('Simulated ARIMA/SARIMA Series')


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWARIMASimulator).run()



# class ARIMASimulatorWidget(widget.OWWidget):
#     name = "ARIMA/SARIMA Simulator"
#     description = "Generate ARIMA or SARIMA time series"
#     icon = "icons/arima.svg"  # You'll need to create this icon
#
#     # Define inputs and outputs
#     class Outputs:
#         data = Output("Data", Table)
#
#     # Settings
#     model_type = settings.Setting("ARIMA")
#     p = settings.Setting(0)
#     d = settings.Setting(0)
#     q = settings.Setting(0)
#     P = settings.Setting(0)
#     D = settings.Setting(0)
#     Q = settings.Setting(0)
#     s = settings.Setting(1)
#     ar_coef = settings.Setting("")
#     ma_coef = settings.Setting("")
#     sar_coef = settings.Setting("")
#     sma_coef = settings.Setting("")
#     n_points = settings.Setting(100)
#
#     def __init__(self):
#         super().__init__()
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Model Parameters")
#
#         # Model type selection
#         gui.comboBox(box, self, "model_type", label="Model Type:",
#                      items=["ARIMA", "SARIMA"], callback=self.update_gui)
#
#         # ARIMA parameters
#         self.p_spin = gui.spin(box, self, "p", 0, 10, label="p:")
#         self.d_spin = gui.spin(box, self, "d", 0, 10, label="d:")
#         self.q_spin = gui.spin(box, self, "q", 0, 10, label="q:")
#
#         # SARIMA parameters (initially hidden)
#         self.sarima_box = gui.widgetBox(box)
#         self.P_spin = gui.spin(self.sarima_box, self, "P", 0, 10, label="P:")
#         self.D_spin = gui.spin(self.sarima_box, self, "D", 0, 10, label="D:")
#         self.Q_spin = gui.spin(self.sarima_box, self, "Q", 0, 10, label="Q:")
#         self.s_spin = gui.spin(self.sarima_box, self, "s", 1, 100, label="s:")
#
#         # Coefficient inputs
#         gui.lineEdit(box, self, "ar_coef", label="AR Coefficients:")
#         gui.lineEdit(box, self, "ma_coef", label="MA Coefficients:")
#         gui.widgetLabel(box, "Enter coefficients as space-separated values:")
#         gui.widgetLabel(box, "Example: 0.7 -0.3")
#         self.sar_input = gui.lineEdit(self.sarima_box, self, "sar_coef", label="SAR Coefficients:")
#         self.sma_input = gui.lineEdit(self.sarima_box, self, "sma_coef", label="SMA Coefficients:")
#
#         # Number of points
#         gui.spin(box, self, "n_points", 1, 10000, label="Number of points:")
#
#         # Generate button
#         self.generate_button = gui.button(box, self, "Generate", callback=self.generate_series)
#
#         self.update_gui()
#
#     def update_gui(self):
#         sarima_visible = self.model_type == "SARIMA"
#         self.sarima_box.setVisible(sarima_visible)
#         self.P_spin.setEnabled(sarima_visible)
#         self.D_spin.setEnabled(sarima_visible)
#         self.Q_spin.setEnabled(sarima_visible)
#         self.s_spin.setEnabled(sarima_visible)
#         self.sar_input.setEnabled(sarima_visible)
#         self.sma_input.setEnabled(sarima_visible)
#
#     def generate_series(self):
#         # Parse coefficients
#         ar = np.array([float(x) for x in self.ar_coef.split()] if self.ar_coef else [])
#         ma = np.array([float(x) for x in self.ma_coef.split()] if self.ma_coef else [])
#         sar = np.array([float(x) for x in self.sar_coef.split()] if self.sar_coef else [])
#         sma = np.array([float(x) for x in self.sma_coef.split()] if self.sma_coef else [])
#
#         # Create model dictionary
#         model = {
#             'order': (self.p, self.d, self.q),
#             'ar': ar,
#             'ma': ma
#         }
#
#         if self.model_type == "SARIMA":
#             model['seasonal_order'] = (self.P, self.D, self.Q, self.s)
#             model['sar'] = sar
#             model['sma'] = sma
#
#         # Generate series
#         series = arima_sim(model, self.n_points)
#
#         # Create Orange Table
#         domain = Domain([ContinuousVariable("value")])
#         table = Table.from_numpy(domain, series.reshape(-1, 1))
#
#         # Send data to output
#         self.Outputs.data.send(table)
#
#
# # Don't forget to register the widget!
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#     WidgetPreview(ARIMASimulatorWidget).run()


def arima_sim(model, n, rand_gen=np.random.normal, **kwargs):
    """
    Simulate from an ARIMA or SARIMA Model
    Parameters:
    model (dict): A dictionary with keys 'order', optional 'seasonal_order', 'ar', 'ma', optional 'sar', 'sma'
    n (int): Length of output series
    rand_gen (function): Function to generate innovations (default: np.random.normal)
    **kwargs: Additional arguments for rand_gen
    Returns:
    numpy.array: Simulated time series
    """
    if not isinstance(model, dict):
        raise ValueError("Model must be a dictionary")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    p, d, q = model.get('order', (0, 0, 0))
    P, D, Q, m = model.get('seasonal_order', (0, 0, 0, 1))

    if not all(isinstance(x, int) and x >= 0 for x in (p, d, q, P, D, Q, m)):
        raise ValueError("All order parameters must be non-negative integers")

    ar = np.array(model.get('ar', []))
    ma = np.array(model.get('ma', []))
    sar = np.array(model.get('sar', []))
    sma = np.array(model.get('sma', []))

    if len(ar) != p or len(ma) != q or len(sar) != P or len(sma) != Q:
        raise ValueError("AR/MA coefficient lengths do not match specified orders")

    max_lag = max(p, q, m * P, m * Q, 1)
    ar_full = np.zeros(max_lag)
    ma_full = np.zeros(max_lag)
    ar_full[:p] = ar
    ma_full[:q] = ma
    for i in range(P):
        ar_full[m * i:m * i + p] += sar[i] * ar
        ar_full[m * (i + 1) - 1] += sar[i]
    for i in range(Q):
        ma_full[m * i:m * i + q] += sma[i] * ma
        ma_full[m * (i + 1) - 1] += sma[i]

    # Ensure we generate an array of random numbers
    if callable(rand_gen):
        innovations = rand_gen(size=n + max_lag, **kwargs)
    else:
        innovations = np.random.normal(size=n + max_lag, **kwargs)

    series = np.zeros(n + max_lag)
    for t in range(max_lag, n + max_lag):
        series[t] = np.sum(ar_full * series[t - max_lag:t][::-1]) + \
                    np.sum(ma_full * innovations[t - max_lag:t][::-1]) + \
                    innovations[t]

    series = series[max_lag:]

    # Apply regular differencing
    for _ in range(d):
        series = np.diff(series)

    # Apply seasonal differencing
    for _ in range(D):
        series = series[m:] - series[:-m]

    # print(f"Generated series shape: {series.shape}")
    # print(f"Series mean: {np.mean(series):.4f}, std: {np.std(series):.4f}")
    return series