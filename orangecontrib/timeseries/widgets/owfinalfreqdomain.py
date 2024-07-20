import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from scipy import signal
import pywt


class OWFrequencyDomainTransforms(widget.OWWidget):
    name = "Frequency Domain Transforms"
    description = "Apply Fourier and Wavelet transforms to time series data"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        transformed_data = Output("Transformed Data", Table)

    want_main_area = True

    # Settings
    transform_method = Setting(0)  # 0: Fourier, 1: Wavelet
    target_variable = Setting("")
    wavelet_type = Setting('db1')
    wavelet_level = Setting(1)

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
            orientation="horizontal", callback=self.on_target_changed)

        # Transform method selection
        self.transform_combo = gui.comboBox(
            box, self, "transform_method", label="Transform Method:",
            items=["Fourier", "Wavelet"],
            orientation="horizontal", callback=self.apply_transform)

        # Parameters box
        params_box = gui.widgetBox(self.controlArea, "Parameters")

        # Wavelet type
        self.wavelet_types = ['haar', 'db1', 'db2', 'db3', 'sym2', 'sym3', 'coif1', 'coif2']

        self.wavelet_combo = gui.comboBox(
            params_box, self, "wavelet_type", label="Wavelet Type:",
            items=self.wavelet_types,
            orientation="horizontal", callback=self.on_wavelet_changed)

        # Wavelet level
        self.level_spin = gui.spin(
            params_box, self, "wavelet_level", minv=1, maxv=10,
            label="Wavelet Level:", callback=self.on_wavelet_changed)

        # Set up the main area with two plot widgets
        self.original_plot = PlotWidget(background="w")
        self.transformed_plot = PlotWidget(background="w")

        gui.vBox(self.mainArea).layout().addWidget(self.original_plot)
        gui.vBox(self.mainArea).layout().addWidget(self.transformed_plot)

    @Inputs.time_series
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.info_label.setText(f"{len(data)} instances on input.")
            self.time_variable = data.time_variable if isinstance(data.time_variable, TimeVariable) else None

            # Update target variable combo box options
            self.target_combo.clear()
            for var in data.domain.variables:
                if var.is_continuous and not isinstance(var, TimeVariable):
                    self.target_combo.addItem(var.name)

            # Ensure target_variable is a string and exists in the data
            if self.target_variable not in data.domain:
                self.target_variable = self.target_combo.itemText(0)

            index = self.target_combo.findText(self.target_variable)
            if index >= 0:
                self.target_combo.setCurrentIndex(index)
            else:
                self.target_variable = self.target_combo.itemText(0)
                self.target_combo.setCurrentIndex(0)

            self.apply_transform()
        else:
            self.info_label.setText("No data on input.")
            self.clear_plots()

    def on_target_changed(self):
        self.target_variable = self.target_combo.currentText()
        self.apply_transform()



    def fourier_transform(self, data):
        return np.abs(np.fft.fft(data))

    def on_wavelet_changed(self):
        self.wavelet_type = self.wavelet_combo.currentText()
        self.apply_transform()

    def apply_transform(self):
        if self.data is None or not self.target_variable:
            return

        self.error()  # Clear any previous error messages
        value_var = self.data.domain[self.target_variable]
        y_values = self.data.get_column(value_var)

        if self.transform_method == 0:  # Fourier
            transformed = self.fourier_transform(y_values)
        else:  # Wavelet
            transformed = self.wavelet_transform(y_values)

        self.plot_data(self.original_plot, y_values, "Original Data")
        self.plot_data(self.transformed_plot, transformed, "Transformed Data")

        # Create output data
        domain = Domain([ContinuousVariable("Transformed")])
        output_data = Table.from_numpy(domain, transformed.reshape(-1, 1))
        self.Outputs.transformed_data.send(output_data)

    def wavelet_transform(self, data):
        try:
            coeffs = pywt.wavedec(data, self.wavelet_type, level=self.wavelet_level)
            return np.concatenate(coeffs)
        except ValueError as e:
            self.error(f"Wavelet transform error: {str(e)}")
            return np.zeros_like(data)

    def plot_data(self, plot_widget, values, title):
        plot_widget.clear()
        plot_widget.plot(values, pen=pg.mkPen(color=(0, 0, 255), width=2))
        plot_widget.setTitle(title)
        plot_widget.setLabel('left', 'Value')
        plot_widget.setLabel('bottom', 'Time/Frequency')

    def clear_plots(self):
        self.original_plot.clear()
        self.transformed_plot.clear()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWFrequencyDomainTransforms).run()