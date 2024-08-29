import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from scipy import stats


class OWDataTransforms(widget.OWWidget):
    name = "Data Transforms"
    description = "Apply various transforms to time series data"
    icon = "icons/ow_transforms.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        transformed_data = Output("Transformed Data", Table)

    want_main_area = True

    # Settings
    transform_method = Setting(0)  # 0: Moving Average, 1: Box-Cox, 2: Log
    window_length = Setting(3)  # For Moving Average
    lambda_param = Setting(0.0)  # For Box-Cox
    target_variable = Setting("")

    def __init__(self):
        super().__init__()

        self.data = None
        self.time_variable = None
        self.optimal_lambda = None

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
            items=["Moving Average", "Box-Cox", "Log"],
            orientation="horizontal", callback=self.apply_transform)

        # Parameters box
        params_box = gui.widgetBox(self.controlArea, "Parameters")

        # Moving Average window length
        self.window_spin = gui.spin(
            params_box, self, "window_length", minv=2, maxv=100,
            label="Window Length:", callback=self.apply_transform)

        # Box-Cox lambda
        self.lambda_spin = gui.doubleSpin(
            params_box, self, "lambda_param", minv=-5, maxv=5, step=0.01,
            label="Lambda:", callback=self.apply_transform)

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

            self.calculate_optimal_lambda()
            self.apply_transform()
        else:
            self.info_label.setText("No data on input.")
            self.clear_plots()

    def on_target_changed(self):
        self.target_variable = self.target_combo.currentText()
        self.calculate_optimal_lambda()
        self.apply_transform()

    def calculate_optimal_lambda(self):
        if self.data is not None and self.target_variable:
            value_var = self.data.domain[self.target_variable]
            y_values = self.data.get_column(value_var)
            # Ensure y_values are positive for Box-Cox transform
            if np.any(y_values <= 0):
                self.optimal_lambda = 0.0  # Log transform
            else:
                _, self.optimal_lambda = stats.boxcox(y_values)
            self.lambda_param = round(self.optimal_lambda, 2)
            self.lambda_spin.setValue(self.lambda_param)

    def apply_transform(self):
        if self.data is None or not self.target_variable:
            return

        value_var = self.data.domain[self.target_variable]
        y_values = self.data.get_column(value_var)

        if self.transform_method == 0:  # Moving Average
            transformed = self.moving_average(y_values, self.window_length)
        elif self.transform_method == 1:  # Box-Cox
            # Ensure y_values are positive for Box-Cox transform
            if np.any(y_values <= 0):
                transformed = np.log1p(y_values)  # Use log1p as a safe alternative
            else:
                transformed = stats.boxcox(y_values, lmbda=self.lambda_param)
        else:  # Log
            transformed = np.log1p(y_values)  # Use log1p as a safe alternative

        self.plot_data(self.original_plot, y_values, "Original Data")
        self.plot_data(self.transformed_plot, transformed, "Transformed Data")

        # Create output data
        domain = Domain([ContinuousVariable("Transformed")])
        output_data = Table.from_numpy(domain, transformed.reshape(-1, 1))
        self.Outputs.transformed_data.send(output_data)

    def moving_average(self, data, window):
        return np.convolve(data, np.ones(window), 'valid') / window

    def plot_data(self, plot_widget, values, title):
        plot_widget.clear()
        plot_widget.plot(values, pen=pg.mkPen(color=(0, 0, 255), width=2))
        plot_widget.setTitle(title)
        plot_widget.setLabel('left', 'Value')
        plot_widget.setLabel('bottom', 'Time')

    def clear_plots(self):
        self.original_plot.clear()
        self.transformed_plot.clear()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDataTransforms).run()






# import numpy as np
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, Domain, ContinuousVariable
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.visualize.utils.plotutils import PlotWidget
# import pyqtgraph as pg
# from PyQt5.QtGui import QFont
# from scipy import stats
#
#
# class OWDataTransforms(widget.OWWidget):
#     name = "Data Transforms"
#     description = "Apply various transforms to time series data"
#     icon = "icons/datatransforms.svg"
#     priority = 10
#
#     class Inputs:
#         time_series = Input("Time series", Table)
#
#     class Outputs:
#         transformed_data = Output("Transformed Data", Table)
#
#     want_main_area = True
#
#     # Settings
#     transform_method = Setting(0)  # 0: Moving Average, 1: Box-Cox, 2: Log
#     window_length = Setting(3)  # For Moving Average
#     lambda_param = Setting(None)  # For Box-Cox, None means auto-calculate
#     target_variable = Setting("")  # Ensure this is initialized as a string
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#         self.time_variable = None
#         self.optimal_lambda = None
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(box, "No data on input.")
#
#         # Target variable selection
#         self.target_combo = gui.comboBox(
#             box, self, "target_variable", label="Target Variable:",
#             orientation="horizontal", callback=self.apply_transform)
#
#         # Transform method selection
#         self.transform_combo = gui.comboBox(
#             box, self, "transform_method", label="Transform Method:",
#             items=["Moving Average", "Box-Cox", "Log"],
#             orientation="horizontal", callback=self.apply_transform)
#
#         # Parameters box
#         params_box = gui.widgetBox(self.controlArea, "Parameters")
#
#         # Moving Average window length
#         self.window_spin = gui.spin(
#             params_box, self, "window_length", minv=2, maxv=100,
#             label="Window Length:", callback=self.apply_transform)
#
#         # Box-Cox lambda
#         self.lambda_spin = gui.doubleSpin(
#             params_box, self, "lambda_param", minv=-5, maxv=5, step=0.1,
#             label="Lambda (auto if blank):", callback=self.apply_transform)
#         self.lambda_spin.setSpecialValueText("Auto")
#         self.lambda_spin.setValue(self.lambda_spin.minimum() - self.lambda_spin.singleStep())
#
#         # Set up the main area with two plot widgets
#         self.original_plot = PlotWidget(background="w")
#         self.transformed_plot = PlotWidget(background="w")
#
#         gui.vBox(self.mainArea).layout().addWidget(self.original_plot)
#         gui.vBox(self.mainArea).layout().addWidget(self.transformed_plot)
#
#     @Inputs.time_series
#     def set_data(self, data):
#         self.data = data
#         if data is not None:
#             self.info_label.setText(f"{len(data)} instances on input.")
#             self.time_variable = getattr(data, 'time_variable', None)
#
#             # Update target variable combo box options
#             self.target_combo.clear()
#             for var in data.domain.variables:
#                 if var.is_continuous:
#                     self.target_combo.addItem(var.name)
#
#             # Ensure target_variable is a string and exists in the data
#             if isinstance(self.target_variable, int) or self.target_variable not in data.domain:
#                 self.target_variable = self.target_combo.itemText(0)
#
#             index = self.target_combo.findText(self.target_variable)
#             if index >= 0:
#                 self.target_combo.setCurrentIndex(index)
#             else:
#                 self.target_variable = self.target_combo.itemText(0)
#                 self.target_combo.setCurrentIndex(0)
#
#             self.apply_transform()
#         else:
#             self.info_label.setText("No data on input.")
#             self.clear_plots()
#
#     def apply_transform(self):
#         if self.data is None or not self.target_variable:
#             return
#
#         value_var = self.data.domain[self.target_variable]
#         y_values = self.data.get_column(value_var)
#
#         if self.transform_method == 0:  # Moving Average
#             transformed = self.moving_average(y_values, self.window_length)
#         elif self.transform_method == 1:  # Box-Cox
#             if self.lambda_param is None or self.lambda_param == self.lambda_spin.minimum() - self.lambda_spin.singleStep():
#                 transformed, self.optimal_lambda = stats.boxcox(y_values)
#                 self.lambda_spin.setValue(self.optimal_lambda)
#             else:
#                 transformed = stats.boxcox(y_values, lmbda=self.lambda_param)
#         else:  # Log
#             transformed = np.log(y_values)
#
#         self.plot_data(self.original_plot, y_values, "Original Data")
#         self.plot_data(self.transformed_plot, transformed, "Transformed Data")
#
#         # Create output data
#         domain = Domain([ContinuousVariable("Transformed")])
#         output_data = Table.from_numpy(domain, transformed.reshape(-1, 1))
#         self.Outputs.transformed_data.send(output_data)
#
#     def moving_average(self, data, window):
#         return np.convolve(data, np.ones(window), 'valid') / window
#
#     def plot_data(self, plot_widget, values, title):
#         plot_widget.clear()
#         plot_widget.plot(values, pen=pg.mkPen(color=(0, 0, 255), width=2))
#         plot_widget.setTitle(title)
#         plot_widget.setLabel('left', 'Value')
#         plot_widget.setLabel('bottom', 'Time')
#
#     def clear_plots(self):
#         self.original_plot.clear()
#         self.transformed_plot.clear()
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#     WidgetPreview(OWDataTransforms).run()