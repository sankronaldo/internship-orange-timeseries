import numpy as np
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtWidgets import QScrollArea, QGridLayout, QWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw


class OWDynamicTimeWarping(widget.OWWidget):
    name = "Dynamic Time Warping"
    description = "Compute and visualize DTW between two time series"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        data_a = Input("Time Series A", Table)
        data_b = Input("Time Series B", Table)

    class Outputs:
        dtw_score = Output("DTW Score", float)

    want_main_area = True

    target_a_index = Setting(0)
    target_b_index = Setting(0)

    def __init__(self):
        super().__init__()

        self.data_a = None
        self.data_b = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target selection
        target_box = gui.widgetBox(self.controlArea, "Target Selection")
        self.target_a_combo = gui.comboBox(
            target_box, self, "target_a_index", label="Target A:",
            callback=self.compute_dtw)
        self.target_b_combo = gui.comboBox(
            target_box, self, "target_b_index", label="Target B:",
            callback=self.compute_dtw)

        # Set up the main area with a scroll area for plots
        self.scroll_area = QScrollArea()
        self.mainArea.layout().addWidget(self.scroll_area)
        self.plot_widget = QWidget()
        self.plot_layout = QGridLayout()
        self.plot_widget.setLayout(self.plot_layout)
        self.scroll_area.setWidget(self.plot_widget)
        self.scroll_area.setWidgetResizable(True)

    @Inputs.data_a
    def set_data_a(self, data):
        self.data_a = data
        self.update_target_combo(self.target_a_combo, data, 'target_a_index')

    @Inputs.data_b
    def set_data_b(self, data):
        self.data_b = data
        self.update_target_combo(self.target_b_combo, data, 'target_b_index')

    def update_target_combo(self, combo, data, index_attr):
        combo.clear()
        if data is not None:
            combo.addItems([var.name for var in data.domain.variables if var.is_continuous])
            if getattr(self, index_attr) >= combo.count():
                setattr(self, index_attr, 0)
        self.compute_dtw()

    def compute_dtw(self):
        self.clear_plots()

        if self.data_a is None or self.data_b is None:
            self.info_label.setText("Waiting for input data.")
            return

        if self.target_a_combo.count() == 0 or self.target_b_combo.count() == 0:
            self.info_label.setText("No suitable variables for DTW.")
            return

        var_a = self.data_a.domain.variables[self.target_a_index]
        var_b = self.data_b.domain.variables[self.target_b_index]

        time_series_a = self.data_a.get_column(var_a)
        time_series_b = self.data_b.get_column(var_b)

        # Standardize the time series data
        scaler = StandardScaler()
        time_series_a_scaled = scaler.fit_transform(time_series_a.reshape(-1, 1)).flatten()
        time_series_b_scaled = scaler.fit_transform(time_series_b.reshape(-1, 1)).flatten()

        # Calculate DTW distance and obtain the warping paths
        distance, paths = dtw.warping_paths(time_series_a_scaled, time_series_b_scaled, use_c=False)
        best_path = dtw.best_path(paths)
        similarity_score = distance / len(best_path)

        self.info_label.setText(f"DTW Similarity Score: {similarity_score:.4f}")
        self.Outputs.dtw_score.send(similarity_score)

        self.plot_results(time_series_a, time_series_b, time_series_a_scaled, time_series_b_scaled, best_path,
                          var_a.name, var_b.name)

    def plot_results(self, time_series_a, time_series_b, time_series_a_scaled, time_series_b_scaled, best_path, name_a,
                     name_b):
        # Original Time Series A Plot
        plot_a = PlotWidget(background="w")
        self.plot_layout.addWidget(plot_a, 0, 0)
        plot_a.plot(time_series_a, pen=pg.mkPen(color=(0, 0, 255), width=2))
        plot_a.setTitle(f"Original Time Series A: {name_a}")

        # Original Time Series B Plot
        plot_b = PlotWidget(background="w")
        self.plot_layout.addWidget(plot_b, 0, 1)
        plot_b.plot(time_series_b, pen=pg.mkPen(color=(255, 0, 0), width=2))
        plot_b.setTitle(f"Original Time Series B: {name_b}")

        # Shortest Path Plot
        plot_path = PlotWidget(background="w")
        self.plot_layout.addWidget(plot_path, 1, 0, 1, 2)
        plot_path.plot([p[0] for p in best_path], [p[1] for p in best_path],
                       pen=pg.mkPen(color=(0, 128, 0), width=3))
        plot_path.plot([0, max(len(time_series_a), len(time_series_b))],
                       [0, max(len(time_series_a), len(time_series_b))],
                       pen=pg.mkPen(color=(255, 165, 0), width=3, style=Qt.DashLine))
        plot_path.setTitle("Shortest Path (Best Path)")
        plot_path.setLabel('bottom', f"Series A: {name_a}")
        plot_path.setLabel('left', f"Series B: {name_b}")

        # DTW Alignment Plot
        plot_dtw = PlotWidget(background="w")
        self.plot_layout.addWidget(plot_dtw, 2, 0, 1, 2)
        plot_dtw.plot(time_series_a_scaled, pen=pg.mkPen(color=(0, 0, 255), width=3),symbol='+')
        plot_dtw.plot(time_series_b_scaled, pen=pg.mkPen(color=(255, 0, 0), width=3), symbol='x')
        for a, b in best_path:
            plot_dtw.plot([a, b], [time_series_a_scaled[a], time_series_b_scaled[b]],
                          pen=pg.mkPen(color=(128, 128, 128), width=1))
        plot_dtw.setTitle("Point-to-Point Comparison After DTW Alignment")

    def clear_plots(self):
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDynamicTimeWarping).run()