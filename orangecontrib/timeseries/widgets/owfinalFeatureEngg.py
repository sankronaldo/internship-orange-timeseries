import numpy as np
import pandas as pd
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable, DiscreteVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QWidget
from PyQt5.QtCore import pyqtSignal

class FeatureItem(QWidget):
    optionChanged = pyqtSignal()

    def __init__(self, parent, feature_type, value, remove_callback):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.feature_type = QComboBox()
        self.feature_type.addItems(["Lag", "Rolling Window", "EWM", "Percentage Change", "K-means"])
        self.feature_type.setCurrentText(feature_type)
        self.feature_type.currentTextChanged.connect(self.on_feature_type_changed)
        self.value = QLineEdit(str(value))
        self.value.textChanged.connect(self.optionChanged.emit)
        self.remove_button = QPushButton("-")
        self.remove_button.clicked.connect(remove_callback)

        layout.addWidget(self.feature_type)
        layout.addWidget(self.value)
        layout.addWidget(self.remove_button)
        self.setLayout(layout)

        self.on_feature_type_changed(feature_type)

    def on_feature_type_changed(self, feature_type):
        if feature_type == "Percentage Change":
            self.value.hide()
        else:
            self.value.show()
        self.optionChanged.emit()

class OWTimeSeriesFeatureEngineering(widget.OWWidget):
    name = "Feature Engineering"
    description = "Create new features for time series data"
    icon = "icons/ow_featengg.svg"
    priority = 10

    class Inputs:
        time_series = Input("Time series", Table)

    class Outputs:
        engineered_series = Output("Engineered series", Table)

    want_main_area = False

    target_variable = Setting("")
    auto_apply = Setting(False)
    features = Setting([])

    def __init__(self):
        super().__init__()
        self.data = None

        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            callback=self.on_target_variable_changed)

        self.feature_box = gui.widgetBox(self.controlArea, "Feature Engineering")
        self.feature_list = QListWidget()
        self.feature_box.layout().addWidget(self.feature_list)

        add_button = QPushButton("+")
        add_button.clicked.connect(self.add_feature)
        self.feature_box.layout().addWidget(add_button)

        gui.auto_apply(self.controlArea, self, "auto_apply", commit=self.commit)

    @Inputs.time_series
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.info_label.setText(f"{len(data)} instances on input.")
            self.target_combo.clear()
            vars = [var for var in data.domain.variables if isinstance(var, (TimeVariable, ContinuousVariable))]
            self.target_combo.addItems([var.name for var in vars])
            self.target_variable = self.target_combo.itemText(0) if self.target_combo.count() > 0 else ""
            self.feature_box.setEnabled(True)
        else:
            self.info_label.setText("No data on input.")
            self.target_variable = ""
            self.feature_box.setEnabled(False)
        self.commit()

    def add_feature(self):
        self.features.append({"type": "Lag", "value": "1"})
        self.update_feature_list()

    def update_feature_list(self):
        self.feature_list.clear()
        for feature in self.features:
            item = QListWidgetItem(self.feature_list)
            widget = FeatureItem(self.feature_list, feature['type'], feature['value'], lambda: self.remove_feature(item))
            widget.optionChanged.connect(self.settings_changed)
            item.setSizeHint(widget.sizeHint())
            self.feature_list.addItem(item)
            self.feature_list.setItemWidget(item, widget)

    def remove_feature(self, item):
        index = self.feature_list.row(item)
        del self.features[index]
        self.update_feature_list()
        self.settings_changed()

    def on_target_variable_changed(self):
        self.target_variable = self.target_combo.currentText()
        self.commit()

    def settings_changed(self):
        self.features = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            widget = self.feature_list.itemWidget(item)
            self.features.append({
                'type': widget.feature_type.currentText(),
                'value': widget.value.text()
            })
        self.commit()

    def commit(self):
        if self.data is None or not self.target_variable:
            self.Outputs.engineered_series.send(None)
            return

        df = pd.DataFrame({var.name: self.data.get_column(var.name) for var in self.data.domain.variables})
        target_series = df[self.target_variable]
        new_features = {}

        for feature in self.features:
            feature_type = feature['type']
            value = feature['value']

            try:
                if feature_type == "Lag":
                    lag = int(value)
                    new_features[f'{self.target_variable}_lag_{lag}'] = target_series.shift(lag)
                elif feature_type == "Rolling Window":
                    window = int(value)
                    new_features[f'{self.target_variable}_rolling_mean_{window}'] = target_series.rolling(
                        window=window).mean()
                    new_features[f'{self.target_variable}_rolling_std_{window}'] = target_series.rolling(
                        window=window).std()
                elif feature_type == "EWM":
                    span = float(value)
                    new_features[f'{self.target_variable}_ewm_{span}'] = target_series.ewm(span=span).mean()
                elif feature_type == "Percentage Change":
                    new_features[f'{self.target_variable}_pct_change'] = target_series.pct_change()
                elif feature_type == "K-means":
                    n_clusters = int(value)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    new_features[f'{self.target_variable}_kmeans_{n_clusters}'] = kmeans.fit_predict(
                        target_series.values.reshape(-1, 1))
            except Exception as e:
                print(f"Error processing {feature_type}: {str(e)}")
                continue

        for name, feature in new_features.items():
            df[name] = feature

        new_attrs = [
            var for var in self.data.domain.attributes
            if var.name in df.columns
        ]
        new_attrs += [
            ContinuousVariable(name) if not name.startswith(f'{self.target_variable}_kmeans') else
            DiscreteVariable(name, values=[str(i) for i in range(int(name.split('_')[-1]))])
            for name in new_features.keys()
        ]

        new_domain = Domain(
            new_attrs,
            self.data.domain.class_vars,
            self.data.domain.metas
        )

        new_X = df[df.columns.intersection([var.name for var in new_domain.attributes])].values
        new_Y = self.data.Y if self.data.Y.size else None
        new_metas = self.data.metas if self.data.metas.size else None

        new_table = Table.from_numpy(new_domain, new_X, new_Y, new_metas)
        self.Outputs.engineered_series.send(new_table)

        self.info_label.setText(f"Created {len(new_features)} new features. Output has {len(new_table)} rows.")

if __name__ == "__main__":
    WidgetPreview(OWTimeSeriesFeatureEngineering).run()







# import numpy as np
# import pandas as pd
# from Orange.widgets import widget, gui
# from Orange.widgets.settings import Setting
# from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
# from Orange.widgets.widget import Input, Output
# from Orange.widgets.utils.widgetpreview import WidgetPreview
# from sklearn.cluster import KMeans
# from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QWidget
# from PyQt5.QtCore import pyqtSignal
#
# class FeatureItem(QWidget):
#     optionChanged = pyqtSignal()
#
#     def __init__(self, parent, feature_type, value, remove_callback):
#         super().__init__(parent)
#         layout = QHBoxLayout()
#         self.feature_type = QComboBox()
#         self.feature_type.addItems(["Lag", "Rolling Window", "EWM", "Percentage Change", "K-means"])
#         self.feature_type.setCurrentText(feature_type)
#         self.feature_type.currentTextChanged.connect(self.on_feature_type_changed)
#         self.value = QLineEdit(str(value))
#         self.value.textChanged.connect(self.optionChanged.emit)
#         self.remove_button = QPushButton("-")
#         self.remove_button.clicked.connect(remove_callback)
#
#         layout.addWidget(self.feature_type)
#         layout.addWidget(self.value)
#         layout.addWidget(self.remove_button)
#         self.setLayout(layout)
#
#         self.on_feature_type_changed(feature_type)
#
#     def on_feature_type_changed(self, feature_type):
#         if feature_type == "Percentage Change":
#             self.value.hide()
#         else:
#             self.value.show()
#         self.optionChanged.emit()
#
# class OWTimeSeriesFeatureEngineering(widget.OWWidget):
#     name = "Time Series Feature Engineering"
#     description = "Create new features for time series data"
#     icon = "icons/TimeSeriesFeatureEngineering.svg"
#     priority = 10
#
#     class Inputs:
#         time_series = Input("Time series", Table)
#
#     class Outputs:
#         engineered_series = Output("Engineered series", Table)
#
#     want_main_area = False
#
#     target_variable = Setting("")
#     auto_apply = Setting(False)
#     features = Setting([])
#
#     class Error(widget.OWWidget.Error):
#         no_target_variable = widget.Msg("No target variable selected.")
#         invalid_input = widget.Msg("Invalid input: {}")
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#
#         box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(box, "No data on input.")
#
#         self.target_combo = gui.comboBox(
#             box, self, "target_variable", label="Target Variable:",
#             callback=self.on_target_variable_changed)
#
#         self.feature_box = gui.widgetBox(self.controlArea, "Feature Engineering")
#
#         self.feature_list = QListWidget()
#         self.feature_box.layout().addWidget(self.feature_list)
#
#         add_button = QPushButton("+")
#         add_button.clicked.connect(self.add_feature)
#         self.feature_box.layout().addWidget(add_button)
#
#         gui.auto_apply(self.controlArea, self, "auto_apply", commit=self.commit)
#
#         self.load_features()
#
#     def load_features(self):
#         for feature in self.features:
#             self.add_feature_item(feature['type'], feature['value'])
#
#     def add_feature(self):
#         self.add_feature_item("Lag", "1")
#
#     def add_feature_item(self, feature_type, value):
#         item = QListWidgetItem(self.feature_list)
#         feature_widget = FeatureItem(self.feature_list, feature_type, value, lambda: self.remove_feature(item))
#         feature_widget.optionChanged.connect(self.settings_changed)
#         item.setSizeHint(feature_widget.sizeHint())
#         self.feature_list.addItem(item)
#         self.feature_list.setItemWidget(item, feature_widget)
#         self.settings_changed()
#
#     def remove_feature(self, item):
#         self.feature_list.takeItem(self.feature_list.row(item))
#         self.settings_changed()
#
#     @Inputs.time_series
#     def set_data(self, data):
#         self.Error.clear()
#         if data is not None:
#             self.data = data
#             self.info_label.setText(f"{len(data)} instances on input.")
#
#             self.target_combo.clear()
#             time_vars = [var for var in data.domain.variables if isinstance(var, TimeVariable)]
#             cont_vars = [var for var in data.domain.variables if isinstance(var, ContinuousVariable)]
#
#             for var in time_vars + cont_vars:
#                 self.target_combo.addItem(var.name)
#
#             if self.target_variable in [var.name for var in data.domain.variables]:
#                 self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))
#             else:
#                 self.target_variable = self.target_combo.itemText(0) if self.target_combo.count() > 0 else ""
#
#             self.feature_box.setEnabled(True)
#         else:
#             self.data = None
#             self.target_variable = ""
#             self.info_label.setText("No data on input.")
#             self.feature_box.setEnabled(False)
#         self.commit()
#
#     def on_target_variable_changed(self):
#         self.target_variable = self.target_combo.currentText()
#         self.commit()
#
#     def settings_changed(self):
#         self.features = []
#         for i in range(self.feature_list.count()):
#             item = self.feature_list.item(i)
#             widget = self.feature_list.itemWidget(item)
#             self.features.append({
#                 'type': widget.feature_type.currentText(),
#                 'value': widget.value.text()
#             })
#         self.commit()
#
#     def commit(self):
#         self.Error.clear()
#         if self.data is None:
#             return
#
#         if not self.target_variable:
#             self.Error.no_target_variable()
#             return
#
#         # Convert Orange Table to pandas DataFrame
#         df = pd.DataFrame({var.name: self.data.get_column(var.name) for var in self.data.domain.variables})
#
#         if self.target_variable not in df.columns:
#             self.Error.invalid_input(f"Target variable '{self.target_variable}' not found in the data.")
#             return
#
#         target_series = df[self.target_variable].copy()  # Create a copy to prevent modifying the original
#         new_features = {}
#
#         for feature in self.features:
#             feature_type = feature['type']
#             value = feature['value']
#
#             try:
#                 if feature_type == "Lag":
#                     lag = int(value)
#                     new_features[f'{self.target_variable}_lag_{lag}'] = target_series.shift(lag)
#                 elif feature_type == "Rolling Window":
#                     window = int(value)
#                     new_features[f'{self.target_variable}_rolling_mean_{window}'] = target_series.rolling(
#                         window=window).mean()
#                     new_features[f'{self.target_variable}_rolling_std_{window}'] = target_series.rolling(
#                         window=window).std()
#                 elif feature_type == "EWM":
#                     span = float(value)
#                     new_features[f'{self.target_variable}_ewm_{span}'] = target_series.ewm(span=span).mean()
#                 elif feature_type == "Percentage Change":
#                     new_features[f'{self.target_variable}_pct_change'] = target_series.pct_change()
#                 elif feature_type == "K-means":
#                     n_clusters = int(value)
#                     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#                     new_features[f'{self.target_variable}_kmeans_{n_clusters}'] = kmeans.fit_predict(
#                         target_series.values.reshape(-1, 1))
#             except ValueError as e:
#                 self.Error.invalid_input(f"Invalid value for {feature_type}: {value}. Error: {str(e)}")
#                 return
#             except Exception as e:
#                 self.Error.invalid_input(f"Error processing {feature_type}: {str(e)}")
#                 return
#
#         # Add new features to the dataframe
#         for name, feature in new_features.items():
#             df[name] = feature
#
#         # Remove rows with missing values
#         df_clean = df.dropna()
#
#         # Create new domain with original variables and new features
#         new_domain = Domain(
#             [var for var in self.data.domain.attributes] +
#             [ContinuousVariable(name) for name in new_features.keys()],
#             self.data.domain.class_vars,
#             self.data.domain.metas
#         )
#
#         try:
#             # Create new Table with original and new features, using only non-missing rows
#             new_X = df_clean.values
#             new_table = Table.from_numpy(new_domain, new_X)
#             self.Outputs.engineered_series.send(new_table)
#         except Exception as e:
#             self.Error.invalid_input(f"Error creating output table: {str(e)}")
#             return
#
#         self.info_label.setText(
#             f"Successfully created {len(new_features)} new features. Output has {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows with missing values).")
#
# if __name__ == "__main__":
#     WidgetPreview(OWTimeSeriesFeatureEngineering).run()