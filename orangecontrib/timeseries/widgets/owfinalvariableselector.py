import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.signals import Input, Output
from Orange.data import Table, Domain
from PyQt5.QtWidgets import QVBoxLayout, QListWidget, QAbstractItemView
import numpy as np


class TimeSeriesSelector(widget.OWWidget):
    name = "Time Series Selector"
    description = "Select target and exogenous variables from multiple time series data sources"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        data = Input("Time Series Data", Table, multiple=True)

    class Outputs:
        target = Output("Target Variable", Table)
        exogenous = Output("Regression Variables", Table)

    want_main_area = False

    target_attr = settings.Setting("")
    exog_attrs = settings.Setting([])

    def __init__(self):
        super().__init__()

        self.data_dict = {}
        self.variable_labels = []

        # GUI
        box = gui.widgetBox(self.controlArea, "Variable Selection")

        self.target_combo = gui.comboBox(box, self, "target_attr",
                                         label="Target Variable:",
                                         callback=self.on_target_changed)

        self.exog_list = QListWidget()
        self.exog_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.exog_list.itemSelectionChanged.connect(self.on_exog_selection_changed)
        box.layout().addWidget(self.exog_list)

        gui.button(self.controlArea, self, "Apply", callback=self.commit)

    @Inputs.data
    def set_data(self, data, id):
        if data is not None:
            self.data_dict[id] = data
        else:
            self.data_dict.pop(id, None)

        self.update_variable_list()

    def update_variable_list(self):
        self.variable_labels = []
        for data in self.data_dict.values():
            self.variable_labels.extend([f"{data.name}:{var.name}" for var in data.domain.variables])

        self.target_combo.clear()
        self.target_combo.addItems(self.variable_labels)

        self.update_exog_list()

    def update_exog_list(self):
        self.exog_list.clear()
        for var in self.variable_labels:
            if var != self.target_attr:
                self.exog_list.addItem(var)

        # Restore previous selections
        for i in range(self.exog_list.count()):
            item = self.exog_list.item(i)
            if item.text() in self.exog_attrs:
                item.setSelected(True)

    def on_target_changed(self):
        self.target_attr = self.target_combo.currentText()
        self.update_exog_list()

    def on_exog_selection_changed(self):
        self.exog_attrs = [item.text() for item in self.exog_list.selectedItems()]

    def commit(self):
        target = None
        exogenous = None

        if self.data_dict and self.target_attr and isinstance(self.target_attr, str):
            try:
                data_name, var_name = self.target_attr.split(':')
                target_data = next((data for data in self.data_dict.values() if data.name == data_name), None)

                if target_data is not None:
                    domain = target_data.domain
                    if var_name in domain:
                        target_var = domain[var_name]
                        target_domain = Domain([target_var])
                        target = target_data.transform(target_domain)

                exog_vars = []
                for attr in self.exog_attrs:
                    data_name, var_name = attr.split(':')
                    exog_data = next((data for data in self.data_dict.values() if data.name == data_name), None)
                    if exog_data is not None and var_name in exog_data.domain:
                        exog_vars.append((exog_data, exog_data.domain[var_name]))

                if exog_vars:
                    exog_domain = Domain([var for _, var in exog_vars])
                    exog_data = np.column_stack([data[:, var.name] for data, var in exog_vars])
                    exogenous = Table.from_numpy(exog_domain, exog_data)

            except AttributeError:
                print(f"Error: self.target_attr is not a string. Current value: {self.target_attr}")
                # You might want to set a default value or show an error message to the user here

        self.Outputs.target.send(target)
        self.Outputs.exogenous.send(exogenous)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(TimeSeriesSelector).run()







# import Orange
# from Orange.widgets import widget, gui, settings
# from Orange.widgets.utils.signals import Input, Output
# from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
# from PyQt5.QtWidgets import QListWidget
#
#
# class TimeSeriesSelector(widget.OWWidget):
#     name = "Time Series Selector"
#     description = "Select target and exogenous variables from time series data"
#     icon = "icons/timeseries.svg"
#     priority = 10
#
#     class Inputs:
#         data = Input("Time Series Data", Table)
#
#     class Outputs:
#         target = Output("Target Variable", Table)
#         exogenous = Output("Exogenous Variables", Table)
#
#     want_main_area = False
#
#     target_attr = settings.Setting("")
#     exog_attrs = settings.Setting([])
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#         self.variable_labels = []
#
#         # GUI
#         box = gui.widgetBox(self.controlArea, "Variable Selection")
#         self.target_combo = gui.comboBox(box, self, "target_attr",
#                                          label="Target Variable:",
#                                          callback=self.commit)
#         self.exog_list = gui.listBox(box, self, "exog_attrs",
#                                      labels="variable_labels",
#                                      selectionMode=QListWidget.ExtendedSelection,
#                                      callback=self.commit)
#
#     @Inputs.data
#     def set_data(self, data):
#         self.data = data
#         if data is not None:
#             self.variable_labels = [var.name for var in data.domain.variables]
#             self.target_combo.clear()
#             self.target_combo.addItems(self.variable_labels)
#             self.exog_list.clear()
#             self.exog_list.addItems(self.variable_labels)
#
#             # Reset selections if they're not in the new data
#             if self.target_attr not in self.variable_labels:
#                 self.target_attr = ""
#             self.exog_attrs = [attr for attr in self.exog_attrs if attr in self.variable_labels]
#         else:
#             self.variable_labels = []
#             self.target_attr = ""
#             self.exog_attrs = []
#
#         self.commit()
#
#     def commit(self):
#         target = None
#         exogenous = None
#
#         if self.data is not None and self.target_attr:
#             domain = self.data.domain
#
#             if self.target_attr in domain:
#                 target_var = domain[self.target_attr]
#                 target_domain = Domain([target_var])
#                 target = self.data.transform(target_domain)
#
#             valid_exog_vars = [domain[attr] for attr in self.exog_attrs
#                                if attr in domain and attr != self.target_attr]
#             if valid_exog_vars:
#                 exog_domain = Domain(valid_exog_vars)
#                 exogenous = self.data.transform(exog_domain)
#
#         self.Outputs.target.send(target)
#         self.Outputs.exogenous.send(exogenous)
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#     WidgetPreview(TimeSeriesSelector).run()