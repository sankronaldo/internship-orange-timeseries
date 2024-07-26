import os
from Orange.widgets import widget, gui, settings
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from AnyQt.QtWidgets import QFileDialog, QGridLayout, QLabel, QComboBox
from AnyQt.QtCore import Qt
import pyreadr
import pandas as pd
import numpy as np


class RDataReader(widget.OWWidget):
    name = "RData Reader"
    description = "Read data from .Rdata files"
    icon = "icons/rdata.svg"
    priority = 10

    class Outputs:
        data = widget.Output("Data", Table)

    want_main_area = True

    filename = settings.Setting("")
    column_roles = settings.Setting({})

    def __init__(self):
        super().__init__()

        self.data = None
        self.df = None
        self.domain_vars = []

        # GUI
        self.controlArea.setMinimumWidth(310)
        box = gui.widgetBox(self.controlArea, "RData File")

        self.file_button = gui.button(
            box, self, "Select File", callback=self.browse_file, autoDefault=False
        )
        self.file_button.setIcon(self.style().standardIcon(self.style().SP_DirOpenIcon))

        self.file_edit = gui.lineEdit(
            box, self, "filename", callback=self.load_data,
            placeholderText="Select a .Rdata file..."
        )

        self.info_box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(self.info_box, "No file selected")

        self.apply_button = gui.button(
            self.controlArea, self, "Apply", callback=self.apply_changes
        )
        self.apply_button.setEnabled(False)

        # Main area
        self.main_box = gui.widgetBox(self.mainArea, "Column Selection")
        self.col_layout = QGridLayout()
        self.main_box.layout().addLayout(self.col_layout)

    def browse_file(self):
        start_dir = os.path.dirname(self.filename) if self.filename else os.path.expanduser("~")
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open RData File', start_dir, 'RData files (*.RData *.rdata)'
        )
        if filename:
            self.filename = filename
            self.load_data()

    def load_data(self):
        if not self.filename:
            return

        try:
            result = pyreadr.read_r(self.filename)
            self.df = result[list(result.keys())[0]]  # Get the first DataFrame
            self.setup_column_selection()
            self.info_label.setText(f"Loaded: {self.filename}\nShape: {self.df.shape}")
            self.apply_button.setEnabled(True)
        except Exception as e:
            self.info_label.setText(f"Error loading file:\n{str(e)}")
            self.df = None
            self.apply_button.setEnabled(False)

    def setup_column_selection(self):
        # Clear existing layout
        for i in reversed(range(self.col_layout.count())):
            self.col_layout.itemAt(i).widget().setParent(None)

        # Set up headers
        headers = ["Column Name", "Data Type", "Role"]
        for i, header in enumerate(headers):
            self.col_layout.addWidget(QLabel(header), 0, i)

        # Set up column info and role selection
        for i, col in enumerate(self.df.columns):
            self.col_layout.addWidget(QLabel(col), i + 1, 0)
            self.col_layout.addWidget(QLabel(str(self.df[col].dtype)), i + 1, 1)

            role_combo = QComboBox()
            role_combo.addItems(["Feature", "Target", "Meta", "Skip"])
            role_combo.setCurrentText(self.column_roles.get(col, "Feature"))
            role_combo.currentTextChanged.connect(self.make_update_column_role(col))
            self.col_layout.addWidget(role_combo, i + 1, 2)

    def make_update_column_role(self, column):
        def update(role):
            self.column_roles[column] = role

        return update

    def apply_changes(self):
        if self.df is None:
            return

        features = []
        class_vars = []
        metas = []

        for col in self.df.columns:
            role = self.column_roles.get(col, "Feature")
            var = self._create_variable(col)
            if role == "Feature":
                features.append(var)
            elif role == "Target":
                class_vars.append(var)
            elif role == "Meta":
                metas.append(var)

        domain = Domain(features, class_vars, metas)

        # Create Table
        X = []
        Y = []
        M = []

        for col in features:
            X.append(self._get_column_data(col))
        for col in class_vars:
            Y.append(self._get_column_data(col))
        for col in metas:
            M.append(self._get_column_data(col))

        X = np.column_stack(X) if X else np.empty((len(self.df), 0))
        Y = np.column_stack(Y) if Y else None
        M = np.column_stack(M) if M else None

        table = Table.from_numpy(domain, X, Y, M)
        self.Outputs.data.send(table)

    def _create_variable(self, col):
        if pd.api.types.is_numeric_dtype(self.df[col]):
            return ContinuousVariable(col)
        elif pd.api.types.is_categorical_dtype(self.df[col]):
            return DiscreteVariable(col, values=list(self.df[col].cat.categories))
        else:
            return StringVariable(col)

    def _get_column_data(self, var):
        if isinstance(var, DiscreteVariable):
            return self.df[var.name].cat.codes.values
        else:
            return self.df[var.name].values


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(RDataReader).run()





# import os
# from Orange.widgets import widget, gui, settings
# from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
# from AnyQt.QtWidgets import QFileDialog
# import pyreadr
# import pandas as pd
# import numpy as np
#
# class RDataReader(widget.OWWidget):
#     name = "RData Reader"
#     description = "Read data from .Rdata files"
#     icon = "icons/rdata.svg"
#     priority = 10
#
#     class Outputs:
#         data = widget.Output("Data", Table)
#
#     want_main_area = False
#
#     filename = settings.Setting("")
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#
#         # GUI
#         self.controlArea.setMinimumWidth(310)
#         box = gui.widgetBox(self.controlArea, "RData File")
#
#         self.file_button = gui.button(
#             box, self, "...", callback=self.browse_file, autoDefault=False
#         )
#         self.file_button.setIcon(self.style().standardIcon(self.style().SP_DirOpenIcon))
#         self.file_button.setSizePolicy(self.file_button.sizePolicy().horizontalPolicy(),
#             self.file_button.sizePolicy().verticalPolicy())
#
#         self.file_edit = gui.lineEdit(
#             box, self, "filename", callback=self.load_data,
#             placeholderText="Select a .Rdata file..."
#         )
#
#         self.info_box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(self.info_box, "No file selected")
#
#     def browse_file(self):
#         start_dir = os.path.dirname(self.filename) if self.filename else os.path.expanduser("~")
#         filename, _ = QFileDialog.getOpenFileName(
#             self, 'Open RData File', start_dir, 'RData files (*.RData *.rdata)'
#         )
#         if filename:
#             self.filename = filename
#             self.load_data()
#
#     def load_data(self):
#         if not self.filename:
#             return
#
#         try:
#             result = pyreadr.read_r(self.filename)
#             df = result[list(result.keys())[0]]  # Get the first DataFrame
#             self.data = self.pandas_to_orange(df)
#             self.info_label.setText(f"Loaded: {self.filename}\nShape: {self.data.X.shape}")
#             self.Outputs.data.send(self.data)
#         except Exception as e:
#             self.info_label.setText(f"Error loading file:\n{str(e)}")
#             self.data = None
#             self.Outputs.data.send(None)
#
#     def pandas_to_orange(self, df):
#         """Convert pandas DataFrame to Orange Table"""
#         def _guess_var_type(s):
#             if pd.api.types.is_numeric_dtype(s):
#                 return ContinuousVariable(s.name)
#             elif pd.api.types.is_categorical_dtype(s):
#                 return DiscreteVariable(s.name, values=list(s.cat.categories))
#             else:
#                 return StringVariable(s.name)
#
#         domain_vars = [_guess_var_type(df[col]) for col in df.columns]
#         domain = Domain(domain_vars)
#
#         # Convert DataFrame to numpy array, handling categorical variables
#         array = np.array([
#             df[col].cat.codes.values if isinstance(domain_vars[i], DiscreteVariable)
#             else df[col].values
#             for i, col in enumerate(df.columns)
#         ]).T
#
#         return Table.from_numpy(domain, array)
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#     WidgetPreview(RDataReader).run()