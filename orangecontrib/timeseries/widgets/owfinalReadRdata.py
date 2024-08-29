import os
from Orange.widgets import widget, gui, settings
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from AnyQt.QtWidgets import QFileDialog, QGridLayout, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, \
    QSplitter
from AnyQt.QtCore import Qt
import pyreadr
import pandas as pd
import numpy as np



class RDataReader(widget.OWWidget):
    name = "RData Reader"
    description = "Read data from .Rdata files"
    icon = "icons/ow_readRdata.svg"
    priority = 10

    class Outputs:
        data = widget.Output("Data", Table)

    want_main_area = True
    resizing_enabled = True

    filename = settings.Setting("")
    displayed_filename = settings.Setting("")
    column_roles = settings.Setting({})
    source_type = settings.Setting(0)
    url = settings.Setting("")

    def __init__(self):
        super().__init__()

        self.data = None
        self.df = None
        self.domain_vars = []

        # Control area
        control_area = gui.widgetBox(self.controlArea, "")
        control_area.setMinimumWidth(300)

        # Source selection
        source_box = gui.widgetBox(control_area, "Source")
        gui.comboBox(source_box, self, "source_type", items=["File", "URL"], callback=self.source_changed)

        file_box = gui.vBox(source_box)
        self.file_edit = gui.lineEdit(file_box, self, "displayed_filename",
                                      placeholderText="Select a .Rdata file...")
        self.file_edit.setReadOnly(True)
        self.file_button = gui.button(file_box, self, "Choose File", callback=self.browse_file)

        self.url_edit = gui.lineEdit(source_box, self, "url", callback=self.load_data,
                                     placeholderText="Enter URL...")

        # Info box
        self.info_box = gui.widgetBox(control_area, "Info")
        self.info_label = gui.widgetLabel(self.info_box, "No file selected")

        self.apply_button = gui.button(control_area, self, "Apply", callback=self.apply_changes)
        self.apply_button.setEnabled(False)

        # Main area
        main_box = gui.widgetBox(self.mainArea, "Columns")
        self.col_table = QTableWidget(main_box)
        self.col_table.setColumnCount(4)
        self.col_table.setHorizontalHeaderLabels(["Name", "Type", "Role", "Values"])
        self.col_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_box.layout().addWidget(self.col_table)

        self.source_changed()

    def source_changed(self):
        self.file_edit.setEnabled(self.source_type == 0)
        self.file_button.setEnabled(self.source_type == 0)
        self.url_edit.setEnabled(self.source_type == 1)

    def browse_file(self):
        start_dir = os.path.dirname(self.filename) if self.filename else os.path.expanduser("~")
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open RData File', start_dir, 'RData files (*.RData *.rdata)'
        )
        if filename:
            self.filename = filename
            self.displayed_filename = os.path.splitext(os.path.basename(filename))[0]
            self.load_data()

    def load_data(self):
        if self.source_type == 0 and not self.filename:
            return
        if self.source_type == 1 and not self.url:
            return

        try:
            # Clear previous data and column roles
            self.df = None
            self.column_roles.clear()

            if self.source_type == 0:
                result = pyreadr.read_r(self.filename)
            else:
                # You may need to implement URL loading for RData files
                raise NotImplementedError("URL loading not implemented for RData files")

            self.df = result[list(result.keys())[0]]  # Get the first DataFrame
            self.setup_column_table()
            self.update_info()
            self.apply_button.setEnabled(True)
        except Exception as e:
            self.info_label.setText(f"Error loading file:\n{str(e)}")
            self.df = None
            self.apply_button.setEnabled(False)

        # Clear the column table if there's an error
        if self.df is None:
            self.col_table.setRowCount(0)

    def update_info(self):
        if self.df is not None:
            n_instances = len(self.df)
            n_features = len(self.df.columns)
            n_meta = sum(1 for role in self.column_roles.values() if role == "Meta")
            n_target = sum(1 for role in self.column_roles.values() if role == "Target")

            info_text = f"{n_instances} instances\n"
            info_text += f"{n_features} features\n"
            info_text += "No missing values\n" if self.df.isnull().sum().sum() == 0 else "Contains missing values\n"
            info_text += f"{'No target' if n_target == 0 else f'{n_target} target'} variable{'s' if n_target > 1 else ''}\n"
            info_text += f"{n_meta} meta attribute{'s' if n_meta != 1 else ''}"

            self.info_label.setText(info_text)
        else:
            self.info_label.setText("No file selected")

        # Adjust the size of the info box to fit the content
        self.info_box.adjustSize()

    def setup_column_table(self):
        self.col_table.setRowCount(len(self.df.columns))
        for i, col in enumerate(self.df.columns):
            self.col_table.setItem(i, 0, QTableWidgetItem(col))
            self.col_table.setItem(i, 1, QTableWidgetItem(str(self.df[col].dtype)))

            role_combo = QComboBox()
            role_combo.addItems(["Feature", "Target", "Meta", "Skip"])
            role_combo.setCurrentText(self.column_roles.get(col, "Feature"))
            role_combo.currentTextChanged.connect(self.make_update_column_role(col))
            self.col_table.setCellWidget(i, 2, role_combo)

            values = str(self.df[col].unique()[:5]).strip('[]')
            if len(self.df[col].unique()) > 5:
                values += ", ..."
            self.col_table.setItem(i, 3, QTableWidgetItem(values))

    def make_update_column_role(self, column):
        def update(role):
            self.column_roles[column] = role
            self.update_info()

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
# from AnyQt.QtWidgets import QFileDialog, QGridLayout, QLabel, QComboBox
# from AnyQt.QtCore import Qt
# import pyreadr
# import pandas as pd
# import numpy as np
#
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
#     want_main_area = True
#
#     filename = settings.Setting("")
#     column_roles = settings.Setting({})
#
#     def __init__(self):
#         super().__init__()
#
#         self.data = None
#         self.df = None
#         self.domain_vars = []
#
#         # GUI
#         self.controlArea.setMinimumWidth(310)
#         box = gui.widgetBox(self.controlArea, "RData File")
#
#         self.file_button = gui.button(
#             box, self, "Select File", callback=self.browse_file, autoDefault=False
#         )
#         self.file_button.setIcon(self.style().standardIcon(self.style().SP_DirOpenIcon))
#
#         self.file_edit = gui.lineEdit(
#             box, self, "filename", callback=self.load_data,
#             placeholderText="Select a .Rdata file..."
#         )
#
#         self.info_box = gui.widgetBox(self.controlArea, "Info")
#         self.info_label = gui.widgetLabel(self.info_box, "No file selected")
#
#         self.apply_button = gui.button(
#             self.controlArea, self, "Apply", callback=self.apply_changes
#         )
#         self.apply_button.setEnabled(False)
#
#         # Main area
#         self.main_box = gui.widgetBox(self.mainArea, "Column Selection")
#         self.col_layout = QGridLayout()
#         self.main_box.layout().addLayout(self.col_layout)
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
#             self.df = result[list(result.keys())[0]]  # Get the first DataFrame
#             self.setup_column_selection()
#             self.info_label.setText(f"Loaded: {self.filename}\nShape: {self.df.shape}")
#             self.apply_button.setEnabled(True)
#         except Exception as e:
#             self.info_label.setText(f"Error loading file:\n{str(e)}")
#             self.df = None
#             self.apply_button.setEnabled(False)
#
#     def setup_column_selection(self):
#         # Clear existing layout
#         for i in reversed(range(self.col_layout.count())):
#             self.col_layout.itemAt(i).widget().setParent(None)
#
#         # Set up headers
#         headers = ["Column Name", "Data Type", "Role"]
#         for i, header in enumerate(headers):
#             self.col_layout.addWidget(QLabel(header), 0, i)
#
#         # Set up column info and role selection
#         for i, col in enumerate(self.df.columns):
#             self.col_layout.addWidget(QLabel(col), i + 1, 0)
#             self.col_layout.addWidget(QLabel(str(self.df[col].dtype)), i + 1, 1)
#
#             role_combo = QComboBox()
#             role_combo.addItems(["Feature", "Target", "Meta", "Skip"])
#             role_combo.setCurrentText(self.column_roles.get(col, "Feature"))
#             role_combo.currentTextChanged.connect(self.make_update_column_role(col))
#             self.col_layout.addWidget(role_combo, i + 1, 2)
#
#     def make_update_column_role(self, column):
#         def update(role):
#             self.column_roles[column] = role
#
#         return update
#
#     def apply_changes(self):
#         if self.df is None:
#             return
#
#         features = []
#         class_vars = []
#         metas = []
#
#         for col in self.df.columns:
#             role = self.column_roles.get(col, "Feature")
#             var = self._create_variable(col)
#             if role == "Feature":
#                 features.append(var)
#             elif role == "Target":
#                 class_vars.append(var)
#             elif role == "Meta":
#                 metas.append(var)
#
#         domain = Domain(features, class_vars, metas)
#
#         # Create Table
#         X = []
#         Y = []
#         M = []
#
#         for col in features:
#             X.append(self._get_column_data(col))
#         for col in class_vars:
#             Y.append(self._get_column_data(col))
#         for col in metas:
#             M.append(self._get_column_data(col))
#
#         X = np.column_stack(X) if X else np.empty((len(self.df), 0))
#         Y = np.column_stack(Y) if Y else None
#         M = np.column_stack(M) if M else None
#
#         table = Table.from_numpy(domain, X, Y, M)
#         self.Outputs.data.send(table)
#
#     def _create_variable(self, col):
#         if pd.api.types.is_numeric_dtype(self.df[col]):
#             return ContinuousVariable(col)
#         elif pd.api.types.is_categorical_dtype(self.df[col]):
#             return DiscreteVariable(col, values=list(self.df[col].cat.categories))
#         else:
#             return StringVariable(col)
#
#     def _get_column_data(self, var):
#         if isinstance(var, DiscreteVariable):
#             return self.df[var.name].cat.codes.values
#         else:
#             return self.df[var.name].values
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#     WidgetPreview(RDataReader).run()