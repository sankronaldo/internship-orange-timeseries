import numpy as np
import pandas as pd
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.widgets.widget import Input, Output
from Orange.widgets.visualize.utils.plotutils import PlotWidget
import pyqtgraph as pg
from PyQt5.QtWidgets import QTabWidget
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, wasserstein_distance, probplot
from ctgan import CTGAN
import torch

class OWCTGAN(widget.OWWidget):
    name = "CTGAN Data Augmentation"
    description = "Augment time series data using CTGAN"
    icon = "icons/final.svg"
    priority = 10

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        augmented_data = Output("Augmented Data", Table)

    want_main_area = True

    # Settings
    target_variable = Setting("")
    epochs = Setting(100)
    batch_size = Setting(500)
    generator_dim_1 = Setting(128)
    generator_dim_2 = Setting(128)
    discriminator_dim_1 = Setting(128)
    discriminator_dim_2 = Setting(128)
    generator_lr = Setting(2e-4)
    discriminator_lr = Setting(2e-4)
    discriminator_steps = Setting(1)
    log_frequency = Setting(True)
    sample_size = Setting(100)
    random_seed = Setting(42)


    def __init__(self):
        super().__init__()

        self.data = None
        self.model = None
        self.synthetic_data = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box, "No data on input.")

        # Target variable selection
        self.target_combo = gui.comboBox(
            box, self, "target_variable", label="Target Variable:",
            orientation="horizontal", callback=self.on_target_variable_changed)

        # CTGAN parameters
        ctgan_box = gui.widgetBox(self.controlArea, "CTGAN Parameters")
        gui.spin(ctgan_box, self, "epochs", 1, 1000, label="Epochs:", callback=self.on_param_changed)
        gui.spin(ctgan_box, self, "batch_size", 1, 1000, label="Batch Size:", callback=self.on_param_changed)
        # gui.lineEdit(ctgan_box, self, "generator_dim", label="Generator Dimensions:", callback=self.on_param_changed)
        # gui.lineEdit(ctgan_box, self, "discriminator_dim", label="Discriminator Dimensions:", callback=self.on_param_changed)


        gui.spin( ctgan_box, self, "random_seed", 0, 1000000, label="Random Seed:",
                  callback=self.on_param_changed)
        gen_dim_box = gui.hBox(ctgan_box)
        gui.spin(gen_dim_box, self, "generator_dim_1", 1, 1024, label="Generator Dim 1:",
                 callback=self.on_param_changed)
        gui.spin(gen_dim_box, self, "generator_dim_2", 1, 1024, label="Generator Dim 2:",
                 callback=self.on_param_changed)

        disc_dim_box = gui.hBox(ctgan_box)
        gui.spin(disc_dim_box, self, "discriminator_dim_1", 1, 1024, label="Discriminator Dim 1:",
                 callback=self.on_param_changed)
        gui.spin(disc_dim_box, self, "discriminator_dim_2", 1, 1024, label="Discriminator Dim 2:",
                 callback=self.on_param_changed)

        gui.doubleSpin(ctgan_box, self, "generator_lr", 1e-5, 1e-1, 1e-5, label="Generator Learning Rate:", callback=self.on_param_changed)
        gui.doubleSpin(ctgan_box, self, "discriminator_lr", 1e-5, 1e-1, 1e-5, label="Discriminator Learning Rate:", callback=self.on_param_changed)
        gui.spin(ctgan_box, self, "discriminator_steps", 1, 10, label="Discriminator Steps:", callback=self.on_param_changed)
        gui.checkBox(ctgan_box, self, "log_frequency", label="Log Frequency", callback=self.on_param_changed)

        # Sample size
        sample_box = gui.widgetBox(self.controlArea, "Sample Settings")
        gui.spin(sample_box, self, "sample_size", 1, 1000, label="Sample Size:", callback=self.on_param_changed)

        # Generate button
        self.generate_button = gui.button(self.controlArea, self, "Generate Synthetic Data", callback=self.generate_synthetic_data)

        # Set up the main area with tab widget
        self.tab_widget = QTabWidget()
        self.mainArea.layout().addWidget(self.tab_widget)

        # Time Series Plot
        self.ts_plot_widget = PlotWidget(background="w")
        self.tab_widget.addTab(self.ts_plot_widget, "Time Series")

        # Distribution Plot
        self.dist_plot_widget = PlotWidget(background="w")
        self.tab_widget.addTab(self.dist_plot_widget, "Distribution")

        # Q-Q Plot
        self.qq_plot_widget = PlotWidget(background="w")
        self.tab_widget.addTab(self.qq_plot_widget, "Q-Q Plot")

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.info_label.setText(f"{len(data)} instances on input.")

            # Update target variable combo box options
            self.target_combo.clear()
            self.target_combo.addItem("")
            for var in data.domain.variables:
                if var.is_continuous and not isinstance(var, TimeVariable):
                    self.target_combo.addItem(var.name)

            # Set initial target variable if previously selected
            if self.target_variable in data.domain:
                self.target_combo.setCurrentIndex(self.target_combo.findText(self.target_variable))

            self.on_target_variable_changed()
        else:
            self.data = None
            self.info_label.setText("No data on input.")
            self.clear_plots()

    def on_target_variable_changed(self):
        self.target_variable = self.target_combo.currentText()

    def on_param_changed(self):
        self.generator_dim = (self.generator_dim_1, self.generator_dim_2)
        self.discriminator_dim = (self.discriminator_dim_1, self.discriminator_dim_2)

    def generate_synthetic_data(self):
        if self.data is None or not self.target_variable:
            return

        # Set the random seed for reproducibility
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        target_data = self.data.get_column(self.target_variable).reshape(-1, 1)
        train_data, _ = train_test_split(target_data, test_size=0.2, random_state=self.random_seed)

        # Initialize and fit the CTGAN model
        self.model = CTGAN(
            epochs=self.epochs,
            batch_size=self.batch_size,
            generator_dim=(self.generator_dim_1, self.generator_dim_2),
            discriminator_dim=(self.discriminator_dim_1, self.discriminator_dim_2),
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=True
        )

        # Set random seeds for PyTorch operations inside CTGAN
        self.model.set_random_state(self.random_seed)

        self.model.fit(train_data)

        # Generate synthetic data
        self.synthetic_data = self.model.sample(self.sample_size)

        self.update_plots()
        self.update_model_info()
        self.output_results()

    def update_plots(self):
        self.clear_plots()
        if self.synthetic_data is None:
            return

        original_data = self.data.get_column(self.target_variable)
        synthetic_data = self.synthetic_data.flatten()

        # Time Series Plot
        self.plot_time_series(original_data, synthetic_data)

        # Distribution Plot
        self.plot_distribution(original_data, synthetic_data)

        # Q-Q Plot
        self.plot_qq(original_data, synthetic_data)

    def plot_time_series(self, original_data, synthetic_data):
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.ts_plot_widget.graphicsItem())

        original_plot = self.ts_plot_widget.plot(np.arange(len(original_data)), original_data,
                                                 pen=pg.mkPen(color=(0, 0, 255), width=2), name='Original')
        legend.addItem(original_plot, 'Original')

        synthetic_plot = self.ts_plot_widget.plot(np.arange(len(original_data), len(original_data) + len(synthetic_data)), synthetic_data,
                                                  pen=pg.mkPen(color=(255, 0, 0), width=2), name='Synthetic')
        legend.addItem(synthetic_plot, 'Synthetic')

        self.ts_plot_widget.setLabel('left', self.target_variable)
        self.ts_plot_widget.setLabel('bottom', 'Time')
        self.ts_plot_widget.setTitle('Original vs Synthetic Data')

    def plot_distribution(self, original_data, synthetic_data):
        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.dist_plot_widget.graphicsItem())

        y, x = np.histogram(original_data, bins=50)
        original_plot = self.dist_plot_widget.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True,
                                                   brush=(0, 0, 255, 150), name='Original')
        legend.addItem(original_plot, 'Original')

        y, x = np.histogram(synthetic_data, bins=50)
        synthetic_plot = self.dist_plot_widget.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True,
                                                    brush=(255, 0, 0, 150), name='Synthetic')
        legend.addItem(synthetic_plot, 'Synthetic')

        self.dist_plot_widget.setLabel('left', 'Frequency')
        self.dist_plot_widget.setLabel('bottom', self.target_variable)
        self.dist_plot_widget.setTitle('Distribution of Original vs Synthetic Data')

    def plot_qq(self, original_data, synthetic_data):
        original_quantiles, _ = probplot(original_data, dist="norm")
        synthetic_quantiles, _ = probplot(synthetic_data, dist="norm")

        original_plot = self.qq_plot_widget.plot(original_quantiles[0], original_quantiles[1],
                                                 pen=None, symbol='o', symbolSize=5, symbolBrush=(0, 0, 255, 150), name='Original')
        synthetic_plot = self.qq_plot_widget.plot(synthetic_quantiles[0], synthetic_quantiles[1],
                                                  pen=None, symbol='o', symbolSize=5, symbolBrush=(255, 0, 0, 150), name='Synthetic')

        legend = pg.LegendItem(offset=(50, 30))
        legend.setParentItem(self.qq_plot_widget.graphicsItem())
        legend.addItem(original_plot, 'Original')
        legend.addItem(synthetic_plot, 'Synthetic')

        self.qq_plot_widget.setLabel('left', 'Ordered Values')
        self.qq_plot_widget.setLabel('bottom', 'Theoretical Quantiles')
        self.qq_plot_widget.setTitle('Q-Q Plot of Original vs Synthetic Data')

    def update_model_info(self):
        if self.model is None:
            return

        original_data = self.data.get_column(self.target_variable)
        synthetic_data = self.synthetic_data.flatten()
        ks_test = ks_2samp(original_data, synthetic_data)
        wd = wasserstein_distance(original_data, synthetic_data)

        model_info = (
            f"CTGAN Model:\n"
            f"  Random Seed: {self.random_seed}\n"  # Add this line
            f"  Epochs: {self.epochs}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Generator dimensions: ({self.generator_dim_1}, {self.generator_dim_2})\n"
            f"  Discriminator dimensions: ({self.discriminator_dim_1}, {self.discriminator_dim_2})\n"
            f"  Generator learning rate: {self.generator_lr}\n"
            f"  Discriminator learning rate: {self.discriminator_lr}\n"
            f"  Discriminator steps: {self.discriminator_steps}\n"
            f"  Log frequency: {self.log_frequency}\n\n"
            f"Performance Metrics:\n"
            f"  Kolmogorov-Smirnov Test:\n"
            f"    Statistic: {ks_test.statistic:.4f}\n"
            f"    p-value: {ks_test.pvalue:.4f}\n"
            f"  Wasserstein Distance: {wd:.4f}"
        )

        self.info_label.setText(model_info)

    def output_results(self):
        if self.synthetic_data is None:
            self.Outputs.augmented_data.send(None)
            return

        # Create a domain with only the target variable
        domain = Domain([ContinuousVariable(self.target_variable)])

        # Create the Table for synthetic data
        synthetic_table = Table.from_numpy(domain, self.synthetic_data)

        # Combine original and synthetic data
        augmented_data = Table.concatenate([self.data.transform(domain), synthetic_table])

        self.Outputs.augmented_data.send(augmented_data)

    def clear_plots(self):
        self.ts_plot_widget.clear()
        self.dist_plot_widget.clear()
        self.qq_plot_widget.clear()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWCTGAN).run()