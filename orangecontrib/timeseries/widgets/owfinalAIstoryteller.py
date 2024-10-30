from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import OWWidget, Input, Output
from AnyQt.QtWidgets import QTextEdit, QComboBox, QVBoxLayout
from Orange.data import Table, TimeVariable

import google.generativeai as genai
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_breuschpagan
from pmdarima import auto_arima
from scipy.stats import normaltest
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class OWTimeSeriesStoryteller(OWWidget):
    name = "Time Series Storyteller"
    description = "Analyzes time series data and provides comprehensive insights using Gemini AI"
    icon = "icons/ow_aistoryteller.svg"
    priority = 10

    class Inputs:
        data = Input("Time Series Data", Table)

    class Outputs:
        narrative = Output("Narrative", str)

    want_main_area = True

    api_key = settings.Setting("")
    system_prompt = settings.Setting("""
    You are an AI assistant specialized in analyzing time series data. Provide a detailed narrative 
    description of the time series, including:
    1. Overview of the data (length, frequency, time range)
    2. Descriptive statistics (mean, median, std dev, min, max, quartiles)
    3. Trends and patterns (linear, non-linear, cyclical)
    4. Seasonality analysis (presence, strength, period)
    5. Stationarity assessment (ADF and KPSS tests)
    6. Autocorrelation and partial autocorrelation analysis
    7. Outlier detection and potential impact
    8. Distribution analysis (normality, skewness, kurtosis)
    9. Heteroscedasticity check (if applicable)
    10. Change point detection (if applicable)
    11. Correlation analysis with other variables (if applicable)
    12. Forecasting model suggestions (ARIMA, Exponential Smoothing, etc.)
    13. Potential challenges in modeling this data
    14. Recommendations for further analysis or data collection
    15. Data quality issues (missing values, inconsistencies)
    16. Any unique characteristics or anomalies in the data

    Use clear, concise language and organize your analysis into sections. Provide as much detail 
    as possible, focusing on insights that would be valuable for understanding and modeling the data.
    Suggest potential real-world factors that might explain patterns or anomalies observed in the data.
    If certain analyses couldn't be performed due to data limitations, explain why and what it might
    imply about the nature of the data.
    """)
    target_variable = settings.Setting("")

    def __init__(self):
        super().__init__()
        self.data = None
        self.df = None
        self.time_var = None
        self.target_var = None
        self.other_vars = None

        # Main layout
        layout = QVBoxLayout()
        self.controlArea.setLayout(layout)
        self.mainArea.setLayout(QVBoxLayout())

        # Control area
        control_box = gui.widgetBox(self.controlArea, "Configuration")

        self.api_key_input = gui.lineEdit(control_box, self, "api_key", "API Key:")

        self.target_var_combo = QComboBox()
        self.target_var_combo.currentTextChanged.connect(self.target_variable_changed)
        control_box.layout().addWidget(self.target_var_combo)

        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setPlainText(self.system_prompt)
        self.system_prompt_input.textChanged.connect(self.system_prompt_changed)
        control_box.layout().addWidget(self.system_prompt_input)

        self.analyze_button = gui.button(control_box, self, "Analyze Time Series", callback=self.analyze_data)

        # Main area
        self.narrative_output = QTextEdit(self.mainArea)
        self.narrative_output.setReadOnly(True)
        self.mainArea.layout().addWidget(self.narrative_output)

    def system_prompt_changed(self):
        self.system_prompt = self.system_prompt_input.toPlainText()

    def target_variable_changed(self, value):
        self.target_variable = value
        self.target_var = value

    def analyze_data(self):
        if not self.data or not self.api_key or not self.target_variable:
            self.narrative_output.setText(
                "Please ensure data is loaded, API key is entered, and target variable is selected.")
            return

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            data_description = self.prepare_data_description()
            prompt = f"{self.system_prompt}\n\nAnalyze the following time series data:\n\n{data_description}"
            response = model.generate_content(prompt)
            self.narrative_output.setText(response.text)
            self.Outputs.narrative.send(response.text)

        except Exception as e:
            self.narrative_output.setText(f"Error analyzing data: {str(e)}")

    def prepare_data_description(self):
        description = f"Time series with {len(self.df)} data points.\n"
        description += f"Target variable: {self.target_var}\n"
        if self.time_var:
            description += f"Time variable: {self.time_var}\n"
            description += f"Time range: from {self.df[self.time_var].min()} to {self.df[self.time_var].max()}\n"
        if self.other_vars:
            description += f"Other variables: {', '.join(self.other_vars)}\n\n"
        else:
            description += "No additional variables.\n\n"

        description += f"Descriptive statistics for {self.target_var}:\n"
        description += self.df[self.target_var].describe().to_string() + "\n\n"

        # Stationarity tests
        try:
            adf_result = adfuller(self.df[self.target_var].dropna())
            description += f"ADF test p-value: {adf_result[1]:.4f}\n"
        except:
            description += "Unable to perform ADF test. This might be due to constant or insufficient data.\n"

        try:
            kpss_result = kpss(self.df[self.target_var].dropna())
            description += f"KPSS test p-value: {kpss_result[1]:.4f}\n\n"
        except:
            description += "Unable to perform KPSS test. This might be due to constant or insufficient data.\n\n"

        # Trend analysis
        try:
            trend, _ = stats.linregress(range(len(self.df)), self.df[self.target_var])[:2]
            description += f"Trend analysis for {self.target_var}:\n"
            description += f"{'Upward' if trend > 0 else 'Downward'} trend (slope: {trend:.4f})\n\n"
        except:
            description += "Unable to perform trend analysis. This might be due to non-numeric data or other issues.\n\n"

        # Outlier detection
        try:
            Q1, Q3 = self.df[self.target_var].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = self.df[
                (self.df[self.target_var] < (Q1 - 1.5 * IQR)) | (self.df[self.target_var] > (Q3 + 1.5 * IQR))]
            description += f"Outlier detection for {self.target_var}:\n"
            description += f"{len(outliers)} potential outliers detected\n\n"
        except:
            description += "Unable to perform outlier detection. This might be due to non-numeric data or other issues.\n\n"

        # Autocorrelation and Partial Autocorrelation
        try:
            acf_values = acf(self.df[self.target_var].dropna(), nlags=10)
            pacf_values = pacf(self.df[self.target_var].dropna(), nlags=10)
            description += f"Autocorrelation (first 10 lags): {acf_values}\n"
            description += f"Partial Autocorrelation (first 10 lags): {pacf_values}\n\n"
        except:
            description += "Unable to compute ACF and PACF. This might be due to non-stationary data or other issues.\n\n"

        # Distribution analysis
        try:
            normality_test = normaltest(self.df[self.target_var].dropna())
            description += f"Normality test (D'Agostino and Pearson's test) p-value: {normality_test.pvalue:.4f}\n"
            description += f"Skewness: {self.df[self.target_var].skew():.4f}\n"
            description += f"Kurtosis: {self.df[self.target_var].kurtosis():.4f}\n\n"
        except:
            description += "Unable to perform distribution analysis. This might be due to non-numeric data or other issues.\n\n"

        # Heteroscedasticity test
        try:
            _, p_value, _, _ = het_breuschpagan(self.df[self.target_var].dropna(),
                                                np.arange(len(self.df[self.target_var].dropna())).reshape(-1, 1))
            description += f"Heteroscedasticity test (Breusch-Pagan) p-value: {p_value:.4f}\n\n"
        except:
            description += "Unable to perform heteroscedasticity test. This might be due to constant data or other issues.\n\n"

        # Seasonality analysis
        try:
            result = seasonal_decompose(self.df[self.target_var].dropna(), model='additive', period=12)
            description += "Seasonality detected. Seasonal decomposition performed.\n"
            description += f"Seasonal strength: {1 - np.var(result.resid) / np.var(result.seasonal + result.resid):.4f}\n\n"
        except:
            description += "Unable to perform seasonal decomposition. Data might not have clear seasonality or sufficient length.\n\n"

        # Correlation analysis
        if self.other_vars:
            try:
                description += "Correlation with other variables:\n"
                correlations = self.df[[self.target_var] + self.other_vars].corr()[self.target_var].drop(
                    self.target_var)
                description += correlations.to_string() + "\n\n"
            except:
                description += "Unable to compute correlations. This might be due to non-numeric data or other issues.\n\n"

        # ARIMA model suggestion
        try:
            model = auto_arima(self.df[self.target_var].dropna(), seasonal=True, suppress_warnings=True)
            description += f"Suggested ARIMA model: ARIMA{model.order}{model.seasonal_order}\n\n"
        except:
            description += "Unable to suggest ARIMA model. Data might have issues or be unsuitable for ARIMA modeling.\n\n"

        # Exponential Smoothing
        try:
            model = ExponentialSmoothing(self.df[self.target_var].dropna()).fit()
            description += f"Exponential Smoothing alpha parameter: {model.params['smoothing_level']:.4f}\n\n"
        except:
            description += "Unable to fit Exponential Smoothing model. This might be due to non-numeric data or other issues.\n\n"

        # Change point detection
        try:
            from ruptures import Pelt
            change_points = Pelt(model="rbf").fit(self.df[self.target_var].values).predict(pen=10)
            description += f"Potential change points detected at indices: {change_points}\n\n"
        except ImportError:
            description += "Change point detection not available (ruptures library not installed).\n\n"
        except:
            description += "Unable to perform change point detection. This might be due to non-numeric data or other issues.\n\n"

        # Data quality issues
        missing_values = self.df[self.target_var].isnull().sum()
        description += f"Missing values in target variable: {missing_values}\n"
        description += f"Percentage of missing data: {(missing_values / len(self.df)) * 100:.2f}%\n\n"

        return description

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.df = pd.DataFrame({var.name: data.get_column(var) for var in data.domain.variables})
            self.time_var = next((var.name for var in data.domain.variables if isinstance(var, TimeVariable)), None)
            variables = [var.name for var in data.domain.variables if not isinstance(var, TimeVariable)]
            self.target_var_combo.clear()
            self.target_var_combo.addItems(variables)
            if variables:
                self.target_variable = variables[0]
                self.target_var = variables[0]
            self.other_vars = [var for var in variables if var != self.target_var]
            self.narrative_output.setText(
                "Time series data received. Select a target variable and click 'Analyze Time Series' to generate narrative.")
        else:
            self.df = None
            self.time_var = None
            self.target_var_combo.clear()
            self.target_variable = ""
            self.target_var = None
            self.other_vars = None
            self.narrative_output.clear()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWTimeSeriesStoryteller).run()




# from Orange.widgets import gui, widget, settings
# from Orange.widgets.widget import OWWidget, Input, Output
# from AnyQt.QtWidgets import QTextEdit, QComboBox, QVBoxLayout
# from Orange.data import Table, TimeVariable
#
# import google.generativeai as genai
# import pandas as pd
# import numpy as np
# from scipy import stats
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from pmdarima import auto_arima
#
#
# class OWTimeSeriesStoryteller(OWWidget):
#     name = "Time Series Storyteller"
#     description = "Analyzes time series data and provides comprehensive insights using Gemini AI"
#     icon = "icons/timeseries_storyteller.svg"
#     priority = 10
#
#     class Inputs:
#         data = Input("Time Series Data", Table)
#
#     class Outputs:
#         narrative = Output("Narrative", str)
#
#     want_main_area = True
#
#     api_key = settings.Setting("")
#     system_prompt = settings.Setting("""
#     You are an AI assistant specialized in analyzing time series data. Provide a detailed narrative
#     description of the time series, including:
#     1. Overview of the data
#     2. Trends and patterns
#     3. Seasonality analysis
#     4. Stationarity assessment
#     5. Outlier detection
#     6. Correlation analysis (if applicable)
#     7. Suggestions for modeling approaches
#     8. Potential challenges in modeling this data
#     9. Recommendations for further analysis or data collection
#
#     Use clear, concise language and organize your analysis into sections. Provide as much detail
#     as possible, focusing on insights that would be valuable for understanding and modeling the data.
#     """)
#     target_variable = settings.Setting("")
#
#     def __init__(self):
#         super().__init__()
#         self.data = None
#         self.df = None
#         self.time_var = None
#         self.target_var = None
#         self.other_vars = None
#
#         # Main layout
#         layout = QVBoxLayout()
#         self.controlArea.setLayout(layout)
#         self.mainArea.setLayout(QVBoxLayout())
#
#         # Control area
#         control_box = gui.widgetBox(self.controlArea, "Configuration")
#
#         self.api_key_input = gui.lineEdit(control_box, self, "api_key", "API Key:")
#
#         self.target_var_combo = QComboBox()
#         self.target_var_combo.currentTextChanged.connect(self.target_variable_changed)
#         control_box.layout().addWidget(self.target_var_combo)
#
#         self.system_prompt_input = QTextEdit()
#         self.system_prompt_input.setPlainText(self.system_prompt)
#         self.system_prompt_input.textChanged.connect(self.system_prompt_changed)
#         control_box.layout().addWidget(self.system_prompt_input)
#
#         self.analyze_button = gui.button(control_box, self, "Analyze Time Series", callback=self.analyze_data)
#
#         # Main area
#         self.narrative_output = QTextEdit(self.mainArea)
#         self.narrative_output.setReadOnly(True)
#         self.mainArea.layout().addWidget(self.narrative_output)
#
#     def system_prompt_changed(self):
#         self.system_prompt = self.system_prompt_input.toPlainText()
#
#     def target_variable_changed(self, value):
#         self.target_variable = value
#         self.target_var = value
#
#     def analyze_data(self):
#         if not self.data or not self.api_key or not self.target_variable:
#             self.narrative_output.setText(
#                 "Please ensure data is loaded, API key is entered, and target variable is selected.")
#             return
#
#         try:
#             genai.configure(api_key=self.api_key)
#             model = genai.GenerativeModel('gemini-pro')
#             data_description = self.prepare_data_description()
#             prompt = f"{self.system_prompt}\n\nAnalyze the following time series data:\n\n{data_description}"
#             response = model.generate_content(prompt)
#             self.narrative_output.setText(response.text)
#             self.Outputs.narrative.send(response.text)
#
#         except Exception as e:
#             self.narrative_output.setText(f"Error analyzing data: {str(e)}")
#
#     def prepare_data_description(self):
#         description = f"Time series with {len(self.df)} data points.\n"
#         description += f"Target variable: {self.target_var}\n"
#         if self.time_var:
#             description += f"Time variable: {self.time_var}\n"
#         if self.other_vars:
#             description += f"Other variables: {', '.join(self.other_vars)}\n\n"
#         else:
#             description += "No additional variables.\n\n"
#
#         description += f"Basic statistics for {self.target_var}:\n"
#         description += self.df[self.target_var].describe().to_string() + "\n\n"
#
#         result = adfuller(self.df[self.target_var].dropna())
#         description += f"Stationarity test for {self.target_var}: p-value = {result[1]:.4f}\n"
#
#         trend, _ = stats.linregress(range(len(self.df)), self.df[self.target_var])[:2]
#         description += f"\nTrend analysis for {self.target_var}:\n"
#         description += f"{'Upward' if trend > 0 else 'Downward'} trend (slope: {trend:.4f})\n"
#
#         Q1, Q3 = self.df[self.target_var].quantile([0.25, 0.75])
#         IQR = Q3 - Q1
#         outliers = self.df[
#             (self.df[self.target_var] < (Q1 - 1.5 * IQR)) | (self.df[self.target_var] > (Q3 + 1.5 * IQR))]
#         description += f"\nOutlier detection for {self.target_var}:\n"
#         description += f"{len(outliers)} potential outliers detected\n"
#
#         if self.other_vars:
#             description += "\nCorrelation with other variables:\n"
#             correlations = self.df[[self.target_var] + self.other_vars].corr()[self.target_var].drop(self.target_var)
#             description += correlations.to_string() + "\n"
#
#         # Add seasonality information
#         try:
#             result = seasonal_decompose(self.df[self.target_var].dropna(), model='additive', period=12)
#             description += "\nSeasonality detected. Seasonal decomposition performed.\n"
#         except:
#             description += "\nUnable to perform seasonal decomposition. Data might not have clear seasonality or sufficient length.\n"
#
#         # Add ARIMA model suggestion
#         try:
#             model = auto_arima(self.df[self.target_var].dropna(), seasonal=True, suppress_warnings=True)
#             description += f"\nSuggested ARIMA model: ARIMA{model.order}{model.seasonal_order}\n"
#         except:
#             description += "\nUnable to suggest ARIMA model. Data might have issues or be unsuitable for ARIMA modeling.\n"
#
#         return description
#
#     @Inputs.data
#     def set_data(self, data):
#         self.data = data
#         if data is not None:
#             self.df = pd.DataFrame({var.name: data.get_column(var) for var in data.domain.variables})
#             self.time_var = next((var.name for var in data.domain.variables if isinstance(var, TimeVariable)), None)
#             variables = [var.name for var in data.domain.variables if not isinstance(var, TimeVariable)]
#             self.target_var_combo.clear()
#             self.target_var_combo.addItems(variables)
#             if variables:
#                 self.target_variable = variables[0]
#                 self.target_var = variables[0]
#             self.other_vars = [var for var in variables if var != self.target_var]
#             self.narrative_output.setText(
#                 "Time series data received. Select a target variable and click 'Analyze Time Series' to generate narrative.")
#         else:
#             self.df = None
#             self.time_var = None
#             self.target_var_combo.clear()
#             self.target_variable = ""
#             self.target_var = None
#             self.other_vars = None
#             self.narrative_output.clear()
#
#
# if __name__ == "__main__":
#     from Orange.widgets.utils.widgetpreview import WidgetPreview
#
#     WidgetPreview(OWTimeSeriesStoryteller).run()