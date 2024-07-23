import logging
from datetime import datetime, timedelta, date
from Orange.data import Table, Domain, TimeVariable, ContinuousVariable
from AnyQt.QtCore import QDate
from AnyQt.QtWidgets import QDateEdit, QComboBox, QFormLayout, QCheckBox, QVBoxLayout
from Orange.data.pandas_compat import table_from_frame
from orangewidget.utils.widgetpreview import WidgetPreview

from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Output

from orangecontrib.timeseries import Timeseries

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OWWeatherData(widget.OWWidget):
    name = 'Weather Data'
    description = "Generate time series from Open-Meteo weather data."
    icon = 'icons/final.svg'
    priority = 10

    class Outputs:
        time_series = Output("Time series", Timeseries)

    QT_DATE_FORMAT = 'yyyy-MM-dd'
    PY_DATE_FORMAT = '%Y-%m-%d'
    MIN_DATE = date(1940, 1, 1)

    date_from = settings.Setting(
        (datetime.now().date() - timedelta(365)).strftime(PY_DATE_FORMAT))
    date_to = settings.Setting(datetime.now().date().strftime(PY_DATE_FORMAT))

    cities = settings.Setting([
        'London,51.5074,-0.1278',
        'New York,40.7128,-74.0060',
        'Tokyo,35.6762,139.6503',
        'Paris,48.8566,2.3522',
        'Sydney,-33.8688,151.2093',
        'Moscow,55.7558,37.6173',
        'Dubai,25.2048,55.2708',
        'Singapore,1.3521,103.8198',
        'Los Angeles,34.0522,-118.2437',
        'Berlin,52.5200,13.4050',
        'Madrid,40.4168,-3.7038',
        'Rome,41.9028,12.4964',
        'Beijing,39.9042,116.4074',
        'Mumbai,19.0760,72.8777',
        'Rio de Janeiro,-22.9068,-43.1729'
    ])

    use_hourly_data = settings.Setting(False)

    want_main_area = False
    resizing_enabled = False

    class Error(widget.OWWidget.Error):
        download_error = widget.Msg('{}\nNo internet? Invalid city selection?')

    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        gui.widgetBox(self.controlArea, True, orientation=layout)

        self.city_combo = QComboBox()
        self.city_combo.addItems([city.split(',')[0] for city in self.cities])
        layout.addRow("City:", self.city_combo)

        minDate = QDate.fromString(self.MIN_DATE.strftime(self.PY_DATE_FORMAT),
                                   self.QT_DATE_FORMAT)
        date_from, date_to = (
            QDateEdit(QDate.fromString(date, self.QT_DATE_FORMAT),
                      displayFormat=self.QT_DATE_FORMAT, minimumDate=minDate,
                      calendarPopup=True)
            for date in (self.date_from, self.date_to))

        @date_from.dateChanged.connect
        def set_date_from(date):
            self.date_from = date.toString(self.QT_DATE_FORMAT)

        @date_to.dateChanged.connect
        def set_date_to(date):
            self.date_to = date.toString(self.QT_DATE_FORMAT)

        layout.addRow("From:", date_from)
        layout.addRow("To:", date_to)

        self.hourly_checkbox = QCheckBox("Use hourly data")
        self.hourly_checkbox.setChecked(self.use_hourly_data)
        self.hourly_checkbox.stateChanged.connect(self.set_use_hourly_data)
        layout.addRow(self.hourly_checkbox)

        self.button = gui.button(
            self.controlArea, self, 'Download', callback=self.download)

    def set_use_hourly_data(self, state):
        self.use_hourly_data = state == 2  # 2 means checked

    def get_weather_data(self, latitude, longitude, start_date, end_date):
        BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

        if self.use_hourly_data:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": ["temperature_2m", "precipitation", "windspeed_10m"],
                "timezone": "UTC"
            }
        else:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                          "precipitation_sum", "windspeed_10m_max", "windspeed_10m_mean"],
                "timezone": "UTC"
            }

        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if self.use_hourly_data:
                if 'hourly' not in data or not data['hourly']['time']:
                    raise ValueError("No hourly data available in the API response")

                df = pd.DataFrame({
                    'Date and Time': pd.to_datetime(data['hourly']['time']),
                    'Temperature (°C)': data['hourly']['temperature_2m'],
                    'Precipitation (mm)': data['hourly']['precipitation'],
                    'Wind Speed (km/h)': data['hourly']['windspeed_10m']
                })
                df.set_index('Date and Time', inplace=True)
            else:
                if 'daily' not in data or not data['daily']['time']:
                    raise ValueError("No daily data available in the API response")

                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['daily']['time']),
                    'Maximum Temperature (°C)': data['daily']['temperature_2m_max'],
                    'Minimum Temperature (°C)': data['daily']['temperature_2m_min'],
                    'Mean Temperature (°C)': data['daily']['temperature_2m_mean'],
                    'Total Precipitation (mm)': data['daily']['precipitation_sum'],
                    'Maximum Wind Speed (km/h)': data['daily']['windspeed_10m_max'],
                    'Mean Wind Speed (km/h)': data['daily']['windspeed_10m_mean']
                })
                df.set_index('Date', inplace=True)

            # Remove rows with missing values
            df = df.dropna()

            if df.empty:
                raise ValueError("No valid data available for the specified date range after removing missing values")

            # Convert DataFrame to Orange Table
            domain = Domain([ContinuousVariable(name) for name in df.columns],
                            metas=[TimeVariable("Date", have_time=True)])

            table = Table.from_numpy(
                domain,
                df.values,
                metas=df.index.astype(int).to_numpy().reshape(-1, 1) // 10 ** 9
            )

            return table

        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            raise
        except KeyError as e:
            logging.error(f"KeyError: {e}. The API response structure might have changed.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def download(self):
        self.Error.clear()

        try:
            date_from = datetime.strptime(self.date_from, self.PY_DATE_FORMAT)
            date_to = datetime.strptime(self.date_to, self.PY_DATE_FORMAT)

            city_info = self.cities[self.city_combo.currentIndex()]
            city_parts = city_info.split(',')
            if len(city_parts) != 3:
                raise ValueError(f"Invalid city information: {city_info}")

            city_name, latitude, longitude = city_parts
            latitude, longitude = float(latitude), float(longitude)

            logging.info(f"Attempting to download data for {city_name} from {date_from} to {date_to}")

            self.button.setDisabled(True)
            data = self.get_weather_data(latitude, longitude, date_from, date_to)
            logging.info(f"Successfully downloaded data. Shape: {data.X.shape}")

            self.Outputs.time_series.send(data)
        except requests.RequestException as e:
            logging.error(f"Network error: {str(e)}")
            self.Error.download_error(f"Network error: {str(e)}")
        except ValueError as e:
            logging.error(f"Data error: {str(e)}")
            self.Error.download_error(f"Data error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            self.Error.download_error(f"Unexpected error: {str(e)}")
        finally:
            self.button.setDisabled(False)


if __name__ == "__main__":
    WidgetPreview(OWWeatherData).run()