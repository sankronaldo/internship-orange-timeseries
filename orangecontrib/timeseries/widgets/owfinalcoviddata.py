import logging
from datetime import datetime, timedelta, date
from Orange.data import Table, Domain, TimeVariable, ContinuousVariable
from AnyQt.QtCore import QDate
from AnyQt.QtWidgets import QDateEdit, QComboBox, QFormLayout
from orangewidget.utils.widgetpreview import WidgetPreview

from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Output

from orangecontrib.timeseries import Timeseries

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OWCOVIDData(widget.OWWidget):
    name = 'COVID-19 Data'
    description = "Fetch COVID-19 data for various countries"
    icon = 'icons/final.svg'
    priority = 10

    class Outputs:
        time_series = Output("Time series", Timeseries)

    QT_DATE_FORMAT = 'yyyy-MM-dd'
    PY_DATE_FORMAT = '%Y-%m-%d'
    MIN_DATE = date(2020, 1, 22)  # First date available in the API

    date_from = settings.Setting(
        (datetime.now().date() - timedelta(365)).strftime(PY_DATE_FORMAT))
    date_to = settings.Setting(datetime.now().date().strftime(PY_DATE_FORMAT))

    countries = settings.Setting([
        'netherlands', 'india', 'us', 'brazil', 'france', 'germany', 'uk', 'italy', 'spain',
        'canada', 'australia', 'mexico', 'south-africa', 'russia', 'china'
    ])

    frequency = settings.Setting('daily')

    want_main_area = False
    resizing_enabled = False

    class Error(widget.OWWidget.Error):
        download_error = widget.Msg('{}\nNo internet? Invalid country selection?')

    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        gui.widgetBox(self.controlArea, True, orientation=layout)

        self.country_combo = QComboBox()
        self.country_combo.addItems(self.countries)
        layout.addRow("Country:", self.country_combo)

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

        self.frequency_combo = QComboBox()
        self.frequency_combo.addItems(['daily', 'weekly', 'monthly', 'quarterly'])
        self.frequency_combo.setCurrentText(self.frequency)
        self.frequency_combo.currentTextChanged.connect(self.set_frequency)
        layout.addRow("Frequency:", self.frequency_combo)

        self.button = gui.button(
            self.controlArea, self, 'Download', callback=self.download)

    def set_frequency(self, value):
        self.frequency = value

    def get_covid_data(self, country, start_date, end_date, frequency):
        api_url = f'https://disease.sh/v3/covid-19/historical/{country}?lastdays=all'

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

            if 'timeline' not in data:
                raise ValueError("No data found in the response")

            timeline = data['timeline']
            cases = timeline.get('cases', {})
            deaths = timeline.get('deaths', {})

            df = pd.DataFrame({
                'Date': pd.to_datetime(list(cases.keys()), format='%m/%d/%y'),
                'Cases': list(cases.values()),
                'Deaths': list(deaths.values())
            })

            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

            df.set_index('Date', inplace=True)
            if frequency == 'daily':
                df = df.resample('D').sum()
            elif frequency == 'weekly':
                df = df.resample('W').sum()
            elif frequency == 'monthly':
                df = df.resample('M').sum()
            elif frequency == 'quarterly':
                df = df.resample('Q').sum()

            df = df.reset_index()

            # Convert DataFrame to Orange Table
            domain = Domain([ContinuousVariable(name) for name in df.columns if name != 'Date'],
                            metas=[TimeVariable("Date", have_time=True)])

            table = Table.from_numpy(
                domain,
                df.drop('Date', axis=1).values,
                metas=df['Date'].astype(int).to_numpy().reshape(-1, 1) // 10 ** 9
            )

            return Timeseries(table)

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

            country = self.countries[self.country_combo.currentIndex()]

            logging.info(f"Attempting to download data for {country} from {date_from} to {date_to}")

            self.button.setDisabled(True)
            data = self.get_covid_data(country, date_from, date_to, self.frequency)
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
    WidgetPreview(OWCOVIDData).run()