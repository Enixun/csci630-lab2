import requests
import re
from datetime import datetime, timedelta
from src.data_collector import WeatherReport
from src.decision_tree import DecisionTree

def get_training_data():
  aggregate = {
    'attributes': WeatherReport.attributes
  }
  for city in ['ROC', 'BUF', 'DTW']:
    aggregate[city] = {}
    for i in range(50):
      res = requests.get(f'https://forecast.weather.gov/product.php?site=BUF&issuedby={city}&product=CF6&format=txt&version={i+1}&glossary=0')
      report_string = re.search('(?:<pre.*?>)(.+)(?=</pre>)', res.text, re.DOTALL).group(1)
      wr = WeatherReport(report_string).to_dict()
      for date in wr[city].keys():
        aggregate[city][date] = wr[city][date]
      print(city, i+1, 'done')
  return aggregate

def construct_hotter_daily_data(weather_data:dict):
  attributes = weather_data['attributes']
  new_attributes = (
    'HOTTER YESTERDAY IN BUF',
    'HOTTER YESTERDAY IN DTW',
    'HOTTER TODAY'
  )
  data = []
  avg_temp_index = attributes.index('AVG TEMP (F)')

  for day in weather_data['ROC'].keys():
    prev_day = WeatherReport.date_to_str(WeatherReport.str_to_date(day) - timedelta(days=1))
    if weather_data['ROC'].get(prev_day, None) is not None:
      hotter_today = weather_data['ROC'][day][avg_temp_index] > weather_data['ROC'][prev_day][avg_temp_index]
      data.append((
        weather_data['BUF'][prev_day][avg_temp_index] > weather_data['ROC'][day][avg_temp_index],
        weather_data['DTW'][prev_day][avg_temp_index] > weather_data['ROC'][day][avg_temp_index],
        hotter_today
      ))
  print(DecisionTree(data,list(new_attributes)))


def predict(model_type:str,day5:dict,day4:dict,day3:dict,day2:dict,day1:dict):
  model = None
  if model_type == 'besttree':
    model = DecisionTree
  # elif model_type == 'bestforest':
    # TODO: RandomForest
  else: raise Exception('Invalid model')
  hotter_than_yesterday = model()
  return []

if __name__ == "__main__":
  data = get_training_data()
  print(data['attributes'])
  construct_hotter_daily_data(data)