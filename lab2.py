import requests
import re
from json import dumps, loads
from datetime import timedelta
from src.data_collector import WeatherReport
from src.decision_tree import DecisionTree

def data_store(file='./raw.json',operation='r',data=None,indent=None):
  with open(file, operation) as file:
    if operation == 'r':
      json_str = file.read()
      return loads(json_str)
    elif operation == 'w':
      file.write(dumps(data,indent=indent))


def get_training_data():
  try:
    return data_store()
  except OSError as ose: {
    print(ose, 'Fetching data...')
  }
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
  data_store(operation='w',data=aggregate,indent=2)
  return aggregate


def check_missing_attribute(attr1, attr2, comparison:callable)->bool:
  if attr1 == 'M' or attr2 == 'M': return None
  return comparison(attr1,attr2)


def check_precip(amount:str|float,threshold=0.2)->bool:
  if amount == 'T': return False
  return check_missing_attribute(amount, threshold, lambda x,y: x > y)


def construct_hotter_daily_data(weather_data:dict):
  attributes = weather_data['attributes']
  new_attributes = (
    'HOTTER YESTERDAY IN BUF',
    'HOTTER YESTERDAY IN DTW',
    'RAINED YESTERDAY IN ROC',
    'HOTTER TODAY'
  )
  data = []
  avg_temp_index = attributes.index('AVG TEMP (F)')
  rain_index = attributes.index('WTR PCPN')

  for day in weather_data['ROC'].keys():
    prev_day = WeatherReport.date_to_str(WeatherReport.str_to_date(day) - timedelta(days=1))
    if weather_data['ROC'].get(prev_day, None) is not None:
      rochester_temp_today = weather_data['ROC'][day][avg_temp_index]
      rochester_temp_yesterday = weather_data['ROC'][prev_day][avg_temp_index]
      buffalo_temp_yesterday = weather_data['BUF'][prev_day][avg_temp_index]
      detroit_temp_yesterday = weather_data['DTW'][prev_day][avg_temp_index]
      rained_yesterday = weather_data['ROC'][prev_day][rain_index]
      greater_than = lambda x,y: x > y
      hotter_today = check_missing_attribute(rochester_temp_today, rochester_temp_yesterday, greater_than)
      if hotter_today is None: continue
      data.append((
        check_missing_attribute(buffalo_temp_yesterday, rochester_temp_yesterday, greater_than),
        check_missing_attribute(detroit_temp_yesterday, rochester_temp_yesterday, greater_than),
        check_precip(rained_yesterday),
        hotter_today
      ))
  # print(data)
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

  construct_hotter_daily_data(data)