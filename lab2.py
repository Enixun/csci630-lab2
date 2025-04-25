import requests
import re
from json import dumps, loads
from datetime import timedelta
from src.data_collector import WeatherReport
from src.decision_tree import DecisionTree

def data_store(file='./raw.json',operation='r',data=None,indent=None):
  """
  Open a file and either read and parse its JSON, or stringify JSON and write to file. Pass in indent to add pretty margins.
  """
  with open(file, operation) as file:
    if operation == 'r':
      json_str = file.read()
      return loads(json_str)
    elif operation == 'w':
      file.write(dumps(data,indent=indent))


def get_training_data():
  """
  Read from data file or query forecast.weather.gov and parse to WeatherReport format, writing contents to file.
  """
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
  """
  Check if either attribute is missing before comparing with a comparison function.
  """
  if attr1 == 'M' or attr2 == 'M': return None
  return comparison(attr1,attr2)


def check_precip(amount:str|float,threshold=0.2)->bool:
  """
  Check for greater than trace amounts of precipitation.
  """
  if amount == 'T': return False
  return check_missing_attribute(amount, threshold, lambda x,y: x > y)


def construct_hotter_daily_data(weather_data:dict)->DecisionTree:
  """
  Contruct a Decision Tree for checking if it will be hotter in Rochester than the previous day. 
  """
  attributes = weather_data['attributes']
  new_attributes = (
    'RAINED YESTERDAY IN BUF',
    'RAINED YESTERDAY IN DTW',
    'RAINED YESTERDAY IN ROC',
    'SNOWED YESTERDAY IN BUF',
    'SNOWED YESTERDAY IN DTW',
    'SNOWED YESTERDAY IN ROC',
    'HOTTER YESTERDAY IN BUF',
    'HOTTER YESTERDAY IN DTW',
    'HOTTER TODAY'
  )
  data = []
  # 0-1, precip indices
  # else, check_missing_attr
  attribute_indices = [
    attributes.index('WTR PCPN'),
    attributes.index('SNOW'),
    attributes.index('AVG TEMP (F)')
  ]
  temp_ind = attribute_indices[len(attribute_indices)-1]
  cities = ['BUF','DTW','ROC']

  for day in weather_data['ROC'].keys():
    roc_today = weather_data['ROC'][day]
    prev_day = WeatherReport.date_to_str(WeatherReport.str_to_date(day) - timedelta(days=1))
    if weather_data['ROC'].get(day, None) is None or weather_data['ROC'].get(prev_day, None) is None:
      continue
    roc_prev = weather_data['ROC'][prev_day]
    greater_than = lambda x,y: x > y
    hotter_today = check_missing_attribute(roc_today[temp_ind],roc_prev[temp_ind],greater_than)
    if hotter_today is None: continue
    attrs = []
    for i in range(len(attribute_indices)):
      ai = attribute_indices[i]
      roc_prev_attr = roc_prev[ai]
      for city in cities:
        city_prev_attr = weather_data[city][prev_day][ai]
        if i < 2:
          attrs.append(check_precip(city_prev_attr))
        elif city == 'ROC' and i == len(attribute_indices) - 1:
          attrs.append(roc_today[ai] > roc_prev_attr)
        else:
          attrs.append(check_missing_attribute(city_prev_attr,roc_prev_attr,greater_than))
    data.append(tuple(attrs))
        
  print(data)
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