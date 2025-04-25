import requests
import re
from json import dumps, loads
from datetime import date, timedelta, datetime
from src.data_collector import WeatherReport
from src.decision_tree import DecisionTree
from src.random_forest import RandomForest

cities = [
  'BUF','DTW','ART','CLE','GRR','LAN','AZO','TOL','ERI','FWA','SBN','PIT',
  'MGW','MDT','BFD','CVG','CMH','DAY','ORD','FNT','MBS','MKG','IND','MKE','ROC'
]

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


def get_training_data(data_file:str='./raw.json'):
  """
  Read from data file or query forecast.weather.gov and parse to WeatherReport format, writing contents to file.
  """
  try:
    return data_store(data_file)
  except OSError as ose: {
    print(ose, 'Fetching data...')
  }
  aggregate = {
    'attributes': WeatherReport.attributes[1:],
    'training': {},
    'testing': {}
  }
  for city in cities:
    for i in range(50):
      res = requests.get(f'https://forecast.weather.gov/product.php?site=BUF&issuedby={city}&product=CF6&format=txt&version={i+1}&glossary=0')
      report_string = re.search('(?:<pre.*?>)(.+)(?=</pre>)', res.text, re.DOTALL).group(1)
      wr = WeatherReport(report_string).to_dict()
      # print(wr)
      subset = 'training' if i > 13 else 'testing'
      for date in wr['reports'].keys():
        if date not in aggregate[subset]: 
          aggregate[subset][date] = {}
        aggregate[subset][date][city] = wr['reports'][date]
      print(city, i+1, 'done')
  data_store(data_file,operation='w',data=aggregate)
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


def construct_hotter_daily_data(weather_data:dict,nested=None):
  """
  Contruct vectors for checking if it will be hotter in Rochester than the previous day. 
  """
  attributes = weather_data['attributes']
  sample_data = weather_data[nested] if nested is not None else weather_data
  new_attributes = []
  data = []
  # 0-1, precip indices
  # else, check_missing_attr
  attribute_indices = [
    attributes.index('WTR PCPN'),
    attributes.index('SNOW'),
    attributes.index('DEP'),
    attributes.index('AVG TEMP (F)')
  ]
  temp_ind = attribute_indices[len(attribute_indices)-1]
  greater_than = lambda x,y: x > y
  less_than = lambda x,y: x < y

  for i in range(len(attribute_indices)):
    for city in cities:
      if i < 2:
        new_attributes.append(f'{attributes[attribute_indices[i]]} IN {city} {1} DAY AGO')
      elif city == 'ROC' and i == len(attribute_indices) - 1:
        new_attributes.append('HOTTER TODAY')
      elif i == 2:
        new_attributes.append(f'{city} LESS {attributes[attribute_indices[i]]} THAN ROC {1} DAY AGO')
      else:
        new_attributes.append(f'{city} MORE {attributes[attribute_indices[i]]} THAN ROC {1} DAY AGO')

  for day in sample_data.keys():
    prev_day = WeatherReport.date_to_str(WeatherReport.str_to_date(day) - timedelta(days=1))
    prev_two_day = WeatherReport.date_to_str(WeatherReport.str_to_date(day) - timedelta(days=2))
    prev_three_day = WeatherReport.date_to_str(WeatherReport.str_to_date(day) - timedelta(days=3))
    prev_four_day = WeatherReport.date_to_str(WeatherReport.str_to_date(day) - timedelta(days=4))
    if (
      sample_data.get(prev_day,None) is None or 
      sample_data.get(prev_two_day,None) is None or 
      sample_data.get(prev_three_day,None) is None or 
      sample_data.get(prev_four_day,None) is None or 
      sample_data[day].get('ROC',None) is None or
      sample_data[prev_day].get('ROC',None) is None
    ):
      continue
    roc_today = sample_data[day]['ROC']
    roc_prev = sample_data[prev_day]['ROC']
    hotter_today = check_missing_attribute(roc_today[temp_ind],roc_prev[temp_ind],greater_than)
    if hotter_today is None: continue
    attrs = []
    for i in range(len(attribute_indices)):
      ai = attribute_indices[i]
      roc_prev_attr = roc_prev[ai]
      for city in cities:
        if sample_data[prev_day].get(city, None) is None:
          attrs.append(None)
          continue
        city_prev_attr = sample_data[prev_day][city][ai]
        if i < 2:
          attrs.append(check_precip(city_prev_attr))
        elif city == 'ROC' and i == len(attribute_indices) - 1:
          attrs.append(roc_today[ai] > roc_prev_attr)
        elif i == 2:
          attrs.append(check_missing_attribute(city_prev_attr,roc_prev_attr,less_than))
        else:
          attrs.append(check_missing_attribute(city_prev_attr,roc_prev_attr,greater_than))
    data.append(tuple(attrs))

  if nested == 'testing':
    answers = tuple(map(lambda x: x[len(x) - 1],data))
    data = tuple(map(lambda x: x[:-1],data))
    new_attributes = new_attributes[:-1]
    return (data, new_attributes, answers)
  return (data, new_attributes)


def predict(model_type:str,day5:dict,day4:dict,day3:dict,day2:dict,day1:dict):
  aggregate = { 
    'attributes': WeatherReport.attributes[1:],
    'reports': {}
    }
  for arg in [day1,day2,day3,day4,day5]:
    day = None
    for city, report in arg.items():
      wr = WeatherReport.parse_report_str(report)
      if day is None:
        d = date.today()
        while d.day != wr[0]:
          d = d - timedelta(days=1)
        day = WeatherReport.date_to_str(d)
      report = wr[1:]
      if aggregate['reports'].get(day, None) is None: aggregate['reports'][day] = {}
      aggregate['reports'][day][city] = report

  report_data = get_training_data('./raw1.json')
  examples, attributes = construct_hotter_daily_data(report_data, 'training')
  model = None
  if model_type == 'besttree':
    model = DecisionTree(examples, list(attributes))
  elif model_type == 'bestforest':
    model = RandomForest(9, len(attributes) - 5)
    model.train(examples, list(attributes))
  else: raise Exception('Invalid model')

  test_examples, _ = construct_hotter_daily_data(aggregate, 'reports')
  return [model.predict(test_examples[0])]


if __name__ == "__main__":
  # data = get_training_data()

  # construct_hotter_daily_data(data)
  print(predict('besttree',
    {'DTW': '16 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '16 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
    {'DTW': '15 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '15 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
    {'DTW': '14 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '14 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
    {'DTW': '13 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '13 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
    {'DTW': '12 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '12 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '12 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '12 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '12 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '12 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '12 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '12 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '12 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '12 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '12 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '12 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '12 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '12 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '12 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '12 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '12 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '12 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '12 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '12 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '12 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '12 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '12 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '12 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '12 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'}
  ))
  # print(predict('bestforest',
  #   {'DTW': '16 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '16 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
  #   {'DTW': '15 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '15 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
  #   {'DTW': '14 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '14 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
  #   {'DTW': '13 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '13 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '13 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '13 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '13 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '13 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '13 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '13 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '13 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '13 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '13 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '13 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '13 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '13 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '13 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '13 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '13 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '13 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '13 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '13 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '13 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '13 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '13 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '13 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '13 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'},
  #   {'DTW': '12 27 21 24 -3 41 0 0.14 1.5 T 10.3 18 50 M M 10 16 27 50', 'ROC': '12 37 23 30 3 35 0 0.21 2.2 4 12.6 21 170 M M 10 126 34 150', 'BUF': '12 31 17 24 -2 41 0 0.13 0.9 4 13.2 23 70 M M 9 16 28 100', 'ART': '12 23 7 15 -5 50 0 T M M 12.7 22 50 M M 6 27 340', 'CLE': '12 39 23 31 1 34 0 0.08 0.0 0 9.6 20 50 M M 10 168 25 160', 'GRR': '12 23 19 21 -5 44 0 0.22 2.0 T 11.2 20 60 M M 10 19 26 60', 'LAN': '12 24 18 21 -4 44 0 0.38 4.5 2 8.9 18 50 M M 10 129 25 40', 'AZO': '12 25 21 23 -3 42 0 0.14 M M 7.9 16 50 M M 10 128 24 60', 'TOL': '12 32 23 28 -1 37 0 0.06 T 0 12.6 24 70 M M 10 16 33 90', 'ERI': '12 38 19 29 1 36 0 0.11 0.3 T 12.2 31 150 M M 9 18 45 160', 'FWA': '12 29 24 27 -1 38 0 0.12 0.1 0 14.6 26 70 M M 10 16 34 70', 'SBN': '12 29 23 26 0 39 0 0.25 2.8 0 8.3 17 100 M M 10 128 25 100', 'PIT': '12 39 30 35 4 30 0 0.07 T T 9.4 17 120 M M 10 1 25 120', 'MGW': '12 45 30 38 4 27 0 0.05 M M 7.5 16 140 M M 10 18 25 130', 'MDT': '12 34 28 31 -2 34 0 0.16 1.0 1 4.4 14 120 M M 10 1 17 120', 'BFD': '12 33 20 27 3 38 0 0.03 M M 11.3 23 150 M M 10 168 37 140', 'CVG': '12 42 30 36 2 29 0 0.18 0.2 3 9.8 18 300 M M 10 126 24 80', 'CMH': '12 43 30 37 5 28 0 0.11 T T 9.2 15 90 M M 10 18 20 90', 'DAY': '12 38 30 34 2 31 0 0.21 T T 10.2 16 70 M M 10 168 23 20', 'ORD': '12 29 19 24 -4 41 0 0.18 2.8 0 11.8 20 60 M M 10 12 26 50', 'FNT': '12 48 25 37 4 28 0 0.00 0.0 0 M M M M M M M M', 'MBS': '12 44 28 36 4 29 0 0.00 0.0 0 9.5 20 40 M M 1 23 40', 'MKG': '12 55 32 44 10 21 0 0.00 0.0 0 10.4 25 110 M M 5 32 120', 'IND': '12 75 39 57 16 8 0 0.00 0.0 0 8.5 20 70 M M 6 28 70', 'MKE': '12 43 32 38 3 27 0 0.00 0.0 0 7.7 18 110 M M 5 25 100'}
  # ))