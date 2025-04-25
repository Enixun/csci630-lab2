import requests
import re
from json import dumps, loads
from datetime import timedelta
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
    'attributes': WeatherReport.attributes,
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
  Contruct a Decision Tree for checking if it will be hotter in Rochester than the previous day. 
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
    if (
      sample_data.get(prev_day,None) is None or 
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


def predict(model_type:str,day5:dict=None,day4:dict=None,day3:dict=None,day2:dict=None,day1:dict=None):
  report_data = get_training_data('./raw1.json')
  examples, attributes = construct_hotter_daily_data(report_data, 'training')
  model = None
  if model_type == 'besttree':
    model = DecisionTree(examples, list(attributes))
  elif model_type == 'bestforest':
    model = RandomForest(9, len(attributes) - 5)
    model.train(examples, list(attributes))
  else: raise Exception('Invalid model')
  test_examples, _, answers = construct_hotter_daily_data(report_data, 'testing')
  num_correct = 0
  for i in range(len(test_examples)):
    if model.predict(test_examples[i]) == answers[i]:
      num_correct += 1
  print('accuracy', num_correct / len(test_examples))
  return [model.predict(test_examples[0])]


if __name__ == "__main__":
  # data = get_training_data()

  # construct_hotter_daily_data(data)
  print(predict('besttree'))
  print(predict('bestforest'))