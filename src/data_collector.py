import requests
import re
from datetime import date, datetime

def month_map(month:str)->int:
  match month:
    case 'JANUARY':
      return 1
    case 'FEBRUARY':
      return 2
    case 'MARCH':
      return 3
    case 'APRIL':
      return 4
    case 'MAY':
      return 5
    case 'JUNE':
      return 6
    case 'JULY':
      return 7
    case 'AUGUST':
      return 8
    case 'SEPTEMBER':
      return 9
    case 'OCTOBER':
      return 10
    case 'NOVEMBER':
      return 11
    case 'DECEMBER':
      return 12
    case _:
      raise Exception('Invalid Month')

class WeatherReport():
  attributes = (
    'DAY',
    'MAX TEMP (F)',
    'MIN TEMP (F)',
    'AVG TEMP (F)',
    'DEP',
    'HDD',
    'CDD',
    'WTR PCPN',
    'SNOW',
    '12Z DPTH',
    'AVG WIND SPD',
    'MAX WIND SPD',
    '2MIN DIR',
    'MIN SUNSHINE',
    'PSBL SUNSHINE',
    'S-S SKY',
    'WX',
    'PK WIND SPD',
    'PK WIND DIR',
  )

  @staticmethod
  def string_to_type(string:str):
    if re.search(r'[a-zA-Z]', string) is not None:
      return string
    elif re.search(r'\.',string) is not None:
      return float(string)
    else:
      return int(string)

  @staticmethod
  def parse_report_str(report_string:str)->tuple[float|str]:
    data = report_string.split()
    optional_column_found = len(data) == len(WeatherReport.attributes)
    parsed_data = []
    for i in range(len(WeatherReport.attributes)):
      if not optional_column_found and i > 16:
        if i > 17: parsed_data.append(WeatherReport.string_to_type(data[i-1]))
        else: parsed_data.append(None)
      else: parsed_data.append(WeatherReport.string_to_type(data[i]))
    return parsed_data

  def parse_report(self, report_string:str)->tuple[float|str]:
    parsed = WeatherReport.parse_report_str(report_string)
    parsed[0] = self.city + '-' + date(int(self.year),month_map(self.month),parsed[0]).strftime('%D')
    # print(datetime.strptime(parsed[0],'%m/%d/%y'))
    return tuple(parsed)
  
  @staticmethod
  def date_to_str(date:datetime):
    return datetime.strftime(date,'%m/%d/%y')
  
  @staticmethod
  def str_to_date(date_str:str):
    return datetime.strptime(date_str,'%m/%d/%y')
  
  def to_dict(self)->dict:
    report_dict = {
      'attributes': self.attributes[1:],
      self.city: {}
    }
    for report in self.reports:
      report_dict[self.city][re.search(r'(?<=-)\S+',report[0]).group(0)] = report[1:]
    return report_dict

  def __init__(self, report:str):
    self.month = re.search(r'MONTH:\s+(\w+)',report,re.DOTALL).group(1)
    self.year = re.search(r'YEAR:\s+(\w+)',report,re.DOTALL).group(1)
    self.city = re.search(r'CF6(\w{3})',report,re.DOTALL).group(1)
    report_data = re.findall(r'(?<=\n) *?(?:(?:\w+|[0-9.\-]+) +?){17,18}(?:\w+|[0-9.\-]+)(?=\n)', report)
    self.reports = tuple(map(lambda r: self.parse_report(r),report_data[1:]))

  def __repr__(self):
    return (
      "WeatherReport(" +
      "\ncity:"+ self.city +
      ",\nmonth:"+ self.month +
      ",\nyear:"+ self.year +
      ",\nattributes:"+ str(WeatherReport.attributes) +
      ",\nreports:\n  "+ ',\n  '.join(map(lambda r: str(r),self.reports)) +
      "\n)"
    )


def main():
  for city in ['ROC', 'BUF', 'DTW']:
    for i in range(1,50):
      res = requests.get(f'https://forecast.weather.gov/product.php?site=BUF&issuedby={city}&product=CF6&format=txt&version={i}&glossary=0')
      report_string = re.search('(?:<pre.*?>)(.+)(?=</pre>)', res.text, re.DOTALL).group(1)
      wr = WeatherReport(report_string)
      print(wr)
      print(wr.to_dict())


if __name__ == '__main__':
  main()