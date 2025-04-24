import requests
import re

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
    return tuple(parsed_data)

  def __init__(self, report:str):
    self.month = re.search(r'MONTH:\s+(\w+)',report,re.DOTALL).group(1)
    self.year = re.search(r'YEAR:\s+(\w+)',report,re.DOTALL).group(1)
    self.city = re.search(r'CF6(\w{3})',report,re.DOTALL).group(1)
    report_data = re.findall(r'(?<=\n) *?(?:(?:\w+|[0-9.\-]+) +?){17,18}(?:\w+|[0-9.\-]+)(?=\n)', report)
    self.reports = tuple(map(lambda r: WeatherReport.parse_report_str(r),report_data[1:]))

  def __repr__(self):
    return (
      "WeatherReport(" +
      "\ncity:"+ self.city +
      "\nmonth:"+ self.month +
      "\nyear:"+ self.year +
      "\nreports:\n  "+ ',\n  '.join(map(lambda r: str(r),self.reports)) +
      "\n)"
    )


def main():
  res = requests.get('https://forecast.weather.gov/product.php?site=BUF&issuedby=ROC&product=CF6&format=txt&version=2&glossary=0')
  report_string = re.search('(?:<pre.*?>)(.+)(?=</pre>)', res.text, re.DOTALL).group(1)
  wr = WeatherReport(report_string)
  print(wr)


if __name__ == '__main__':
  main()