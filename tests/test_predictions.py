from unittest import TestCase
from lab2 import get_training_data, construct_hotter_daily_data
from src.decision_tree import DecisionTree

class DecisionTreeTests(TestCase):
  def setUp(self):
    self.data = get_training_data()

  def test_hotter_today_dt(self):
    data, attributes = construct_hotter_daily_data(self.data, 'training')
    dt = DecisionTree(data,list(attributes))
    self.assertIsNotNone(dt)
    test_data, test_attrs = construct_hotter_daily_data(self.data, 'testing')
    number_correct = 0
    for example in test_data:
      if dt.predict(example) is True:
        number_correct += 1
    accuracy = number_correct / len(test_data)
    print(number_correct, len(test_data), accuracy)
