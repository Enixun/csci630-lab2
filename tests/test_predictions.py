from unittest import TestCase
from lab2 import get_training_data, construct_hotter_daily_data
from src.decision_tree import DecisionTree
from src.random_forest import RandomForest

class DecisionTreeTests(TestCase):
  def setUp(self):
    self.data = get_training_data()

  def test_hotter_today_decision_tree(self):
    data, attributes = construct_hotter_daily_data(self.data, 'training')
    dt = DecisionTree(data,list(attributes))
    self.assertIsNotNone(dt)
    test_data, test_attrs = construct_hotter_daily_data(self.data, 'testing')
    number_correct = 0
    none_count = 0
    for example in test_data:
      if dt.predict(example) is True:
        number_correct += 1
      elif dt.predict(example) is None:
        none_count += 1
    accuracy = number_correct / len(test_data)
    print('decision tree', number_correct, len(test_data), accuracy)

  def test_hotter_today_random_forest(self):
    data, attributes = construct_hotter_daily_data(self.data, 'training')
    test_data, test_attrs = construct_hotter_daily_data(self.data, 'testing')
    rf = RandomForest(10,len(test_attrs)-3,len(test_attrs))
    rf.train(data,attributes)
    self.assertIsNotNone(rf)
    number_correct = 0
    none_count = 0
    for example in test_data:
      if rf.predict(example) is True:
        number_correct += 1
      elif rf.predict(example) is None:
        none_count += 1
    accuracy = number_correct / (len(test_data) - none_count)
    print('random forest', number_correct, len(test_data), accuracy)
