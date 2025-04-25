from unittest import TestCase
from lab2 import get_training_data, construct_hotter_daily_data
from src.decision_tree import DecisionTree
from src.random_forest import RandomForest

class DecisionTreeTests(TestCase):
  def setUp(self):
    self.data = get_training_data('./raw.json')

  def test_hotter_today_decision_tree(self):
    data, attributes = construct_hotter_daily_data(self.data, 'training')
    dt = DecisionTree(data,list(attributes))
    self.assertIsNotNone(dt)
    test_data, test_attrs, answers = construct_hotter_daily_data(self.data, 'testing')
    number_correct = 0
    none_count = 0
    for i in range(len(test_data)):
      if dt.predict(test_data[i]) == answers[i]:
        number_correct += 1
      elif dt.predict(test_data[i]) is None:
        none_count += 1
    accuracy = number_correct / (len(test_data) - none_count)
    print('decision tree', number_correct, len(test_data) - none_count, accuracy)

  def test_hotter_today_random_forest(self):
    data, attributes = construct_hotter_daily_data(self.data, 'training')
    test_data, test_attrs, answers = construct_hotter_daily_data(self.data, 'testing')
    rf = RandomForest(10,len(test_attrs) - 1,8)
    rf.train(data,attributes)
    self.assertIsNotNone(rf)
    number_correct = 0
    none_count = 0
    for i in range(len(test_data)):
      if rf.predict(test_data[i]) == answers[i]:
        number_correct += 1
      elif rf.predict(test_data[i]) is None:
        none_count += 1
    accuracy = number_correct / (len(test_data) - none_count)
    print('random forest', number_correct, len(test_data) - none_count, accuracy)
