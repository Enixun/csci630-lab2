from unittest import TestCase
from src.decision_tree import DecisionTree
from reprlib import Repr

class DecisionTreeTests(TestCase):
  def test_class_data(self):
    test_data = [
      [0,1,1,1,1],
      [0,0,0,1,0],
      [0,0,1,0,0],
      [0,1,0,1,1],
      [1,1,1,0,1],
      [0,0,1,0,0],
      [0,1,1,0,1],
      [0,0,0,1,0],
      [0,0,1,0,1],
      [0,1,0,0,0],
    ]
    index_map = [
      'reserv?', 'long wait?', 'weekend?', 'rain?', 'will wait?'
    ]

    dt = DecisionTree(test_data, index_map)
    print(dt)

    self.assertIsNotNone(dt, 'Expected Decision Tree to exists')

  def test_hw_data(self):
    test_data = [
      [0,1,0,1,1],
      [0,0,0,0,1],
      [1,1,1,1,1],
      [1,0,1,0,0],
      [0,0,1,1,0],
      [0,1,0,0,1],
      [1,1,0,1,1],
      [0,1,1,0,0],
      [1,0,0,1,1],
      [1,0,0,0,0],
      [1,1,1,1,0],
      [0,1,0,0,1],
      [0,0,1,1,1],
      [1,0,1,0,0],
      [1,0,0,1,1],
      [0,0,1,0,1],
      [1,1,1,1,0],
      [0,1,0,0,1],
      [1,0,1,0,1],
      [1,1,0,0,0],
    ]
    index_map = [
      'veg?', 'iphone?', 'student?', 'american?', 'drinks coffee'
    ]

    dt = DecisionTree(test_data, index_map)
    print(dt)

    self.assertIsNotNone(dt, 'Expected Decision Tree to exists')