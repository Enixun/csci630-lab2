from unittest import TestCase
from src.decision_tree import DecisionTree

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

    expected = {
      'best_question':'student?',
      'children':{
        0: {
          'best_question':'veg?',
          'children':{
            0: {'value':1},
            1: {
              'best_question':'american?',
              'children':{
                0:{'value':0},
                1:{'value':1}
              }
            }
          }
        },
        1:{
          'best_question':'iphone?',
          'children':{
            0:{
              'best_question':'veg?',
              'children':{
                0:{'value':1},
                1:{'value':0}
              }
            },
            1:{
              'best_question':'veg?',
              'children':{
                0:{'value':0},
                1:{'value':0},
              }
            }
          }
        }
      }
    }

    dt = DecisionTree(test_data, index_map, 3)
    self.assertEqual(expected,dt, "Build decision tree did not match expected output.")