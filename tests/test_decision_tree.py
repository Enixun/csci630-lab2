from unittest import TestCase
from src.decision_tree import DecisionTree

class DecisionTreeTests(TestCase):
  def setUp(self):
    self.test_data = [
      ['N','Y','N','Y','Y'],
      ['N','N','N','N','Y'],
      ['Y','Y','Y','Y','Y'],
      ['Y','N','Y','N','N'],
      ['N','N','Y','Y','N'],
      ['N','Y','N','N','Y'],
      ['Y','Y','N','Y','Y'],
      ['N','Y','Y','N','N'],
      ['Y','N','N','Y','Y'],
      ['Y','N','N','N','N'],
      ['Y','Y','Y','Y','N'],
      ['N','Y','N','N','Y'],
      ['N','N','Y','Y','Y'],
      ['Y','N','Y','N','N'],
      ['Y','N','N','Y','Y'],
      ['N','N','Y','N','Y'],
      ['Y','Y','Y','Y','N'],
      ['N','Y','N','N','Y'],
      ['Y','N','Y','N','Y'],
      ['Y','Y','N','N','N'],
    ]

    self.index_map = [
      'veg?', 'iphone?', 'student?', 'american?', 'drinks coffee'
    ]

  def test_class_data(self):
    test_data = [
      ['N','Y','Y','Y','Y'],
      ['N','N','N','Y','N'],
      ['N','N','Y','N','N'],
      ['N','Y','N','Y','Y'],
      ['Y','Y','Y','N','Y'],
      ['N','N','Y','N','N'],
      ['N','Y','Y','N','Y'],
      ['N','N','N','Y','N'],
      ['N','N','Y','N','Y'],
      ['N','Y','N','N','N'],
    ]
    index_map = [
      'reserv?', 'long wait?', 'weekend?', 'rain?', 'will wait?'
    ]

    dt = DecisionTree(test_data, index_map)
    print(dt)

    self.assertIsNotNone(dt, 'Expected Decision Tree to exists')

  def test_hw_data(self):
    expected = {
      'best_question':'student?',
      'children':{
        'N': {
          'best_question':'veg?',
          'children':{
            'N': {'value':'Y'},
            'Y': {
              'best_question':'american?',
              'children':{
                'N':{'value':'N'},
                'Y':{'value':'Y'}
              }
            }
          }
        },
        'Y':{
          'best_question':'iphone?',
          'children':{
            'N':{
              'best_question':'veg?',
              'children':{
                'N':{'value':'Y'},
                'Y':{'value':'N'}
              }
            },
            'Y':{
              'best_question':'veg?',
              'children':{
                'N':{'value':'N'},
                'Y':{'value':'N'},
              }
            }
          }
        }
      }
    }

    dt = DecisionTree(self.test_data, self.index_map, 3)
    self.assertEqual(expected,dt, "Build decision tree did not match expected output.")

  def test_predict_simple_data(self):
    sample = self.test_data[0][:-1]
    expected = self.test_data[0][len(sample)]
    dt = DecisionTree(self.test_data, self.index_map, 3)
    actual = dt.predict(sample)
    self.assertEqual(expected,actual, f"Expected prediction of {sample} to be {expected}.")

  def test_predict_contradictory_data(self):
    samples = list(filter(lambda x : (x[0] == 'Y' and x[1] == 'Y' and x[2] == 'Y' and x[3] == 'Y'), self.test_data))
    majority = DecisionTree.majority(samples)
    dt = DecisionTree(self.test_data, self.index_map, 3)
    for sample in samples:
      input = sample[:-1]
      predicted_outcome = dt.predict(input)
      real_outcome = sample[len(input)]
      if real_outcome == majority:
        self.assertEqual(
          real_outcome, predicted_outcome, 
          f"Outcome {real_outcome} should have matched predicted outcome {predicted_outcome}"
        )
      else:
        self.assertNotEqual(
          real_outcome, predicted_outcome,
          f"Outcome {real_outcome} should not have matched predicted outcome {predicted_outcome}"
        )
      