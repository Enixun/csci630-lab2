from unittest import TestCase
from src.random_forest import RandomForest
from src.decision_tree import DecisionTree

class RandomForestTests(TestCase):
  def setUp(self):
    self.test_data = [
      ('N','Y','N','Y','Y'),
      ('N','N','N','N','Y'),
      ('Y','Y','Y','Y','Y'),
      ('Y','N','Y','N','N'),
      ('N','N','Y','Y','N'),
      ('N','Y','N','N','Y'),
      ('Y','Y','N','Y','Y'),
      ('N','Y','Y','N','N'),
      ('Y','N','N','Y','Y'),
      ('Y','N','N','N','N'),
      ('Y','Y','Y','Y','N'),
      ('N','Y','N','N','Y'),
      ('N','N','Y','Y','Y'),
      ('Y','N','Y','N','N'),
      ('Y','N','N','Y','Y'),
      ('N','N','Y','N','Y'),
      ('Y','Y','Y','Y','N'),
      ('N','Y','N','N','Y'),
      ('Y','N','Y','N','Y'),
      ('Y','Y','N','N','N'),
    ]

    self.index_map = [
      'veg?', 'iphone?', 'student?', 'american?', 'drinks coffee'
    ]
    self.rf = RandomForest(4, 2, 3)
    self.rf.train(self.test_data,self.index_map)

  def test_hw_data(self):
    rf = RandomForest(4, 2, 3)
    rf.train(self.test_data, self.index_map)
    # print(rf)

  def test_predict_simple_data(self):
    sample = self.test_data[0][:-1]
    expected = self.test_data[0][len(sample)]
    actual = self.rf.predict(sample)
    self.assertEqual(expected,actual, f"Expected prediction of {sample} to be {expected}.")

  def test_predict_contradictory_data(self):
    samples = list(filter(lambda x : (x[0] == 'Y' and x[1] == 'Y' and x[2] == 'Y' and x[3] == 'Y'), self.test_data))
    majority = DecisionTree.majority(samples)
    for sample in samples:
      input = sample[:-1]
      predicted_outcome = self.rf.predict(input)
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
