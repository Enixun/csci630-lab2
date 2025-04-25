from .decision_tree import DecisionTree
import random

class RandomForest():
  def __init__(self, num_trees=4, num_attributes=5,max_depth=5):
    self.trees:list[DecisionTree] = []
    self.num_trees = num_trees
    self.num_attributes = num_attributes
    self.max_depth = max_depth

  
  def attribute_subset(self,attributes):
    indices = set()
    indices.add(len(attributes) - 1)
    while len(indices) <= min(len(attributes) - 1, self.num_attributes):
      indices.add(random.randint(0, len(attributes) - 2))
    list(indices).sort()
    return [attr for attr in map(lambda i: attributes[i], indices)]


  def train(self, examples:list, attributes:list):
    if not (len(attributes) > self.num_attributes):
      raise Exception(f'Invalid number of attribute for random forest. Must have fewer attributes than {self.num_attributes}, provided {len(attributes)}')
    used_subsets = set()
    while(len(self.trees) < self.num_trees):
      subset = self.attribute_subset(attributes)
      if str(subset) not in used_subsets:
        self.trees.append(
          DecisionTree(
            examples,
            subset,
            self.max_depth
          )
        )
        used_subsets.add(str(subset))

  def predict(self, example):
    freq = {}
    max = None
    for tree in self.trees:
      result = tree.predict(example)
      if freq.get(result, None) is None:
        freq[result] = 0
      freq[result] += 1
      if max is None or freq[max] < freq[result]:
        max = result
    # print(freq)
    return max


  def __repr__(self):
    return (
      "RandomForest(" + 
      "num_trees:" + str(self.num_trees) +
      "num_attributes:" + str(self.num_attributes) +
      "max_dept:" + str(self.max_depth) +
      "trees:" + str(self.trees) +
      ")"
    )