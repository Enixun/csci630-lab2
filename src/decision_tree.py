from math import log
from reprlib import Repr

class DecisionTree(dict):
  threshold = 0.00001
  @staticmethod
  def generate_answers(examples, answer_index=None):
    if answer_index is None:
      answer_index = len(examples[0]) - 1
    answers = {}
    for example in examples:
      answer = example[answer_index]
      if answer not in answers:
        answers[answer] = 0
      answers[answer] += 1
    return answers

  @staticmethod
  def majority(examples):
    cur_max = None
    answers = DecisionTree.generate_answers(examples)
    for answer in answers:
      if cur_max is None or answers[cur_max] < answers[answer]:
        cur_max = answer
    return cur_max
  
  @staticmethod
  def get_entropy(examples):
    answers = DecisionTree.generate_answers(examples)
    entropy = 0
    for answer in answers:
      frequency = answers[answer] / len(examples)
      entropy += -frequency*log(frequency, 2)
    return entropy
  
  @staticmethod
  def partition(examples, attribute_index, answer):
    partition = []
    for example in examples:
      if example[attribute_index] == answer:
        partition.append(example)
    return partition
  
  @staticmethod
  def best_question(examples, attributes, parent_entropy=None):
    if parent_entropy is None:
      parent_entropy = DecisionTree.get_entropy(examples)
    total_count = len(examples)
    best_attr = None
    best_gain = 0
    best_options = None
    for attr_index in range(len(attributes) - 1):
      if attributes[attr_index] is None: continue
      gain = parent_entropy
      options = DecisionTree.generate_answers(examples, attr_index)
      for option, count in options.items():
        new_examples = DecisionTree.partition(examples, attr_index, option)
        options[option] = new_examples
        new_entropy = DecisionTree.get_entropy(new_examples)
        gain -= new_entropy * count / total_count
      if best_attr is None or best_gain < gain:
        best_attr = attr_index
        best_gain = gain
        best_options = options

    return [best_attr, best_options]
  
  @staticmethod
  def no_more(attributes):
    for attr_index in range(len(attributes) - 1):
      if attributes[attr_index] is not None:
        return False
    return True

  def __init__(self, examples, attributes, parent=None):
    self.value = None
    self.best_question = None
    self.children = None
    if len(examples) == 0:
      self.value = self.majority(parent)
      return
    entropy = DecisionTree.get_entropy(examples)
    if (
      entropy == 0 or entropy < DecisionTree.threshold or 
      DecisionTree.no_more(attributes)
    ):
      print(attributes,entropy)
      self.value = DecisionTree.majority(examples)
      return
    best_question, answers = DecisionTree.best_question(examples, attributes, entropy)
    self.best_question = attributes[best_question]
    new_attributes = attributes.copy()
    new_attributes[best_question] = None
    self.children = {}
    for answer, new_examples in answers.items():
      self.children[answer] = DecisionTree(new_examples,new_attributes,self)

  def __repr__(self):
    return (
      "DecisionTree(" + 
      (("value:" + str(self.value)) if self.value is not None else '') +
      (("\nquestion:" + self.best_question) if self.best_question is not None else '')+
      (("\nchildren{\n" + '\n'.join(map(lambda i: self.best_question + " " +repr(i[0]) + ': ' + repr(i[1]),self.children.items()))+"\n}") if self.children is not None else '')+
      ")"
    )