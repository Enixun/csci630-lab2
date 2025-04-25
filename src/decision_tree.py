from math import log, inf

class DecisionTree():
  @staticmethod
  def generate_answers(examples:list, answer_index:int=None):
    """
    Generate a dict of possible values for an attribute in an set of entities with their frequency.

    Args:
      examples - a list of tuples representing data points.
      answer_index - the attribute of an entity being searched. Defaults to the last attribute.

    Returns:
      answers
    """
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
  def majority(examples:list):
    """
    Find the majority answer (final column) within a set of examples.

    Args:
      examples - a list of tuples representing data points.

    Returns:
      majority answer
    """
    cur_max = None
    answers = DecisionTree.generate_answers(examples)
    for answer in answers:
      if cur_max is None or answers[cur_max] < answers[answer]:
        cur_max = answer
    return cur_max
  
  @staticmethod
  def get_entropy(examples:list) -> float:
    """
    Calculate the entropy of set of examples based on their answer attribute (final column).

    Args:
      examples - a list of tuples representing data points.

    Returns:
      entropy
    """
    answers = DecisionTree.generate_answers(examples)
    entropy = 0
    for answer in answers:
      frequency = answers[answer] / len(examples)
      entropy += -frequency*log(frequency, 2)
    return entropy
  
  @staticmethod
  def partition(examples:list, attribute_index:int, answer) -> list:
    """
    Generate a subset of examples where a provided attribute matches an answer.

    Args:
      examples - a list of tuples representing data points.
      answer_index - the attribute of an entity being searched.
      answer - searched value to match

    Returns:
      partition
    """
    partition = []
    for example in examples:
      if example[attribute_index] == answer:
        partition.append(example)
    return partition
  
  @staticmethod
  def best_question(examples:list, attributes:list, parent_entropy:float=None):
    """
    Find the best attribute to search for by comparing information gain of possible attributes.

    Args:
      examples - a list of tuples representing data points.
      attributes - a list of remaining attributes to act as next question. Last column is excluded (assumed actual answer)
      parent_entropy - Entropy of the previous question. If not provided, is calculated from given examples.

    Returns:
      [best_attribute, best_options] - Index of the chosen attribute and dictionary of with keys of possible answers 
      and values of partitioned data with attribute values of the key option.
    """
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
  def no_more(attributes:list) -> bool:
    """
    Check if there are any remaining attributes (excluding answer column).
    """
    for attr_index in range(len(attributes) - 1):
      if attributes[attr_index] is not None:
        return False
    return True

  def __init__(self, examples, attributes, max_depth=inf, threshold=0.00001, parent=None):
    self.value = None
    self.best_attr_index = None
    self.best_attribute = None
    self.children = None
    if len(examples) == 0:
      self.value = self.majority(parent)
      return
    entropy = DecisionTree.get_entropy(examples)
    if (
      entropy == 0 or entropy < threshold or 
      DecisionTree.no_more(attributes) or max_depth == 0
    ):
      # print(attributes,entropy)
      self.value = DecisionTree.majority(examples)
      return
    best_question_index, answers = DecisionTree.best_question(examples, attributes, entropy)
    self.best_attr_index = best_question_index
    self.best_attribute = attributes[best_question_index]
    new_attributes = attributes.copy()
    new_attributes[best_question_index] = None
    self.children = {}
    for answer, new_examples in answers.items():
      self.children[answer] = DecisionTree(new_examples,new_attributes,max_depth-1,threshold,self)

  def predict(self, example):
    """
    Predict an outcome based on training data. Example should have a length of completed data array - 1.
    Recurse down tree looking at attribute for each node until leaf is found.
    """
    if self.value is not None: return self.value
    option = example[self.best_attr_index]
    return self.children[option].predict(example) if self.children.get(option,None) is not None else None

  def __repr__(self):
    return (
      "DecisionTree(" + 
      (("value:" + str(self.value)) if self.value is not None else '') +
      (("\nquestion:" + self.best_attribute) if self.best_attribute is not None else '')+
      (("\nchildren{\n" + '\n'.join(map(lambda i: self.best_attribute + " " +repr(i[0]) + ': ' + repr(i[1]),self.children.items()))+"\n}")
       if self.children is not None else ''
      )+
      ")"
    )
  
  def __eq__(self, dt):
    if self.value is not None:
      return self.value == dt.get('value',None)
    else:
      if 'children' not in dt or len(self.children.keys()) != len(dt['children'].keys()):
        return False
      for key in self.children.keys():
        if dt['children'][key] != self.children[key]:
          return False
      return self.best_attribute == dt.get('best_question', None)
  
  def __ne__(self, dt):
    if self.value is not None:
      return self.value != dt.get('value',None)
    else:
      if 'children' not in dt or len(self.children.keys()) != len(dt['children'].keys()):
        return True
      for key in self.children.keys():
        if dt['children'][key] != self.children[key]:
          return True
      return self.best_attribute != dt.get('best_question', None)