import math
import random
from collections import Counter

def generate_binary_statements(relation, amount):
    yes = ['K'] * math.floor(amount * relation)
    no = ['N'] * (amount - len(yes))
    statements = yes + no
    random.shuffle(statements)
    return statements

class Classifier:
    def __init__(self, sep, spp):
        self.sep = sep
        self.spp = spp
    
    def get_individual_classification(self, value):
        rand = random.uniform(0, 1.0)
        if value == 'N':
            if rand > self.sep: return 'K'
            else: return 'N'
        else:
            if rand > self.spp: return 'N'
            else: return 'K'

    def get_test_classifications(self, gt):
        return list(map(self.get_individual_classification, gt))


def generate_classifiers(sep_low, sep_up, spp_low, spp_up, amount):
    return [Classifier(random.uniform(sep_low, sep_up), random.uniform(spp_low, spp_up))] * amount

def majority_vote(data):
    counter = Counter(data)
    return counter.most_common(1)[0][0]
