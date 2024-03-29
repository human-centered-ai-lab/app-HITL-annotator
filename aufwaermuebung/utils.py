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
        self.gt = []
        self.classified = []
        random.seed(42)

    
    def reset(self):
        self.gt = []
        self.classified = []
    
    def get_sep(self):
        return self.sep
    
    def get_spp(self):
        return self.spp
    
    def get_estimated_sep(self):
        count = 0
        correct = 0
        for x in zip(self.gt, self.classified): 
            if x[0] == 'N': 
                count = count + 1
                if x[1] == 'N': correct = correct + 1
        if count == 0: return 0
        return correct / count
    
    def get_estimated_spp(self):
        count = 0
        correct = 0
        for x in zip(self.gt, self.classified): 
            if x[0] == 'K': 
                count = count + 1
                if x[1] == 'K': correct = correct + 1
        if count == 0: return 0
        return correct / count

    def set_gt(self, value):
        for x in value: self.gt.append(x)
    
    def get_individual_classification(self, value):
        rand = random.uniform(0, 1.0)
        classified = None
        if value == 'N':
            if rand > self.sep: classified = 'K'
            else: classified = 'N'
        else:
            if rand > self.spp: classified = 'N'
            else: classified =  'K'
        self.classified.append(classified)
        return classified

    def get_test_classifications(self, gt):
        return list(map(self.get_individual_classification, gt))


def generate_classifiers(sep_low, sep_up, spp_low, spp_up, amount):
    #return [Classifier(0.75, 0.85) for _ in range(amount)]
    return [Classifier(random.uniform(sep_low, sep_up), random.uniform(spp_low, spp_up)) for _ in range(amount)]

def majority_vote(data):
    counter = Counter(data)
    return counter.most_common(1)[0][0]
