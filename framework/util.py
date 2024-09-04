import numpy as np
import random

def get_label_from_probabilities(probabilities, labels):
        result = []
        label_probabilities = []
        for x in probabilities:
            #print(x)
            max = np.max(x)
            index = x.index(max)
            result.append(labels[index])
            label_probabilities.append(max)
            #print(max)
            #print(x)
        return result, label_probabilities
    

def normalize_probabilities(probabilities):
    max = np.max(probabilities)
    normalized = [(x / max) for x in probabilities]
    return normalized

def get_primitive_classification(classes, label, accuracy):
    rand = random.random()
    if rand <= accuracy:
        return label
    else:
        choice = []
        for x in classes:
            if x != label:
                choice.append(x)
        result = random.choice(choice)
        return result

def calculate_needed_annotators(mina, maxa, low, high, prob):
    if prob == None: return mina
    cut_prob = prob - low
    cut_high = high - low
    prob_normalized = cut_prob * ( 1 / cut_high )
    needed = (prob_normalized * maxa) + mina
    return round(needed)

def get_annotator_availabilities(annotators):
    availabilities = {}
    for annotator in annotators:
        classifications = annotator.get_classification_count()
        limit = annotator.get_limit()
        id = annotator.get_id()
        availabilities[id] = limit - classifications if limit != None else None
    return availabilities

def get_available_annotators(availabilities, annotators):
    unused_annotators = []
    rated_annotators = []
    for annotator in annotators:
        classification_count = annotator.get_classification_count()
        id = annotator.get_id()
        available = availabilities[id]
        if classification_count < 10 and (available is None or available > 0):
            unused_annotators.append(annotator)
        if classification_count >= 10 and (available is None or available > 0):
            rated_annotators.append(annotator)
    random.shuffle(unused_annotators)
    sorted_rated_annotators = sorted(rated_annotators, key=lambda annotator: annotator.get_performance(), reverse=True)
    return [*unused_annotators, *sorted_rated_annotators]

def majority_vote(labels):
    counts = {}
    for label in labels:
        if label in counts:
            counts[label] = counts[label] + 1
        else:
            counts[label] = 1
    keys = counts.keys()
    sorted_keys = sorted(keys, key=lambda key: counts[key])
    return sorted_keys[0]

     