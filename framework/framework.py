from functools import reduce
import random
import math
from util import normalize_probabilities, get_primitive_classification, majority_vote, get_available_annotators, get_label_from_probabilities, calculate_needed_annotators, get_annotator_availabilities
import numpy as np
import concurrent.futures

class HITLAnnotator:
    # annotators: instances of annotator class
    # a: fraction of the dataset that is used for initial classification
    # call_annotator: function that gets called with an id of an annotator and a sample of the dataset when annotation of a specific annotator is requested
    # prob_high: upper class probability threshold for numan annotation
    # prob_low: lower class probability threshold for numan annotation
    # mode: 'binary' for binary classification and 'multi' for multi-class classification
    def __init__(self, annotators, a, call_annotator, prob_high, prob_low, mode, classifier, training_data, possible_labels, min_annotators, max_annotators, test_data_X, test_data_y, images) -> None:
        self.annotator = annotators
        self.a = a
        self.prob_high = prob_high
        self.prob_low = prob_low
        self.call_annotator = call_annotator
        self.classifier = classifier
        self.annotators = annotators
        self.mode = mode
        self.images = images
        self.test_data_X = test_data_X
        self.test_data_y = test_data_y
        self.training_data = training_data
        self.possible_labels = possible_labels
        self.min_annotators = min_annotators
        self.max_annotators = max_annotators
        self.annotations = {}
    
    # perform the main task of the framework
    # ground_truth: the ground truth only needed if primitve debug annotators are used
    def train_classifier_with_human_in_the_loop(self, ground_truth=None):
        classifier = self.classifier
        data = self.training_data
        data_with_id = []
        zipped_data_images = zip(data, self.images) if self.images is not None else data
        for index, x in enumerate(zipped_data_images):
            datapoint = x[0] if self.images is not None else x
            image = x[1] if self.images is not None else None
            data_with_id.append([index, datapoint, image])
        a = self.a
        prob_high = self.prob_high
        prob_low = self.prob_low
        annotators = self.annotators
        for annotator in annotators:
            if annotator.mode == 'primitive' and ground_truth is None:
                raise "Ground Truth needed if a annotator is primitive" 
        intial_amount = math.floor(len(data) * a)
        X_train_initial = data_with_id[:intial_amount]
        ground_truth_initial = ground_truth[:intial_amount] if ground_truth is not None else None
        probs_initial = [None] * len(X_train_initial)
        y_initial = [None] * len(X_train_initial)
        y_train_initial = self.consult_classifiers_with_majority_vote(X_train_initial, probs_initial, y_initial, ground_truth_initial) if ground_truth is not None else self.consult_classifiers_with_majority_vote(X_train_initial, probs_initial, y_initial)
        X_train_iteration = []
        y_train_iteration = []
        #gt_train_iteration = []
        current_classifier = classifier
        itercount = 0
        while True:
            itercount = itercount + 1
            X_current = X_train_iteration if len(X_train_iteration) > 0 else X_train_initial
            y_current = y_train_iteration if len(y_train_iteration) > 0 else y_train_initial
            #gt_current = gt_train_iteration if len(gt_train_iteration) > 0 else ground_truth_initial
            #print(itercount)
            #print(X_current)
            current_classifier.fit(list(map(lambda x: x[1], X_current)), y_current)
            y_pred = current_classifier.predict_proba(data)
            y_pred_labels, y_pred_probabilities = get_label_from_probabilities(y_pred.tolist(), self.possible_labels)
            y_pred_probabilities_norm = normalize_probabilities(y_pred_probabilities)
            to_annotate_X = []
            to_annotate_probs = []
            to_annotate_y = []
            to_annotate_gt = []
            auto_annotated_X = []
            auto_annotated_y = []
            zipped = zip(data_with_id, y_pred_probabilities_norm, y_pred_labels, ground_truth) if ground_truth is not None else zip(data_with_id, y_pred_probabilities_norm, y_pred_labels)
            for sample in zipped:
                prob = sample[1]
                item = sample[0]
                label = sample[2]
                gt_y = sample[3] if ground_truth is not None else None
                if prob < prob_high and prob >= prob_low:
                    to_annotate_X.append(item)
                    to_annotate_y.append(label)
                    to_annotate_probs.append(prob)
                    if gt_y is not None: to_annotate_gt.append(gt_y)
                if prob >= prob_high:
                    #datapoint = item[1]
                    id = item[0]
                    #if not id in self.annotations:
                    #    self.annotations[id] = label
                    auto_annotated_X.append(item)
                    auto_annotated_y.append(label)
            do_break = False
            try:
                y_annotated = self.consult_classifiers_with_majority_vote(to_annotate_X, to_annotate_probs, to_annotate_y, to_annotate_gt) if ground_truth is not None else self.consult_classifiers_with_majority_vote(to_annotate_X, to_annotate_probs, to_annotate_y)
                X_train_iteration = [*X_train_initial, *auto_annotated_X, *to_annotate_X]
                y_train_iteration = [*y_train_initial, *auto_annotated_y, *y_annotated]
            except:
                print('')
                print('---------------------------------')
                print('ABORTING')
                do_break = True
            #test the currenct classifier
            X_test = self.test_data_X
            y_test = self.test_data_y
            y_pred = current_classifier.predict_proba(X_test)
            #y_pred = sgd_clf.predict(X_test_prepared)
            y_pred_labels, y_pred_probabilities = get_label_from_probabilities(y_pred.tolist(), self.possible_labels)
            percentage = 100*np.sum(y_pred_labels == y_test)/len(y_test)
            print('')
            print('---------------------------------')
            print('ANNOTATION CASE')
            print('Iteration: ', itercount)
            print('Percentage correct: ', percentage)
            print('Consultations used so far: ', np.sum([x.get_classification_count() for x in annotators]))
            if itercount >= 10 or do_break: break
        return current_classifier
    
    # contact the annotators and retrieve their annotation
    # X: the data to be annotated
    # prob: the probabilities returned by the machine classifier to determine the number of used classifiers
    # y: the labels returned by the machine classifier
    # gt: the ground truth (only needed when using primitive classifiers)
    def consult_classifiers_with_majority_vote(self, X, prob, y, gt=None):
        max_annotators = self.max_annotators
        min_annotators = self.min_annotators
        call_function = self.call_annotator
        possible_classes = self.possible_labels
        low = self.prob_low
        high = self.prob_high
        needed_annotators = []
        for probability in prob:
            needed_annotators.append(calculate_needed_annotators(min_annotators, max_annotators, low, high, probability))
        zipped = zip(X, prob, needed_annotators, y, gt) if gt is not None else zip(X, prob, needed_annotators, y)
        sample_results = []
        for sample in zipped:
            availabilities = get_annotator_availabilities(self.annotators)
            sample_X = sample[0]
            sample_X_sample = sample_X[2]
            sample_X_id = sample_X[0]
            sample_prob = sample[1]
            sample_needed_annotators = sample[2]
            sample_y = sample[3]
            sample_gt = sample[4] if gt is not None else None
            if sample_X_id in self.annotations:
                sample_results.append(self.annotations[sample_X_id])
                continue
            available_annotators = get_available_annotators(availabilities, self.annotators)
            if len(available_annotators) < sample_needed_annotators: raise "NOT_ENOUGH_ANNOTATORS_AVAILABLE"
            selected_annotators = available_annotators[:sample_needed_annotators]
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(annotator_thread, annotator, call_function, sample_X_sample, sample_prob, sample_y, possible_classes, sample_gt) for annotator in selected_annotators]
                for future in futures:
                    result = future.result()
                    results.append(result)
            #print(results)
            result = majority_vote(map(lambda x: x[1], results))
            self.annotations[sample_X_id] = result
            sample_results.append(result)
            for x in zip(results, selected_annotators):
                r = x[0]
                cid = r[0]
                lab = r[1]
                ann = x[1]
                if lab == result:
                    ann.update_classification_result(cid, True)
        return sample_results
                


def annotator_thread(annotator, call_function, sample_X, sample_prob, sample_y, possible_classes, sample_gt=None):
    result = annotator.do_classification(call_function, sample_X, sample_prob, sample_y, possible_classes, sample_gt)
    return result

    
class Annotator:
    # id: id of the annotator, must be unique
    # mode: 'function' or 'primitive'
    # accuracy_low: lower boundary for randomly generated accuracy of primitive classifier
    # accuracy_high: upper boundary for randomly generated accuracy of primitive classifier
    # random_state: to manually adjust random state
    def __init__(self, id, mode, limit=None, accuracy_low=None, accuracy_high=None, random_state=None):
        self.id = id
        if mode not in ['function', 'plugin', 'primitive']: raise "Not a valid mode"
        self.mode = mode
        if self.mode != 'primitive' and (accuracy_high is not None or accuracy_low is not None or random_state is not None) : raise "accuracy_high, accuracy_low and random_state only supported in 'primitive' mode"
        if random_state: self.random_state = random_state
        if accuracy_low: self.accuracy_low = accuracy_low
        if accuracy_high: self.accuracy_high = accuracy_high
        self.classifications = []
        self.classification_id = 0
        self.limit = limit
        if mode == "primitive":
            accuracy = random.uniform(accuracy_low, accuracy_high)
            self.accuracy = accuracy
    
    # call_function the function to be called for external classification, not applied when mode = 'primitive' or 'plugin'
    # sample: the sample to be classified
    # returns the result of the human classification and the id of the classification
    def do_classification(self, call_function, sample_X, sample_prob, sample_y, possible_classes, sample_gt=None):
        if self.limit and len(self.classifications) >= self.limit: raise "ANNOTATOR_LIMIT"
        classification_id = self.classification_id
        self.classification_id = self.classification_id + 1
        self.classifications.append({'correct': False, 'sample': sample_X, 'sample_prob' : sample_prob, 'sample_y': sample_y, 'id': classification_id})
        result = None
        if (self.mode == 'function'): result = call_function(self.id, sample_X, sample_y, sample_prob, possible_classes) 
        if (self.mode == 'primitive'): result = get_primitive_classification(possible_classes, sample_gt, self.accuracy)
        return [classification_id, result]
    
    def get_id(self):
        return self.id
    
    # returns classification count of this annotator
    def get_classification_count(self):
        return len(self.classifications)
    
    # returns classification limit of this annotator
    def get_limit(self):
        return self.limit
    
    # id the id of the c lassification to be updated
    # result the result that should be applied to the update
    # returns true of update was successful, False otherwise
    def update_classification_result(self, id, result):
        updated = False
        for classification in self.classifications:
            if classification['id'] == id:
                classification['correct'] = result
                updated = True
        return updated
    
    # returns the performance of this annotator based on historical classifications, a value between 0 and 1
    def get_performance(self):
        classified = len(self.classifications)
        if classified == 0: return classified
        correct = reduce(lambda acc, x: acc + 1 if x['correct'] else acc, self.classifications, 0)
        return correct / classified
    
    # return the id of the annotator
    def get_id(self):
        return self.id
