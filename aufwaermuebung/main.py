from utils import generate_binary_statements, generate_classifiers, majority_vote
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

n = 2500
m = 10
k = 10
sep_low = 0.7
sep_up = 0.9
spp_low = 0.8
spp_up = 0.98

classifiers = generate_classifiers(sep_low, sep_up, spp_low, spp_up, m)

def estimate_sep_spp(gt, k, output, reset=True):
    tests = classifiers[:k]
    if reset: 
        for classifier in tests: classifier.reset()
    results = [classifier.get_test_classifications(gt) for classifier in tests]
    majority_voted = []
    for result in zip(*results):
        vote_result = majority_vote(result)
        majority_voted.append(vote_result)
    sepe = []
    sppe = []
    spp = []
    sep = []
    for index, classifier in enumerate(tests): 
        classifier.set_gt(majority_voted)
        sepe.append(classifier.get_estimated_sep())
        sppe.append(classifier.get_estimated_spp())
        sep.append(classifier.get_sep())
        spp.append(classifier.get_spp())
        if output:
            print('')
            print('classifier ', index)
            print('sep estimated: ', classifier.get_estimated_sep())
            print('spp estimated: ', classifier.get_estimated_spp())
            print('sep          : ', classifier.get_sep())
            print('spp          : ', classifier.get_spp())
    return sepe, sppe, sep, spp


gt = generate_binary_statements(0.25, n)

def plot(x,y,y1,xaxislabel, yaxislabel, ylabel, y1label, title):
    plt.figure(figsize=(10,10))
    plt.ylim([0, 1])
    plt.plot(x, y, label=ylabel)  # Plot the chart
    plt.xlabel(xaxislabel)
    plt.ylabel(yaxislabel)
    plt.plot(x, y1, label=y1label)  # Plot the chart
    plt.legend(loc = 'lower right')
    plt.title(title)
    plt.show()  # display

def estimate_convergence(k):
    sepes = []
    sppes = []
    spps = []
    seps = []
    X = []
    y = []
    y1 = []
    first = True
    for x in tqdm(gt):
        sepe, sppe, sep, spp = estimate_sep_spp([x], k, False, first)
        first = False
        sepes.append(sepe)
        spps.append(spp)
        seps.append(sep)
        sppes.append(sppe)
    for index, iteration in enumerate(zip(sepes, sppes, seps, spps)):
        avg_sepe = iteration[0][0]
        avg_sppe = iteration[1][0]
        X.append(index)
        y.append(avg_sepe)
        y1.append(avg_sppe)
    plot(X, y, y1, 'Iteration', 'sep', 'estimated spp', 'estimated spp', f'Estimation for sep and spp for {k} classifiers')

for x in range(3, k+1): estimate_convergence(x)



x = []
y_sep = []
line_sep = []
y_spp = []
line_spp = []



# for i in tqdm(range(len(gt))):
#     x.append(i)
#     gti = gt[:i]
#     sepe, sppe, sep, spp = estimate_sep_spp(gti, k, False)
#     for values in zip(sepe, sppe, sep, spp):
#         y_sep.append(values[0])
#         line_sep.append(values[2])
#         y_spp.append(values[1])
#         line_spp.append(values[3])
#         break
# plot(x, y_sep, line_sep, 'amount statements', 'sep', 'Estimated Sep', 'True Sep', 'Sep Estimation with 10 Classifiers')
# plot(x, y_spp, line_spp, 'amount statements', 'spp', 'Estimated Spp', 'True Spp', 'Spp Estimation with 10 Classifiers')

x_1 = []
y_sep_1 = []
line_sep_1 = []
y_spp_1 = []
line_spp_1 = []

""" for i in tqdm(range(1, 400)):
    x_1.append(i)
    sepe, sppe, sep, spp = estimate_sep_spp(gt, i, False)
    for values in zip(sepe, sppe, sep, spp):
        y_sep_1.append(values[0])
        line_sep_1.append(values[3])
        y_spp_1.append(values[1])
        line_spp_1.append(values[2])
        break
plot(x_1, y_sep_1, line_sep_1, 'amount classifiers', 'sep', 'Estimated Sep', 'True Sep', 'Sep Estimation with 10000 Statements')
plot(x_1, y_spp_1, line_spp_1, 'amount classifiers', 'spp', 'Estimated Spp', 'True Spp', 'Spp Estimation with 10000 Statements') """



