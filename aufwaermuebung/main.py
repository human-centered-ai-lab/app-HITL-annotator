from utils import generate_binary_statements, generate_classifiers, majority_vote
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

n = 1000
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
    error_sep = []
    error_spp = []
    for index, classifier in enumerate(tests): 
        classifier.set_gt(majority_voted)
        sepe.append(classifier.get_estimated_sep())
        sppe.append(classifier.get_estimated_spp())
        sep.append(classifier.get_sep())
        spp.append(classifier.get_spp())
        error_sep.append(abs(classifier.get_estimated_sep() - classifier.get_sep()))
        error_spp.append(abs(classifier.get_estimated_spp() - classifier.get_spp()))
        if output:
            print('')
            print('classifier ', index)
            print('sep estimated: ', classifier.get_estimated_sep())
            print('spp estimated: ', classifier.get_estimated_spp())
            print('sep          : ', classifier.get_sep())
            print('spp          : ', classifier.get_spp())
    return sepe, sppe, sep, spp, error_sep, error_spp


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

def plot_twin(x, y1, y2, y3, y4, xaxislabel, y1axislabel, y2axislabel, y1label, y2label, y3label, y4label, title):
    #plt.figure(figsize=(10,10))
    fig, ax1 = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax1.set_ylim([0,1])
    ax1.set_ylabel(y1axislabel)
    ax1.set_xlabel(xaxislabel)
    ax1.plot(x, y1, label=y1label, color='#ff0088')
    ax1.plot(x, y2, label=y2label, color='#8800ff')
    plt.legend(loc = (0.8, 0.52))
    ax2 = ax1.twinx()
    ax2.set_ylim([min([*y3, *y4]), max([*y3, *y4])])
    ax2.set_ylabel(y2axislabel)
    ax2.plot(x, y3, label=y3label, color='#ff6600')
    ax2.plot(x, y4, label=y4label, color='#66ff00')
    fig.tight_layout()
    plt.legend(loc = (0.8, 0.45))
    plt.title(title)
    plt.show()  # display



def estimate_convergence(k, no_plot=False):
    sepes = []
    sppes = []
    spps = []
    seps = []
    error_seps = []
    error_spps = []
    X = []
    y = []
    y1 = []
    ye = []
    ye_1 = []
    first = True
    for x in tqdm(gt, leave=False):
        sepe, sppe, sep, spp, error_sep, error_spp = estimate_sep_spp([x], k, False, first)
        first = False
        sepes.append(sepe)
        spps.append(spp)
        seps.append(sep)
        sppes.append(sppe)
        error_seps.append(error_sep)
        error_spps.append(error_spp)
    for index, iteration in enumerate(zip(sepes, sppes, seps, spps, error_seps, error_spps)):
        # first_sepe = iteration[0][0]
        # first_sppe = iteration[1][0]
        # first_error_sep = iteration[4][0]
        # first_error_spp = iteration[5][0]
        # X.append(index)
        # y.append(first_sepe)
        # y1.append(first_sppe)
        # ye.append(first_error_sep)
        # ye_1.append(first_error_spp)
        avg_sepe = np.average(iteration[0])
        avg_sppe = np.average(iteration[1])
        avg_error_sep = np.average(iteration[4])
        avg_error_spp = np.average(iteration[5])
        X.append(index)
        y.append(avg_sepe)
        y1.append(avg_sppe)
        ye.append(avg_error_sep)
        ye_1.append(avg_error_spp)
    if not no_plot: plot_twin(
        X,
        y,
        y1,
        ye,
        ye_1,
        'Iteration',
        'sep and spp',
        'error',
        'estimated sep',
        'estimated spp',
        'error sep',
        'error spp',
        f'Estimation for sep and spp for {k} classifiers'
    )
    return X, y1, ye_1, y, ye
    #plot(X, y, y1, 'Iteration', 'sep', 'estimated spp', 'estimated spp', f'Estimation for sep and spp for {k} classifiers')
    #plot(X, ye, ye_1, 'Iteration', 'error', 'sep error', 'spp error', f'Error of estimation for sep and spp for {k} classifiers')

def plot_multi(x, ys, ylabels, yaxislabel, xaxislabel, title, legend, filename, upper=1.0):
    colors = ['#ff0088', '#8800ff', '#ff6600', '#66ff00', '#ff0000', '#00ff00', '#0000ff', '#95B8D1', '#EDAFB8', '#666A86']
    plt.figure(figsize=(6,6))
    plt.ylim([0, upper])
    for index, y in enumerate(zip(ys, ylabels)):
        plt.plot(x, y[0], label=y[1], color=colors[index])  # Plot the chart
    plt.xlabel(xaxislabel)
    plt.ylabel(yaxislabel)
    plt.legend(loc = legend)
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()  # display

ys_spp = []
ys_sep = []
labels = []
Xs = []
errors_spp = []
errors_sep = []

results = {}

combined = [2,3,6,10]

for x in [2,3,4,5,6,7,8,9,10]: 
    print(f'for {x} classifiers')
    random.seed(42)

    temp_spp = []
    temp_err_spp = []
    temp_sep = []
    temp_err_sep = []

    for _ in tqdm(range(1000)):
        X, y_spp, err_spp, y_sep, err_sep = estimate_convergence(x, True)
        temp_err_spp.append(err_spp)
        temp_spp.append(y_spp)
        temp_err_sep.append(err_sep)
        temp_sep.append(y_sep)
    
    temp_spp_avg = []
    temp_err_spp_avg = []
    temp_sep_avg = []
    temp_err_sep_avg = []

    for values in zip(*temp_spp):
        temp_spp_avg.append(np.average(values))
    
    for values in zip(*temp_err_spp):
        temp_err_spp_avg.append(np.average(values))

    for values in zip(*temp_sep):
        temp_sep_avg.append(np.average(values))
    
    for values in zip(*temp_err_sep):
        temp_err_sep_avg.append(np.average(values))
    
    plot_multi(X, [temp_err_spp_avg], [f'{x} classifiers'], 'Spp error', 'Iteration', f'Spp error for {x} classifiers', 'upper right', f'./imgout/individual_{x}_spp.png', 1.0)
    plot_multi(X, [temp_err_sep_avg], [f'{x} classifiers'], 'Sep error', 'Iteration', f'Sep error for {x} classifiers', 'upper right', f'./imgout/individual_{x}_sep.png', 1.0)
    #print(y)
    if x in combined:
        ys_spp.append(temp_spp_avg)
        errors_spp.append(temp_err_spp_avg)
        ys_sep.append(temp_sep_avg)
        errors_sep.append(temp_err_sep_avg)
        Xs = X
        labels.append(f'{x} classifiers')

plot_multi(Xs, errors_spp, labels, 'Spp error', 'Iteration', 'Spp error for 2,3,6,10 classifiers', 'upper right', './imgout/combined_spp.png', 1.0)
plot_multi(Xs, errors_sep, labels, 'Sep error', 'Iteration', 'Sep error for 2,3,6,10 classifiers', 'upper right', './imgout/combined_sep.png', 1.0)
#plot_multi(Xs, ys, labels, 'spp', 'Iteration', 'Spp for 3,4,5,6 classifiers', 'lower right')




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



