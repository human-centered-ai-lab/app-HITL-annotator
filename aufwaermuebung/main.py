from utils import generate_binary_statements, generate_classifiers, majority_vote

n = 10000
m = 10
k = 10
sep_low = 0.7
sep_up = 0.9
spp_low = 0.8
spp_up = 0.98

gt = generate_binary_statements(0.25, 10000)

classifiers = generate_classifiers(sep_low, sep_up, spp_low, spp_up, m)

for index, classifier in enumerate(classifiers):
    results = []
    for _ in range(k):
        result = classifier.get_test_classifications(gt)
        results.append(result)
    majority_voted = []
    for result in zip(*results):
        vote_result = majority_vote(result)
        majority_voted.append(vote_result)
    false_positive = 0
    false_negative = 0
    for i in zip(gt, majority_voted):
        if i[1] == 'K' and i[0] == 'N': false_positive = false_positive + 1
        if i[1] == 'N' and i[0] == 'K': false_negative = false_negative + 1
    false_positive_fraction = false_positive / len(gt)
    false_negative_fraction = false_negative / len(gt)
    print('')
    print('---------------------------')
    print(f'classifier {index}')
    print(f'false_positive: {false_positive_fraction}')
    print(f'false_negative: {false_negative_fraction}')
    print(f'sep: {classifier.sep}')
    print(f'spp: {classifier.spp}')


for run in range(k):
    results = []
    seps = 0
    spps = 0
    for classifier in classifiers:
        result = classifier.get_test_classifications(gt)
        results.append(result)
        seps = seps + classifier.sep
        spps = spps + classifier.spp
    majority_voted = []
    for result in zip(*results):
        vote_result = majority_vote(result)
        majority_voted.append(vote_result)
    false_positive = 0
    false_negative = 0
    for i in zip(gt, majority_voted):
        if i[1] == 'K' and i[0] == 'N': false_positive = false_positive + 1
        if i[1] == 'N' and i[0] == 'K': false_negative = false_negative + 1
    false_positive_fraction = false_positive / len(gt)
    false_negative_fraction = false_negative / len(gt)
    avg_sep = seps / len(classifiers)
    avg_spp = spps / len(classifiers)
    print('')
    print('---------------------------') 
    print(f'run {run}')
    print(f'false_positive: {false_positive_fraction}')
    print(f'false_negative: {false_negative_fraction}')
    print(f'avg_sep: {avg_sep}')
    print(f'avg_spp: {avg_spp}')