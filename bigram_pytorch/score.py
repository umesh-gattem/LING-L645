import sys, re
import numpy as np
import math

from model import *

###############################################################################

blob = torch.load('model.lm')
idx2word = blob['vocab']
word2idx = {k: v for v, k in idx2word.items()}
vocabulary = set(idx2word.values())

model = BigramNNmodel(len(vocabulary), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM)
model.load_state_dict(blob['model'])

###############################################################################

BATCH_SIZE = 1

# line = sys.stdin.readline()
test_lines = ["where are you?",
              "were you in england?",
              "are you in mexico?",
              "i am in mexico.",
              "are you still in mexico?"]
for line in test_lines:
    tokens = preprocess(line)

    x_test = []
    y_test = []
    for i in range(len(tokens) - 1):  # !!!#
        x_test.append([word2idx[tokens[i]]])  # !!!#
        y_test.append([word2idx[tokens[i + 1]]])  # !!!#

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    test_set = np.concatenate((x_test, y_test), axis=1)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    total_prob = 1.0
    for i, data_tensor in enumerate(test_loader):
        context_tensor = data_tensor[:, 0:1]  # !!!#
        #target_tensor = data_tensor[:, 1]  # !!!#
        log_probs = model(context_tensor)
        probs = torch.exp(log_probs)
        #predicted_label = int(torch.argmax(probs, dim=1)[0])

        true_label = y_test[i][0]
        true_word = idx2word[true_label]

        prob_true = float(probs[0][true_label])
        total_prob *= prob_true

    print('%.6f\t%.6f\t' % (total_prob, math.log(total_prob)), tokens)

    # line = sys.stdin.readline()