import sys, re
import numpy as np
import math

from model import *

###############################################################################

training_samples = []
vocabulary = set(['<UNK>'])
lines = ['are you still here ?', 'where are you ?', 'are you tired ?', 'i am tired .',
         'are you in england ?', 'are you in england?', 'were you in mexico?']
# line = sys.stdin.readline()
for line in lines:
    tokens = preprocess(line)
    for i in tokens:
        vocabulary.add(i)
    training_samples.append(tokens)
    # line = sys.stdin.readline()

word2idx = {k: v for v, k in enumerate(vocabulary)}
idx2word = {v: k for k, v in word2idx.items()}

x_train = []
y_train = []
for tokens in training_samples:
    for i in range(len(tokens) - 1): #!!!#
        x_train.append([word2idx[tokens[i]]]) #!!!#
        y_train.append([word2idx[tokens[i+1]]]) #!!!#

x_train = np.array(x_train)
y_train = np.array(y_train)

###############################################################################

BATCH_SIZE = 1
NUM_EPOCHS = 10

train_set = np.concatenate((x_train, y_train), axis=1)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

loss_function = nn.NLLLoss()
model = BigramNNmodel(len(vocabulary), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(NUM_EPOCHS):
    for i, data_tensor in enumerate(train_loader):
        context_tensor = data_tensor[:,0:1] #!!!#
        target_tensor = data_tensor[:,1] #!!!#

        model.zero_grad()

        log_probs = model(context_tensor)
        loss = loss_function(log_probs, target_tensor)

        loss.backward()
        optimiser.step()

    print('Epoch:', epoch, 'loss:', float(loss))

torch.save({'model': model.state_dict(), 'vocab': idx2word}, 'model.lm')

print('Model saved.')