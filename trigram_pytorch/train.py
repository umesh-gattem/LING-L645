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
z_train = []
for tokens in training_samples:
    for i in range(len(tokens) - 2): #!!!#
        x_train.append([word2idx[tokens[i]], word2idx[tokens[i+1]]]) #!!!#
        y_train.append([word2idx[tokens[i+2]]]) #!!!#
        # z_train.append([word2idx[tokens[i+2]]]) #!!!#

x_train = np.array(x_train)
y_train = np.array(y_train)
# z_train = np.array(z_train)

###############################################################################

BATCH_SIZE = 1
NUM_EPOCHS = 10

train_set = np.concatenate((x_train, y_train), axis=1)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

loss_function = nn.NLLLoss()
model = TrigramNNmodel(len(vocabulary), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(NUM_EPOCHS):
    for i, data_tensor in enumerate(train_loader):
        context_tensor = data_tensor[:,0:2] #!!!#
        target_tensor = data_tensor[:,2] #!!!#

        model.zero_grad()

        log_probs = model(context_tensor)
        loss = loss_function(log_probs, target_tensor)

        loss.backward()
        optimiser.step()

    print('Epoch:', epoch, 'loss:', float(loss))

torch.save({'model': model.state_dict(), 'vocab': idx2word}, '../trigram_pytorch/model.lm')

print('Model saved.')