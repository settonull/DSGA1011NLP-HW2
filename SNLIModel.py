'''
This script is used to performa training runs in the HPC/prince environment for 
faster training.  All code other than __main__ taken from hw2.ipynb.
Values from previous runs are used to chose subsequent ones.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import csv
import pickle
import math
import sys
import os

CUDA=True

def index_it(sentence):
    s_i = []
    for w in sentence.split():
        if w not in words:
            s_i.append(OOV_IDX)
        else:
            s_i.append(words[w])
    return s_i


# load our data, it is in a tab delimited row, with permise, hypothesis, label
# our sentences are already pre-procesed, we just need to split
def loadsnli(filename):
    data = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip the header
        for line in reader:
            prem = index_it(line[0])
            hypo = index_it(line[1])

            if (line[2] == 'neutral'):
                target = 0
            elif (line[2] == 'entailment'):
                target = 1
            elif (line[2] == 'contradiction'):
                target = 2
            else:
                target = 3  # shouldn't ever happen
            data.append((prem, hypo, target))
    return np.array(data)



# define our dataset for our investigations


class SNLIDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, prem_list, hypo_list, target_list, max_sentenance_length):
        """
        @param data_list: list of newsgroup tokens
        @param target_list: list of newsgroup targets

        """
        self.prem_list = prem_list
        self.hypo_list = hypo_list
        self.target_list = target_list
        self.max_sentenance_length = max_sentenance_length
        assert (len(self.prem_list) == len(self.target_list))
        assert (len(self.hypo_list) == len(self.target_list))

    def __len__(self):
        return len(self.prem_list)  # they should all be the same

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """

        prem_token_idx = self.prem_list[key][:self.max_sentenance_length]
        prem_token_len = len(prem_token_idx)

        prem_token_idx = np.pad(np.array(prem_token_idx),
                                pad_width=((0, self.max_sentenance_length - prem_token_len)),
                                mode="constant", constant_values=0)

        hypo_token_idx = self.hypo_list[key][:self.max_sentenance_length]
        hypo_token_len = len(hypo_token_idx)

        hypo_token_idx = np.pad(np.array(hypo_token_idx),
                                pad_width=((0, self.max_sentenance_length - hypo_token_len)),
                                mode="constant", constant_values=0)

        label = self.target_list[key]
        # return [torch.Tensor(prem_token_idx).cuda(),
        #        torch.Tensor(prem_token_len).cuda(),
        #        torch.Tensor(hypo_token_idx).cuda(),
        #        torch.Tensor(hypo_token_len).cuda(),
        #        torch.Tensor(label).cuda()]
        return [prem_token_idx, prem_token_len, hypo_token_idx, hypo_token_len, label]





class CNN(nn.Module):
    def __init__(self, emb, hidden_size, kernel, num_classes):
        super(CNN, self).__init__()

        self.hidden_size = hidden_size

        pad = math.floor(kernel / 2)

        # use our previous loaded embedding matrix
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_IDX)
        self.embedding.load_state_dict({'weight': emb})
        self.embedding.weight.requires_grad = False

        self.conv1 = nn.Conv1d(embedding_dim, hidden_size, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel, padding=pad)

        self.linear1 = nn.Linear(hidden_size*2, hidden_size)
        #self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x1, lengths1, x2, lengths2):
        batch_size, seq_len = x1.size()  # assumes seq_len is same for both, which our loader does

        embed1 = self.embedding(x1)
        hidden1 = self.conv1(embed1.transpose(1, 2)).transpose(1, 2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, seq_len, hidden1.size(-1))

        hidden1 = self.conv2(hidden1.transpose(1, 2)).transpose(1, 2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, seq_len, hidden1.size(-1))

        # now maxppool, need to move things around to pool the right dimension
        hidden1 = hidden1.transpose(1, 2)
        hidden1 = F.max_pool1d(hidden1, kernel_size=hidden1.size()[2])
        hidden1 = hidden1.transpose(1, 2)
        hidden1 = hidden1.view(batch_size, self.hidden_size)

        # TODO: make this a max pool instead
        # hidden1 = torch.sum(hidden1, dim=1)

        embed2 = self.embedding(x2)
        hidden2 = self.conv1(embed2.transpose(1, 2)).transpose(1, 2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, seq_len, hidden2.size(-1))

        hidden2 = self.conv2(hidden2.transpose(1, 2)).transpose(1, 2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, seq_len, hidden2.size(-1))

        # now maxppool, need to move things around to pool the right dimension
        hidden2 = hidden2.transpose(1, 2)
        hidden2 = F.max_pool1d(hidden2, kernel_size=hidden2.size()[2])
        hidden2 = hidden2.transpose(1, 2)
        hidden2 = hidden2.view(batch_size, self.hidden_size)

        # TODO: make this a max pool instead
        # hidden2 = torch.sum(hidden2, dim=1)

        # print("hidden1:", hidden1.size())
        # print("hidden2:", hidden2.size())

        full = torch.cat((hidden1, hidden2), dim=1)
        #full = torch.mul(hidden1, hidden2)
        # print("full:", full.size())

        full = self.linear1(full)
        full = F.relu(full)
        logits = self.linear2(full)
        # print("logits:", logits.size())

        return logits


# For the RNN, a single-layer, bi-directional GRU will suffice.
# We can take the last hidden state as the encoder output. (In the case of bi-directional, the last of each direction, although PyTorch takes care of this.)


class RNN(nn.Module):
    def __init__(self, emb, hidden_size, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1

        # use our previous loaded embedding matrix
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_IDX)
        self.embedding.load_state_dict({'weight': emb})
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embedding_dim, hidden_size, 1, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x1, lengths1, x2, lengths2):
        batch_size, seq_len = x1.size()  # assumes seq_len is same for both, which our loader does

        h1 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size)
        if CUDA:
            h1 = h1.cuda()

        embed1 = self.embedding(x1)

        # pack padded sequence
        # embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths1.numpy(), batch_first=True)
        # fprop though RNN
        rnn_out, h1 = self.gru(embed1, h1)

        # TODO: Is this really how we want to combine them?
        h1 = torch.sum(h1, dim=0)

        h2 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size)
        if CUDA:
            h2 = h2.cuda()

        embed2 = self.embedding(x2)

        # pack padded sequence
        # embed2 = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths2.numpy(), batch_first=True)
        # fprop though RNN
        rnn_out, h2 = self.gru(embed2, h2)

        # TODO: Is this really how we want to combine them?
        h2 = torch.sum(h2, dim=0)

        # print("hidden1:", hidden1.size())
        # print("hidden2:", hidden2.size())

        full = torch.cat((h1, h2), dim=1)
        # full = torch.mul(hidden1, hidden2)
        # print("full:", full.size())

        full = self.linear1(full)
        full = F.relu(full)
        logits = self.linear2(full)
        # print("logits:", logits.size())

        return logits


#from the pytorch website forums - https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
#we iterate over all the model paramaters that have a gradient (thus not our untrained embedding) and sum them
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data1, lengths1, data2, lengths2, labels in loader:
        if (CUDA):
            data1, lengths1, data2, lengths2, labels = data1.cuda(), lengths1.cuda(), data2.cuda(), lengths2.cuda(), labels.cuda()
        outputs = F.softmax(model(data1, lengths1, data2, lengths2), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


# train our model, this allows us to grid search some paramaters.  For now loss and optimizer are kept the same
def train_model(model, t_loader, v_loader, learning_rate=1e-3, num_epochs=10, verbose='', weight_decay=0):
    # Train the model
    total_step = len(t_loader)

    # report progress within an epoch
    report_progress = False
    report_freq = -1

    if (verbose == 'v'):
        report_progress = True
        report_freq = total_step + 1

    if (verbose == 'vv'):
        report_progress = True
        report_freq = total_step / 4

    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    accs = []

    # time_start = time.time()
    max_val = 0
    for epoch in range(num_epochs):
        for i, (data1, lengths1, data2, lengths2, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            if (CUDA):
            	data1, lengths1, data2, lengths2, labels = data1.cuda(), lengths1.cuda(), data2.cuda(), lengths2.cuda(), labels.cuda()
            outputs = model(data1, lengths1, data2, lengths2)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            # validate every 100 iterations
            if report_progress and i > 0 and i % report_freq == 0:
                # validate
                val_acc = test_model(v_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, val_acc))

        train_acc = test_model(t_loader, model)
        val_acc = test_model(v_loader, model)
        accs.append((train_acc, val_acc))

        if report_progress:
            print('Epoch: [{}/{}], Train Acc, {} Validation Acc: {}'.format(
                epoch + 1, num_epochs, train_acc, val_acc), flush=True)

        if (val_acc > max_val):
            max_val = val_acc

        if (val_acc + (0.1 * max_val) < max_val):
            break;
    return accs

def save_data(run_name, res_run):

    with open(os.path.join(data_dir,(run_name + '.pkl')), 'wb') as f:
        pickle.dump(res_run, f)

if __name__ == '__main__':

    '''
    First we'll load our pre-trained embedding.
    We'll prefix our embeddings with a PAD and OOV
    '''

    if len(sys.argv) < 3:
        print("Usage: SNLIModel <RNN1, RNN2, CNN1, CNN2> num_epochs datadir")

    model_type = sys.argv[1]
    num_epochs = int(sys.argv[2])
    data_dir = sys.argv[3]

    print("Starting", sys.argv[0], "with", sys.argv[1], flush=True)

    #embedding_home = '/home/gandalf/NYUCDS/DS-GA1011/lab5/'
    #if not os.path.isdir(embedding_home):
    #    embedding_home = 'C:\\development\\NYUCDS\\DSGA1011\\'

    words_to_load = 50000
    embedding_source = 'wiki-news-300d-1M.vec'
    embedding_dim = 300
    num_embeddings = words_to_load + 2

    words = {}
    idx2words = {}
    ordered_words = []
    loaded_embeddings = np.zeros((words_to_load + 2, embedding_dim))

    # prefix with PAD and OOV
    PAD_IDX = 0
    OOV_IDX = 1
    words['<PAD>'] = PAD_IDX
    idx2words[PAD_IDX] = '<PAD>'
    words['<OOV>'] = OOV_IDX
    idx2words[OOV_IDX] = '<OOV>'
    loaded_embeddings[OOV_IDX, :] = np.random.normal(size=embedding_dim)

    print("loading embeddings...", end='', flush=True)
    #with open(embedding_home + embedding_source, 'r', encoding='utf-8') as f:
    with open(data_dir + embedding_source, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= words_to_load:
                break
            s = line.split()
            loaded_embeddings[i + 2, :] = np.asarray(s[1:])
            words[s[0]] = i + 2
            idx2words[i + 2] = s[0]
            ordered_words.append(s[0])

    loaded_embeddings = torch.Tensor(loaded_embeddings)
    print(len(loaded_embeddings), "loaded", flush=True)

    '''
    Let's load our data now
    '''

    #data_src = '/home/gandalf/Dropbox/NYU CDS/DS-GA 1011 NLP/HW2/hw2_data/'
    #if not os.path.isdir(data_src):
    #    data_src = 'E:\\cloudstation\\Dropbox\\NYU CDS\\DS-GA 1011 NLP\\HW2\\hw2_data\\'

    train_snli_name = 'snli_train.tsv'
    val_snli_name = 'snli_val.tsv'

    train_data = loadsnli(data_dir + train_snli_name)
    val_data = loadsnli(data_dir + val_snli_name)

    BATCH_SIZE = 32
    MAX_SENTENANCE_LENGTH = 35
    train_dataset = SNLIDataset(train_data[:, 0], train_data[:, 1], train_data[:, 2], MAX_SENTENANCE_LENGTH)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = SNLIDataset(val_data[:, 0], val_data[:, 1], val_data[:, 2], MAX_SENTENANCE_LENGTH)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)


    if (model_type == 'CNN1'):
        model_res_CNN = {}
        model_state_CNN = {}

        kernel=3
        for hidden_size in [50,100,200,500,1000]:
            model = CNN(emb=loaded_embeddings, hidden_size=hidden_size, kernel=kernel, num_classes=3)
            if CUDA:
                model.cuda()
            print("Training",model_type, "for", num_epochs, "epochs with hidden_size=",   hidden_size,
                  "and kernel=", kernel, "and", count_parameters(model), "paramaters", flush=True)
            res = train_model(model, train_loader, val_loader, learning_rate=1e-3, num_epochs = num_epochs, verbose='v')
            model_res_CNN['CNN-'+str(hidden_size)] = res
            model_state_CNN['CNN-'+str(hidden_size)] = model.state_dict()

        save_data("CNN1_results", model_res_CNN)
        save_data("CNN1_models", model_state_CNN)


    if (model_type == 'CNN2'):
        model_res_CNN2 = {}
        model_state_CNN2 = {}

	#best from previous run
        hidden_size = 200
        for kernel in [1,3,5,7]:
            model = CNN(emb=loaded_embeddings, hidden_size=hidden_size, kernel=kernel, num_classes=3)
            if CUDA:
                model.cuda()
            print("Training", model_type, "for", num_epochs, "epochs with hidden_size=", hidden_size, "and kernel=",
                  kernel, "and", count_parameters(model), "paramaters", flush=True)
            res = train_model(model, train_loader, val_loader, learning_rate=1e-3, num_epochs = num_epochs, verbose='v')
            model_res_CNN2['CNN-'+str(kernel)] = res
            model_state_CNN2['CNN-'+str(kernel)] = model.state_dict()

        save_data("CNN2_results", model_res_CNN2)
        save_data("CNN2_models", model_state_CNN2)


    if (model_type == 'RNN1'):
        model_res = {}
        model_state = {}

        for hidden_size in [50,100,200,500,1000]:
            model = RNN(emb=loaded_embeddings, hidden_size=hidden_size, num_classes=3)
            if CUDA:
                model.cuda()
            print("Training", model_type, "for", num_epochs, "epochs with hidden_size=", hidden_size,
                  "and", count_parameters(model), "paramaters", flush=True)
            res = train_model(model, train_loader, val_loader, learning_rate=1e-3, num_epochs = num_epochs, verbose='v')
            model_res['RNN-'+str(hidden_size)] = res
            model_state['RNN-'+str(hidden_size)] = model

        save_data("RNN1_results", model_res)
        save_data("RNN1_models", model_state)

    if (model_type == 'RNN2'):
        model_res = {}
        model_state = {}

        hidden_size = 200
        for weight_decay in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            model = RNN(emb=loaded_embeddings, hidden_size=hidden_size, num_classes=3)
            if CUDA:
                model.cuda()
            print("Training", model_type, "for", num_epochs, "epochs with hidden_size=", hidden_size,
                  "and weight decay:", weight_decay, "and", count_parameters(model), "paramaters", flush=True)
            res = train_model(model, train_loader, val_loader, learning_rate=1e-3, num_epochs = num_epochs, verbose='v', weight_decay=weight_decay)
            model_res['RNN-'+str(weight_decay)] = res
            model_state['RNN-'+str(weight_decay)] = model

        save_data("RNN2_results", model_res)
        save_data("RNN2_models", model_state)
