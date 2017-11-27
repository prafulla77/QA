import parse
import model
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

FloatTensor = torch.cuda.FloatTensor if model.USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if model.USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if model.USE_CUDA else torch.ByteTensor

def _set_config(vocab_size):
    config = dict()
    config['input_size'] = vocab_size
    config['hidden_size'] = 80
    config['output_size'] = 1
    config['dropout_p'] = 0.2
    config['batch_size'] = 1000
    config['learning_rate'] = 0.01
    config['num_episode'] = 30
    config['epoch'] = 1
    config['stop_early'] = True
    return config

def get_batch(batch_size,train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        yield batch
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if w in to_index.keys() else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def batch_to_input(batch, word2index):
    contexts, questions, answer_I, answer_II, gold_output = list(zip(*batch))
    max_ctxt = max([len(ctxt) for ctxt in contexts])
    max_len = max([ctxt.size(1) for ctxt in parse.flatten(contexts)])
    max_q = max([qq.size(1) for qq in questions])
    max_a_I = max([aa.size(1) for aa in answer_I])
    max_a_II = max([aa.size(1) for aa in answer_II])
    
    c_p, c_mask, q_p, a_I_p, a_II_p = [], [], [], [], []
    for i in range(len(batch)):
        temp_c_p = []
        for j in range(len(contexts[i])):
            if contexts[i][j].size(1) == max_len:
                temp_c_p.append(contexts[i][j])
            else:
                temp_c_p.append(torch.cat([contexts[i][j], Variable(LongTensor([word2index['<PAD>']]*(max_len-contexts[i][j].size(1)))).view(1,-1)], 1))
        while(len(temp_c_p) < max_ctxt):
            temp_c_p.append(Variable(LongTensor([word2index['<PAD>']]*max_len)).view(1,-1))

        temp_c_p = torch.cat(temp_c_p)
        c_p.append(temp_c_p)
        c_mask.append(torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))),volatile=False) for t in temp_c_p]).view(temp_c_p.size(0),-1))

        if questions[i].size(1) == max_q:
            q_p.append(questions[i])
        else:
            q_p.append(torch.cat([questions[i],Variable(LongTensor([word2index['<PAD>']]*(max_q-questions[i].size(1)))).view(1,-1)],1))
        
        if answer_I[i].size(1) == max_a_I:
            a_I_p.append(answer_I[i])
        else:
            a_I_p.append(torch.cat([answer_I[i],Variable(LongTensor([word2index['<PAD>']]*(max_a_I-answer_I[i].size(1)))).view(1,-1)],1))
        
        if answer_II[i].size(1) == max_a_II:
            a_II_p.append(answer_II[i])
        else:
            a_II_p.append(torch.cat([answer_II[i],Variable(LongTensor([word2index['<PAD>']]*(max_a_II-answer_II[i].size(1)))).view(1,-1)],1))
    q_p = torch.cat(q_p)
    a_I_p = torch.cat(a_I_p)
    a_II_p = torch.cat(a_II_p)
    q_mask = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))),volatile=False) for t in q_p]).view(q_p.size(0),-1)
    a_I_mask = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))),volatile=False) for t in a_I_p]).view(a_I_p.size(0),-1)
    a_II_mask = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))),volatile=False) for t in a_II_p]).view(a_II_p.size(0),-1)
    return c_p, c_mask, q_p, q_mask, a_I_p, a_I_mask, a_II_p, a_II_mask, Variable(LongTensor(gold_output))

def _set_config(vocab_size):
    config = dict()
    config['input_size'] = vocab_size
    config['hidden_size'] = 100
    config['output_size'] = 1
    config['dropout_p'] = 0.2
    config['batch_size'] = 50
    config['learning_rate'] = 0.01
    config['num_episode'] = 3
    config['epoch'] = 30
    config['stop_early'] = False
    return config

def _train():
    _data = parse._parse_file('../data/train_qa.xml')
    #test_data = _parse_file('../data/dev_qa.xml')
    vocab = parse._get_vocab(_data)
    word2index, index2word = parse._get_word_to_index(vocab)
    config = _set_config(len(word2index))
    train_data = parse._get_training_data(_data)
    for t in train_data:
        for i,fact in enumerate(t[0]):
            t[0][i] = prepare_sequence(fact,word2index).view(1,-1)
        t[1] = prepare_sequence(t[1],word2index).view(1,-1)
        t[2] = prepare_sequence(t[2],word2index).view(1,-1)
        t[3] = prepare_sequence(t[3],word2index).view(1,-1)
    DMN = model.DMN(config)
    DMN.init_weight()
    if model.USE_CUDA: DMN = DMN.cuda()
    print "model intialized"
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(DMN.parameters(), lr=config['learning_rate'])
    for epoch in range(config['epoch']):
        losses = []
        if config['stop_early']: break
        for i, batch in enumerate(get_batch(config['batch_size'], train_data)):
            c_p, c_mask, q_p, q_mask, a_I_p, a_I_mask, a_II_p, a_II_mask, gold_output = batch_to_input(batch, word2index)
            DMN.zero_grad()
            pred = DMN(c_p, c_mask, q_p, q_mask, a_I_p, a_I_mask, a_II_p, a_II_mask, True)
            loss = loss_function(pred, gold_output)
            losses.append(loss.data.tolist()[0])
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print("[%d/%d] mean_loss : %0.2f" %(epoch,config['epoch'],np.mean(losses)))
            
            if np.mean(losses)<0.01:
                config['stop_early']=True
                print("Early Stopping!")
                break
            losses=[]
    torch.save(DMN.state_dict(), 'baseline.pt')

if __name__ == '__main__':
    _train()