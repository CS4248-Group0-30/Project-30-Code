from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import sys
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

isENCN = True  #global modifier, for EN-CN or CN-EN translation
isReverse = False # global modifier, default ENCN, reverse CNEN
useGlove = False #global modifier, use glove to embed english
useTencent = False #global modifier, use tencent AI lab chinese embedding
n_fileCut = 200000
MAX_LENGTH = 10  #<---- param
n_epochs = 500000

ifSave = True


# Load GloVe embeddings
def load_glove_embeddings(path, embedding_dim):
    embeddings = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            embeddings[word] = embedding
    # Add an embedding for unknown words
    embeddings['<unk>'] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embeddings

os.chdir(r'E:\National University of Singapore\Lin Yuan Xun, Caleb - CS4248\RNN enc_dec_baseline/')
glove_path = 'glove.6B.100d.txt'
glove_embeddings = load_glove_embeddings(glove_path, embedding_dim=100)

# Create embedding layer
def create_embedding_layer(word_to_index, embeddings):
    #embeddings = glove_embeddings
    vocab_size = len(word_to_index)
    embedding_dim = int(embeddings[next(iter(embeddings))].shape[0])
    embedding_matrix = np.zeros((vocab_size+3, embedding_dim), dtype='float32')
    for word, index in word_to_index.items():
        word = word.lower()
        if word == "'Driving'":
            print('error, word format',index,word,embedding_matrix.shape)
            sys.exit()
        embedding = embeddings.get(word)
        if embedding is not None:
            embedding_matrix[index] = embedding
        else:
            try:
                embedding_matrix[index] = embeddings['<unk>']
            except:
                print('error, embedding index',index,word,embedding_matrix.shape)
                sys.exit()
                
    embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))
    return embedding_layer


# Load the Tencent AI Lab embedding file
os.chdir(r'E:\National University of Singapore\Lin Yuan Xun, Caleb - CS4248\RNN enc_dec_baseline/')
embeddings_file_CN = 'tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
word_vectors_CN = KeyedVectors.load_word2vec_format(embeddings_file_CN, binary=False)

# Define the PyTorch embedding layer
embedding_dim_CN = word_vectors_CN.vector_size
num_words_CN = len(word_vectors_CN.index_to_key)
embedding_matrix_CN = np.zeros((num_words_CN, embedding_dim_CN))
for i, word in enumerate(word_vectors_CN.index_to_key):
    embedding_matrix_CN[i] = word_vectors_CN[word]
embedding_layer_CN = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix_CN), freeze=True)


SOS_token = 0
EOS_token = 1
UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 3  # Count SOS, EOS and UNK

    def addSentence(self, sentence):
        if isENCN:
            wordLst = pd.Series(sentence.split(','))
            wordLst = wordLst.str.strip(' ')
            wordLst = wordLst.values.tolist()   
        else:
            wordLst = sentence.split(' ')
                
        for word in wordLst:
            self.addWord(word)
                
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# lang2 translate into lang1 if not reverse
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    os.chdir(r'E:\National University of Singapore\Lin Yuan Xun, Caleb - CS4248/')
    tokenized_EN_CN = pd.read_csv('tokenized_train_data_en_zh_cleaned.csv')
    tokenized_EN_CN = tokenized_EN_CN[:n_fileCut]

    # reorganize csv into lines
    pairs = tokenized_EN_CN[['tokens','tokens_zh']]
    pairs['tokens'] = pairs['tokens'].str.lower()
    pairs = pairs.values.tolist()
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



def filterPair(p):
    return len(p[0].split(',')) < MAX_LENGTH and \
            len(p[1].split(',')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

"""
-  Read text file and split into lines, split lines into pairs
-  Normalize text, filter by length and content
-  Make word lists from sentences in pairs
"""

def prepareData(lang1, lang2, reverse=False): 
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('EN', 'CN', isReverse) # control reverse here
print(random.choice(pairs))

word_to_index = input_lang.word2index
embedding_layer = create_embedding_layer(word_to_index, glove_embeddings)
word_to_index["SOS"] = 0
word_to_index["EOS"] = 1
word_to_index['UNK'] = 2

word_to_index_CN = output_lang.word2index #so the english letters were tokenized with ''
word_to_index_CN["SOS"] = 0
word_to_index_CN["EOS"] = 1
word_to_index_CN['UNK'] = 2

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if useGlove and not isReverse:
            self.embedding = embedding_layer
        elif useTencent and isReverse:
            self.embedding = embedding_layer_CN
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)
            
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        if useTencent and not isReverse:
            self.embedding = embedding_layer_CN
        elif useGlove and isReverse:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        
        #print(embedded[0].shape,hidden[0].shape)
        
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):

    wordLst = pd.Series(sentence.split(','))
    wordLst = wordLst.str.strip(' ')
    wordLst = wordLst.values.tolist()   
     
    lang_word2index_list = []
    for word in wordLst:
        if word in lang.word2index.keys():
            lang_word2index_list.append(lang.word2index[word])
        else:
            lang_word2index_list.append(lang.word2index['UNK'])
    return lang_word2index_list

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5  #<----param, all previous was set to 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        try:
            encoder_outputs[ei] = encoder_output[0, 0]
        except:
            print(input_length,ei,"error on encoder_outputs in 'train' func",
                  encoder_outputs.shape,
                  encoder_output[0, 0].shape)
        
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


#helper function
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

"""
-  Start a timer
-  Initialize optimizers and criterion
-  Create set of training pairs
-  Start empty losses array for plotting
"""

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
 
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # LR schedule
        for g in encoder_optimizer.param_groups:
            if n_iters/10000 >3:
                g['lr'] = 0.005
            if n_iters/10000 >6:
                g['lr'] = 0.001
                
        for g in decoder_optimizer.param_groups:
            if n_iters/10000 >3:
                g['lr'] = 0.005
            if n_iters/10000 >6:
                g['lr'] = 0.001
                
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

# Plotting results

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# Evaluation
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        
        # testing, rem to comment out
        #sentence = line
        #max_length = MAX_LENGTH
        #encoder = encoder1
        #decoder = attn_decoder1
        
        
        input_tensor = tensorFromSentence(input_lang, sentence)
        if len(input_tensor) >= max_length:
            input_tensor = input_tensor[:max_length]
            input_tensor[max_length-1] = EOS_token
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            try:
                encoder_outputs[ei] += encoder_output[0, 0]
            except:
                print('error on encoder_outputs[ei] += encoder_output[0, 0]')
                print(ei,encoder_output.shape)
            
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]



def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
def evaluateRandomlyReturnDF(encoder, decoder, n=10000):
    evaluatedDF = pd.DataFrame(columns=['en','zh','translated'])
    for i in range(n):
        pair = random.choice(pairs)
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        #print('<', output_sentence)
        #print('')
        if not isReverse:
            lineDF = pd.DataFrame({'en':pair[0],'zh':pair[1],'translated':output_sentence},index=[0])
        else:
            lineDF = pd.DataFrame({'en':pair[1],'zh':pair[0],'translated':output_sentence},index=[0])  
        evaluatedDF = pd.concat([evaluatedDF,lineDF])
        
    return evaluatedDF
        
             
"""## Training and Evaluating
"""
# parameter and suffixs
#print('NOT SUPPORTED currently: hidden_size_EN need to equal hidden_size_CN')
if useGlove or useTencent:
    hidden_size = 100
else:
    hidden_size = 256
    
str_lang1_lang2 = 'EN_CN'
if isReverse:
    str_lang1_lang2 = 'CN_EN'

str_bespoke_embed = '_'
if useGlove:
    str_bespoke_embed = str_bespoke_embed+'glove_'
if useTencent:
    str_bespoke_embed = str_bespoke_embed+'Tencent_'
    
# train model initialize 
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# alternatively, load half trained         
"""
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
encoder1.load_state_dict(torch.load('encoder_'+str_lang1_lang2+str_bespoke_embed+str(MAX_LENGTH)+'.dict'))
attn_decoder1.load_state_dict(torch.load('decoder_'+str_lang1_lang2+str_bespoke_embed+str(MAX_LENGTH)+'.dict'))
"""

trainIters(encoder1, attn_decoder1, n_epochs, print_every=1000)
evaluateRandomly(encoder1, attn_decoder1)

if not ifSave:
    sys.exit()

# save model
os.chdir(r'E:\National University of Singapore\Lin Yuan Xun, Caleb - CS4248\RNN enc_dec_baseline\training_results/')
torch.save(encoder1.state_dict(), 'encoder_'+str_lang1_lang2+str_bespoke_embed+str(MAX_LENGTH)+'_fileCut_'+str(n_fileCut)+'.dict')
torch.save(attn_decoder1.state_dict(),'decoder_'+str_lang1_lang2+str_bespoke_embed+str(MAX_LENGTH)+'_fileCut_'+str(n_fileCut)+'.dict')
   
# output some results
evaluatedDF_output = evaluateRandomlyReturnDF(encoder1, attn_decoder1,n=10000)
os.chdir(r'E:\National University of Singapore\Lin Yuan Xun, Caleb - CS4248\RNN enc_dec_baseline\training_results/')
evaluatedDF_output.to_csv('translated_test_sentences_'+str_lang1_lang2+str_bespoke_embed+str(MAX_LENGTH)+'_fileCut_'+str(n_fileCut)+'.csv')


# load model 
#"""
os.chdir(r'E:\National University of Singapore\Lin Yuan Xun, Caleb - CS4248\RNN enc_dec_baseline\training_results/')
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
encoder1.load_state_dict(torch.load('encoder_'+str_lang1_lang2+str_bespoke_embed+str(MAX_LENGTH)+'_fileCut_'+str(n_fileCut)+'.dict'))
attn_decoder1.load_state_dict(torch.load('decoder_'+str_lang1_lang2+str_bespoke_embed+str(MAX_LENGTH)+'_fileCut_'+str(n_fileCut)+'.dict'))
evaluateRandomly(encoder1, attn_decoder1)


#
def evaluateBespoke(encoder, decoder, line_raw, n=10):
    line = line_raw.replace(' ',',')
    line = line.lower()
    print('>', line)
    output_words, attentions = evaluate(encoder, decoder, line)
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')
    
line_raw = "Soon after when I was walking past a train station I saw something terrible that to this day I cant erase from my memory."
evaluateBespoke(encoder1, attn_decoder1, line_raw)

