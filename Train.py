import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from Model import SpamDetect
from SpamData import SpamDataset
from pytorchtools import EarlyStopping
from Utils import logger

torch.manual_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Trainer():
    def __init__(self, epochs = 100, 
        use_pretrained = True, use_char = True, use_lm = True, use_crf = True, use_cnn = True):
        super().__init__()

        self.char_emb_dim = 100
        self.word_emb_dim = 100
        self.hidden_dim = 256
        self.lstm_layers = 1
        self.dropout = 0.1
        self.epochs = epochs
        self.batch_size = 8
        self.use_pretrained = use_pretrained
        self.use_char = use_char
        self.use_cnn = use_cnn
        self.use_crf = use_crf
        self.use_lm = use_lm
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger('device = {}'.format(self.device))

        self.dataset = SpamDataset()
        self.train_set = self.dataset[: self.dataset.train_num]
        self.valid_set = self.dataset[self.dataset.train_num: self.dataset.train_num + self.dataset.valid_num]
        self.test_set = self.dataset[self.dataset.train_num + self.dataset.valid_num: self.dataset.total_num]
        self.n_words = len(self.dataset.word_to_ix)
        self.n_chars = len(self.dataset.char_to_ix)

    def set_model(self, norm = None):
        self.model = SpamDetect(
            self.n_words, self.n_chars,
            self.word_emb_dim, self.char_emb_dim, self.hidden_dim, self.lstm_layers, 
            self.dropout, norm
        ).to(self.device)

    def train(self):
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr = 1e-4)
        early_stopping = EarlyStopping(patience = 5, verbose = False)
        entrophy = nn.CrossEntropyLoss()

        avg_train_losses = []
        avg_valid_losses = []
        train_texts, train_word_ids, train_word_masks, train_char_ids, train_label_ids = self.train_set
        valid_texts, valid_word_ids, valid_word_masks, valid_char_ids, valid_label_ids = self.valid_set
        for epoch in range(self.epochs):
            train_losses = []
            valid_losses = []
            model.train()
            i = 0
            train_correct = 0
            valid_correct = 0
            while i < self.dataset.train_num:
                if i + self.batch_size < self.dataset.train_num:
                    text, word_id, word_mask, char_id, label_id = \
                        train_texts[i: i + self.batch_size], train_word_ids[i: i + self.batch_size], train_word_masks[i: i + self.batch_size], train_char_ids[i: i + self.batch_size], train_label_ids[i: i + self.batch_size]
                else:
                    text, word_id, word_mask, char_id, label_id = \
                        train_texts[i:], train_word_ids[i:], train_word_masks[i:], train_char_ids[i:], train_label_ids[i:]
                i += self.batch_size
                word_id = torch.tensor(word_id, dtype = torch.long, device = self.device)
                word_mask = torch.tensor(word_mask, dtype = torch.bool, device = self.device)
                char_id = torch.tensor(char_id, dtype = torch.long, device = self.device)
                label_id = torch.tensor(label_id, dtype = torch.long, device = self.device)
                
                optimizer.zero_grad()
                output = model(None, word_id, word_mask, char_id) # (batch_size, tagset_size)
                loss = entrophy(output, label_id)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                
                predict = torch.argmax(output, dim = -1)
                train_correct += torch.sum(predict == label_id)

            model.eval()
            with torch.no_grad():
                i = 0
                while i < self.dataset.valid_num:
                    if i + self.batch_size < self.dataset.valid_num:
                        text, word_id, word_mask, char_id, label_id = \
                            valid_texts[i: i + self.batch_size], valid_word_ids[i: i + self.batch_size], valid_word_masks[i: i + self.batch_size], valid_char_ids[i: i + self.batch_size], valid_label_ids[i: i + self.batch_size]
                    else:
                        text, word_id, word_mask, char_id, label_id = \
                            valid_texts[i:], valid_word_ids[i:], valid_word_masks[i:], valid_char_ids[i:], valid_label_ids[i:]
                    i += self.batch_size
                    word_id = torch.tensor(word_id, dtype = torch.long, device = self.device)
                    word_mask = torch.tensor(word_mask, dtype = torch.bool, device = self.device)
                    char_id = torch.tensor(char_id, dtype = torch.long, device = self.device)
                    label_id = torch.tensor(label_id, dtype = torch.long, device = self.device)
                    
                    output = model(None, word_id, word_mask, char_id) # (batch_size, tagset_size)
                    loss = entrophy(output, label_id)
                    valid_losses.append(loss.item())

                    predict = torch.argmax(output, dim = -1)
                    valid_correct += torch.sum(predict == label_id)
                avg_train_loss = np.average(train_losses)
                avg_valid_loss = np.average(valid_losses)
                avg_train_losses.append(avg_train_loss)
                avg_valid_losses.append(avg_valid_loss)
                logger('[epoch {:3d}] train_loss: {:.8f}  valid_loss: {:.8f}  train_acc: {:.4f}  valid_acc: {:.4f}'.format(epoch + 1, avg_train_loss, avg_valid_loss, train_correct / self.dataset.train_num, valid_correct / self.dataset.valid_num))
                early_stopping(avg_valid_loss, model)
                if early_stopping.early_stop:
                    logger("Early stopping")
                    break
        self.model = model
        self.test()
        return avg_train_losses
            
    def test(self):
        model = self.model
        model.eval()
        tp, tn, fp, fn = 0, 0, 0, 0
        correct = 0
        total = self.dataset.test_num
        logger('Begin testing.')
        test_texts, test_word_ids, test_word_masks, test_char_ids, test_label_ids = self.test_set
        with torch.no_grad():
            i = 0
            while i < self.dataset.test_num:
                if i + self.batch_size < self.dataset.test_num:
                    text, word_id, word_mask, char_id, label_id = \
                        test_texts[i: i + self.batch_size], test_word_ids[i: i + self.batch_size], test_word_masks[i: i + self.batch_size], test_char_ids[i: i + self.batch_size], test_label_ids[i: i + self.batch_size]
                else:
                    text, word_id, word_mask, char_id, label_id = \
                        test_texts[i:], test_word_ids[i:], test_word_masks[i:], test_char_ids[i:], test_label_ids[i:]
                i += self.batch_size
                word_id = torch.tensor(word_id, dtype = torch.long, device = self.device)
                word_mask = torch.tensor(word_mask, dtype = torch.bool, device = self.device)
                char_id = torch.tensor(char_id, dtype = torch.long, device = self.device)
                label_id = torch.tensor(label_id, dtype = torch.long, device = self.device)
                
                output = model(None, word_id, word_mask, char_id) # (batch_size, sen_len, tagset_size)
                predict = torch.argmax(output, dim = -1)
                correct += torch.sum(predict == label_id)
                tp += torch.sum((predict == label_id)[predict == self.dataset.label_to_ix['spam']])
                tn += torch.sum((predict == label_id)[predict != self.dataset.label_to_ix['spam']])
                fp += torch.sum((predict != label_id)[predict == self.dataset.label_to_ix['spam']])
                fn += torch.sum((predict != label_id)[predict != self.dataset.label_to_ix['spam']])
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 0.0000001)
            logger('[Test] Test accuracy: {:.8f}'.format(correct / self.dataset.test_num))
            logger('[Test] precision: {:.8f}, recall: {:.8f}, f1: {:.8f}'.format(precision, recall, f1))


if __name__ == '__main__':
    pretrained_path = '/home/gene/Documents/Data/Glove/glove.6B.100d.txt'

    trainer = Trainer(epochs = 100, 
                use_pretrained = True, use_char = True, use_lm = True, use_crf = True)
    logger('No normalization:')
    trainer.set_model()
    none_loss = trainer.train()
    logger('Layer normalization:')
    trainer.set_model(norm = 'layer')
    layer_loss = trainer.train()
    logger('Batch normalization:')
    trainer.set_model(norm = 'batch')
    batch_loss = trainer.train()

    model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(none_loss)
    plt.plot(layer_loss)
    plt.plot(batch_loss)
    plt.legend(['no norm', 'layer norm', 'batch norm'])
    plt.savefig('./result/loss_{}.png'.format(model_time), format = 'png')
    
