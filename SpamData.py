import csv
import random
import torch
from Utils import logger

class SpamDataset():
    def __init__(self):
        self.file_name = './spam.csv'
        self.OOV_TAG = '<OOV>'
        self.PAD_TAG = '<PAD>'
        self.max_sen_len = 64
        self.max_word_len = 16
        self.text = []
        self.word_ids = []
        self.word_masks = []
        self.char_ids = []
        self.label_ids = []
        self.load_spam(shuffle = True)
        self.split_data()
        logger('Load data. train: {}, valid: {}, test: {}'.format(self.train_num, self.valid_num, self.test_num))

    def load_spam(self, shuffle = True):
        self.data = []
        self.word_list = [self.PAD_TAG, self.OOV_TAG]
        self.word_to_ix = {self.PAD_TAG: 0, self.OOV_TAG: 1}
        self.char_list = set([self.PAD_TAG, self.OOV_TAG])
        self.label_list = set()
        
        f = open(self.file_name, 'r', encoding = 'windows-1252')
        csv_reader=  csv.reader(f)
        for line in csv_reader:
            break
        for line in csv_reader:
            label = line[0]
            text = line[1]
            words = text.replace('.', ' ').replace(',', ' ').replace(':', ' ').replace('!', ' ').replace('?', ' ').split(' ')
            for word in words:
                if word and word.lower() not in self.word_to_ix:
                    self.word_list.append(word.lower())
                    self.word_to_ix[word.lower()] = len(self.word_to_ix)
            self.data.append([list(filter(None, words)), label])
            self.char_list |= set(text)
            self.label_list.add(label)
        f.close()
        self.char_list = list(self.char_list)
        self.char_to_ix = {self.char_list[i]: i for i in range(len(self.char_list))}
        self.label_list = list(self.label_list)
        self.label_to_ix = {self.label_list[i]: i for i in range(len(self.label_list))}
        # if shuffle:
        #     random.shuffle(self.data)
        for line in self.data:
            words, label = line
            self.map_and_pad(words, label)
    
    def map_and_pad(self, text, label):
        word_ids = self.map_word(text, dim = 2)
        char_ids = self.map_char(text, dim = 2)
        word_ids, word_masks = self.padding_fixed(word_ids, padding_value = 0, dim = 2)
        char_ids, _ = self.padding_fixed(char_ids, padding_value = 0, dim = 3)
        label_id = self.label_to_ix[label]

        self.text.append(text)
        self.word_ids.append(word_ids)
        self.word_masks.append(word_masks)
        self.char_ids.append(char_ids)
        self.label_ids.append(label_id)
    
    def padding_fixed(self, sentence, padding_value = 0, dim = 2):
        '''
        sentences: list, (list(list))
        dim: 
            dim = 2, word padding, result = (sen_len)
            dim = 3, char padding, result = (sen_len, word_len)
        '''
        max_sen_len = self.max_sen_len
        max_word_len = self.max_word_len
        if dim == 2: # word padding
            padded_data = sentence + [padding_value] * (max_sen_len - len(sentence))
            padded_mask = [1] * len(sentence) + [0] * (max_sen_len - len(sentence))
            return padded_data[: max_sen_len], padded_mask[: max_sen_len]
        if dim == 3: # char padding, [[char1, char2, ..], [], ...]
            zero_padding = [padding_value] * max_word_len # [0, 0, 0, ..]
            zero_mask = [0] * max_word_len
            padded_data = [word[: max_word_len] + [padding_value] * (max_word_len - len(word)) for word in sentence] + [zero_padding] * (max_sen_len - len(sentence))
            padded_mask = [[1] * len(word[: max_word_len]) + [0] * (max_word_len - len(word)) for word in sentence] + [zero_mask] * (max_sen_len - len(sentence))
            return padded_data[: max_sen_len], padded_mask[: max_sen_len]

    def __len__(self):
        return len(self.text) # number of sentence

    def __getitem__(self, index):
        return self.text[index], self.word_ids[index], self.word_masks[index], self.char_ids[index], self.label_ids[index]


    def split_data(self, train_ratio = 0.7, valid_ratio = 0.2, shuffle = False):
        self.total_num = len(self.word_ids)
        self.valid_num = int(self.total_num * train_ratio * valid_ratio)
        self.train_num = int(self.total_num * train_ratio) - self.valid_num
        self.test_num = self.total_num - self.train_num - self.valid_num
        if shuffle:
            index = list(range(self.total_num))
            random.shuffle(index)
    
    def map_word(self, words, dim = 2):
        if dim == 2: # words
            return [self.word_to_ix[word.lower()] if word.lower() in self.word_to_ix else self.word_to_ix[self.OOV_TAG] for word in words]
        if dim == 1:
            return self.word_to_ix[words] if words in self.word_to_ix else self.word_to_ix[self.OOV_TAG]
    
    def map_char(self, words, dim = 2):
        if dim == 2: # words
            return [[self.char_to_ix[char] if char in self.char_to_ix else self.char_to_ix[self.OOV_TAG] for char in word] for word in words]
        if dim == 1: # word
            return [self.char_to_ix[char] if char in self.char_to_ix else self.char_to_ix[self.OOV_TAG] for char in words]

if __name__ == '__main__':
    spam_data = SpamDataset()
    print(spam_data[0])




