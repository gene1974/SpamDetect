import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from WordEmbedding import WordEmbedding
from CharEmbedding import CharEmbedding
from Norm import LayerNorm, BatchNorm

class SpamDetect(nn.Module):
    def __init__(self, n_words, n_chars,
        word_emb_dim, char_emb_dim, hidden_size, lstm_layers, 
        dropout = 0.1, norm = None):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.word_emb_dim = word_emb_dim
        self.char_emb_dim = char_emb_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.max_sen_len = 64
        self.norm = norm

        self.emb_dim = word_emb_dim + char_emb_dim
        
        self.char_embeds = CharEmbedding(n_chars, char_emb_dim, dropout = dropout, use_cnn = True)
        self.word_embeds = WordEmbedding(n_words, word_emb_dim)
        if norm == 'batch':
            self.batch_norm = BatchNorm(self.hidden_size)
        if norm == 'layer':
            self.layer_norm = LayerNorm(self.hidden_size)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, lstm_layers)
        self.flat = nn.Flatten(start_dim = 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.max_sen_len * hidden_size, 2)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        

    def forward(self, text, word_ids, word_mask, char_ids): # (batch_size, sen_len)
        word_emb = self.word_embeds(word_ids) # (batch_size, sen_len, word_embed_size)
        char_emb = self.char_embeds(char_ids) # (batch_size, sen_len, char_embed_size)
        embeds = torch.cat((word_emb, char_emb), dim = -1)
        embeds = self.dropout1(embeds)
        
        batch_size, max_sen_len = embeds.shape[:2]
        sen_len = torch.sum(word_mask, dim = 1, dtype = torch.int64).to('cpu') # (batch_size)
        pack_seq = pack_padded_sequence(embeds, sen_len, batch_first = True, enforce_sorted = False)
        lstm_out, _ = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first = True) # (batch_size, seq_len, hidden_size)
        lstm_out = torch.cat((lstm_out, torch.zeros((batch_size, max_sen_len - max(sen_len), self.hidden_size), device = self.device)), dim = 1)
        lstm_feats = self.dropout2(lstm_out) # (batch_size, seq_len, hidden_size)
        if self.norm == 'batch':
            lstm_feats = self.batch_norm(lstm_feats)
        if self.norm == 'layer':
            lstm_feats = self.layer_norm(lstm_feats)
        lstm_feats = self.flat(lstm_feats)
        lstm_feats = self.classifier(lstm_feats) # （batch_size, tagset_size)
        return lstm_feats
        

