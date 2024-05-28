import torch
import torch.nn as nn

class CharEmbedding(nn.Module):
    def __init__(self, n_chars, char_emb_dim = 30, dropout = 0.5, use_cnn = True):
        super().__init__()
        self.n_chars = n_chars
        self.char_emb_dim = char_emb_dim
        self.use_cnn = use_cnn

        self.char_embeds = nn.Embedding(self.n_chars, char_emb_dim)
        if use_cnn:
            self.dropout = nn.Dropout(p = dropout)
            self.cnn = nn.Conv1d(in_channels = char_emb_dim, out_channels = char_emb_dim, kernel_size = 3, padding = 1)
        
    def forward(self, char_ids):
        '''
        input:
            char_ids: (batch_size, max_sen_len, max_word_len)
        output:
            char_embeds: (batch_size, max_sen_len, embed_size)
        '''
        batch_size, max_sen_len, max_word_len = char_ids.shape # (batch_size, max_sen_len, max_word_len)
        emb_size = self.char_emb_dim
        char_emb = self.char_embeds(char_ids) # (batch_size, max_sen_len, max_word_len, embed_size)
        if self.use_cnn:
            char_emb = char_emb.reshape(batch_size * max_sen_len, max_word_len, emb_size)
            char_emb = char_emb.permute(0, 2, 1) # (batch_size * max_sen_len, embed_size, max_word_len)
            char_emb = self.dropout(char_emb)
            char_emb = self.cnn(char_emb) # (batch_size * max_sen_len, embed_size, max_word_len)
            char_emb = char_emb.reshape(batch_size, max_sen_len, -1, emb_size) # (batch_size, max_sen_len, max_word_len, embed_size)
            char_emb = torch.max(char_emb, dim = 2).values # (batch_size, max_sen_len, embed_size)
        return char_emb