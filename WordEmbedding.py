import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, n_words, word_emb_dim = 100):
        super().__init__()
        self.n_words = n_words
        self.word_emb_dim = word_emb_dim
        self.word_embeds = nn.Embedding(self.n_words, self.word_emb_dim)

    def forward(self, word_ids):
        '''
        input:
            word_ids: (batch_size, max_sen_len)
        output:
            word_embeds: (batch_size, max_sen_len, emb_len)
        '''
        word_emb = self.word_embeds(word_ids)
        return word_emb