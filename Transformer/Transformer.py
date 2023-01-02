d_model = 100
d_hid = 256 # 原本512
n_layer = 3
n_head = 4

#d_model = 128
class Classification2(nn.Module):
    def __init__(self):
        super(Classification2, self).__init__()
        TEMP = nn.TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = d_hid, activation = 'relu'
                                         , norm_first = True)
        #self.enc_embedding = nn.Embedding(len(en_dict), d_model)
        #self.enc_embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        self.enc_embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), freeze = False)
        self.pos_embedding = PositionalEncoding()
        self.encoder = nn.TransformerEncoder(TEMP, num_layers = n_layer)
        self.NORM1 = nn.LayerNorm(d_model)
        self.NORM2 = nn.LayerNorm(2500) # 2500, 3200
        self.NORM3 = nn.LayerNorm(512)
        self.POOL = nn.AvgPool1d(4)
        self.Dense = nn.Linear(2500 , 128) #本來是512
        self.predict = nn.Linear(128, 4)
    def forward(self, enc_input):
        N = enc_input.shape[1]
        
        x = self.enc_embedding(enc_input)
        x = self.pos_embedding.forward(x)
        #x = self.EMB(enc_input)
        x = self.encoder(x, src_key_padding_mask = enc_input.eq(0).t()) #0 , en_dict['<PAD>']
        x = x.transpose(1, 0)
        x = self.NORM1(x)
        x = self.POOL(x)
        x = nn.Dropout(0.5)(x)
        x = x.reshape(N, -1)
        x = self.NORM2(x)
        x = nn.ReLU()(self.Dense(x))
        x = nn.Dropout(0.5)(x)
        predict = nn.Softmax(dim = 1)(self.predict(x))
        return predict
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = d_model, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)