import numpy as np
import torch
import math

class Time2VecFixed(torch.nn.Module):
    def __init__(self):
        super(Time2VecFixed, self).__init__()
        self.w = torch.tensor([60, 600, 3600, 43200, 86400, 604800, 2.419e6]).unsqueeze(0)
        self.b_sin = torch.nn.parameter.Parameter(torch.randn(1, 7))
        self.b_cos = torch.nn.parameter.Parameter(torch.randn(1, 7))
        self.sin = torch.sin
        self.cos = torch.cos

    def forward(self, tau):
        #print(tau)
        self.w = torch.tensor([60, 600, 3600, 43200, 86400, 604800, 2.419e6], device = tau.device).unsqueeze(0)
        self.b_sin.to(tau.device)
        self.b_cos.to(tau.device)
        #print(self.w)
        #print(self.b_sin.device)
        t1 = tau.tile((7,1)).t().to(tau.device)
        #print("t1: ", t1)
        v1 = self.sin((t1*((2*np.pi)/self.w)) +self.b_sin)
        #print("v1: ", v1)
        v2 = self.cos((t1*((2*np.pi)/self.w)) +self.b_cos)
        return torch.hstack((v1, v2))


class FinalPointWiseFeedForward(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):

        super(FinalPointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Linear(in_dim, 4*in_dim)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Linear(4*in_dim, out_dim)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs)))))
        return outputs

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class T2VSASRec(torch.nn.Module):
    def __init__(self, item_num, config):
        super(T2VSASRec, self).__init__()

        self.item_num = item_num

        # TODO: loss += config.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, config.hidden_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(config.max_seq_len, config.hidden_dim) # TO IMPROVE
        self.time_emb = Time2VecFixed()
        self.emb_dropout = torch.nn.Dropout(p=config.dropout_rate)
        self.tot_dim = config.hidden_dim + 14 #7sin+7cos

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.tot_dim, eps=1e-8)

        for _ in range(config.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.tot_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.tot_dim,
                                                            config.num_heads,
                                                            config.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.tot_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.tot_dim, config.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        self.final_forward = FinalPointWiseFeedForward(self.tot_dim, config.hidden_dim, config.dropout_rate)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, time_seqs):

        #print("original: ", log_seqs)
        #print("original shape: ", log_seqs.shape)

        seqs = self.item_emb(log_seqs) 

        #print("embedded: ", seqs)

        seqs *= self.item_emb.embedding_dim ** 0.5

        ##print("mul by sqrt: ", seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(log_seqs.device)
        #print("pos tile: ", positions)
        
        seqs += self.pos_emb(positions)  
        seqs = self.emb_dropout(seqs)

        time_embs = self.time_emb(time_seqs.reshape(-1)).reshape(time_seqs.shape[0], time_seqs.shape[1], 14) #getting time features
    
        seqs = torch.cat((seqs, time_embs), -1) #adding time features
        #print("new emb seq: ", seqs)
        #print("new emb seq shape: ", seqs.shape)

        timeline_mask = torch.tensor(log_seqs == 0, dtype = torch.bool, device = log_seqs.device)
        #timeline_mask = torch.BoolTensor(log_seqs == 0)

        ##print("pad mask: ", ~timeline_mask.unsqueeze(-1))

        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        ##print("pad masked 1: ", seqs)

        ##print(timeline_mask.shape)
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device))

        #print("atten mask: ", attention_mask)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)

            ##print("Q: ", Q)

            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?

            ##print("att out: ", mha_outputs)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

            #print("linear: ", seqs)

            seqs *=  ~timeline_mask.unsqueeze(-1)

            #print("pad masked 2: ", seqs)

        seqs = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        log_feats = self.final_forward(seqs)

        return log_feats

    def forward(self, log_seqs, time_seqs): # for training        
        log_feats = self.log2feats(log_seqs, time_seqs) # user_ids hasn't been used yet
        return log_feats
