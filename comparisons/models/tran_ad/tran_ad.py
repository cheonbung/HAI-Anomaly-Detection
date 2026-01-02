import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerDecoder
from .modules import PositionalEncoding, TransformerEncoderLayer, TransformerDecoderLayer

class TranAD(nn.Module):
    def __init__(self, feats, window_size, device='cpu'):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.n_feats = feats
        self.n_window = window_size
        self.device = device
        
        # Positional Encoding for (2 * feats) because we concat (src, c)
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        
        # Encoder
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        
        # Decoder 1 (Phase 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        
        # Decoder 2 (Phase 2)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        
        # Final Output Projection
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        # src: (Seq, Batch, Feats)
        # c: (Seq, Batch, Feats) -> Focus Score / Deviation
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores (c=0)
        c = torch.zeros_like(src)
        tgt_enc, memory = self.encode(src, c, tgt)
        x1 = self.fcn(self.transformer_decoder1(tgt_enc, memory))
        
        # Phase 2 - With anomaly scores (c = (x1 - src)^2)
        c = (x1 - src) ** 2
        tgt_enc, memory = self.encode(src, c, tgt)
        x2 = self.fcn(self.transformer_decoder2(tgt_enc, memory))
        
        return x1, x2