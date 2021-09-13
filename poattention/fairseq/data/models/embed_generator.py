import torch
import torch.nn as nn
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

class EmbeddingGenarator(nn.Module):
    def __init__(self, args, embed, padding_idx, token_padding_idx):
        super(EmbeddingGenarator, self).__init__()
        num_layers = 2

        self.embed = embed
        self.padding_idx = padding_idx
        self.token_padding_idx = token_padding_idx
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(args) for i in range(num_layers)]
        )
        #position embedding
        self.embed_positions = nn.Embedding(100, args.encoder_embed_dim)
        #type embedding
        self.embed_type = nn.Embedding(4, args.encoder_embed_dim)
        self.token_embedding = torch.nn.Parameter(torch.Tensor(1, args.encoder_embed_dim))
        self.token_pos_embedding = torch.nn.Parameter(torch.Tensor(1, args.encoder_embed_dim))
        self.add_mask = (torch.zeros(1, 1, requires_grad = False) != 0).cuda()

    def forward(self, chld_prt_tokens, types, positions): # TODO: change to word index tensor
        seqlen = chld_prt_tokens.size(-1)
        bsz = chld_prt_tokens.size(0)
        num = chld_prt_tokens.size(1)
        chld_prt_tokens = chld_prt_tokens.view(-1, chld_prt_tokens.size(-1))
        types = types.view(-1, types.size(-1))
        positions = positions.view(-1, positions.size(-1))
        token_embs = self.embed(chld_prt_tokens)
        # print(token_embs)
        type_embs = self.embed_type(types)
        position_embs = self.embed_positions(positions)
        # print(self.token_padding_idx)
        padding_mask = chld_prt_tokens.eq(self.token_padding_idx)
        # print(chld_prt_tokens)
        token_embedding = self.token_embedding.unsqueeze(0)
        # print(token_embedding.size())
        token_embedding = token_embedding.repeat(chld_prt_tokens.size(0), 1, 1)
        token_embedding = token_embedding + self.token_pos_embedding

        layer_hiddens = torch.cat([token_embedding, token_embs], dim=1)
        # add sep token padding mask
        add_mask = self.add_mask.repeat(bsz * num, 1)
        padding_mask = torch.cat([add_mask, padding_mask], dim = 1)
        # print(padding_mask.size())
        # print(layer_hiddens.size())
        # print(padding_mask)
        # print(chld_prt_tokens)
        layer_hiddens = layer_hiddens.transpose(0, 1)
        for layer in self.layers:
            layer_hiddens = layer(layer_hiddens, padding_mask)

        layer_hiddens = layer_hiddens.transpose(0, 1)
        new_token_emb = layer_hiddens[:, 0].view(bsz, num, -1)

        return new_token_emb