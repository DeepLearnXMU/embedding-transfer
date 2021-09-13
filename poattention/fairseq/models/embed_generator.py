import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import TransformerEncoderLayer

class EmbeddingGenerator(nn.Module):
    def __init__(self, args, embed_tokens, gather_padding_index, padding_idx):
        super(EmbeddingGenerator, self).__init__()

        num_layers = 2

        self.embed = embed_tokens
        self.gather_padding_index = gather_padding_index
        self.padding_idx = padding_idx
        # self.layers = nn.ModuleList([])
        # self.layers.extend(
        #     [TransformerEncoderLayer(args) for i in range(num_layers)]
        # )

        # self.embed_positions = nn.Embedding(100, embed_tokens.embedding_dim)
        # self.embed_type = nn.Embedding(4, embed_tokens.embedding_dim)
        # self.token_pos_embedding = nn.Parameter(torch.FloatTensor(1, 1, embed_tokens.embedding_dim), requires_grad=True)
        self.token_embedding = nn.Parameter(torch.FloatTensor(1, 1, embed_tokens.embedding_dim), requires_grad=True)
        # self.type_embedding = nn.Parameter(torch.FloatTensor(1, 1, embed_tokens.embedding_dim), requires_grad=True)
        self.emb_dim = embed_tokens.embedding_dim
        # nn.init.normal_(self.token_pos_embedding)
        nn.init.normal_(self.token_embedding)
        self.embed_positions = nn.Embedding(7, embed_tokens.embedding_dim)
        # nn.init.normal_(self.type_embedding)

        self.add_mask = (torch.zeros(1, 1, requires_grad=False).cuda() != 0)

    def forward(self, chld_prt_tokens, types, positions):
        if len(chld_prt_tokens.size()) == 3:
            bsz = chld_prt_tokens.size(0)
            num = chld_prt_tokens.size(1)
            seq_len = chld_prt_tokens.size(-1)
            now_chld_prt_tokens = chld_prt_tokens.view(bsz * num, seq_len)
            now_types = types.view(bsz * num, seq_len)
            now_positions = positions.view(bsz * num, seq_len)
        else:
            bsz = 1
            num = chld_prt_tokens.size(0)
            seq_len = chld_prt_tokens.size(-1)
            now_chld_prt_tokens = chld_prt_tokens
            now_types = types
            now_positions = positions
        # print(chld_prt_tokens)

        token_embs = self.embed(now_chld_prt_tokens)

        this_token_embs = self.embed_positions(now_positions)
#         this_token_embs = pos_embs.repeat(bsz * num, 1, 1)
        # this_types_embs = self.type_embedding.repeat(bsz * num, 1, 1)
        # this_position_embs = self.token_pos_embedding.repeat(bsz * num, 1, 1)

        padding_mask = now_chld_prt_tokens.eq(self.padding_idx)                                                                                                                                                                                                                                                                                                                
        # now_add_mask = self.add_mask.repeat(padding_mask.size(0), 1)
        # padding_mask = torch.cat([now_add_mask, padding_mask], dim=1)
#         scores = torch.bmm(token_embs, this_token_embs.transpose(1, -1)).squeeze(-1)
        # print(token_embs.size())
        # print(this_token_embs.size())
        # print(now_chld_prt_tokens.size())
        # print(now_positions.size())
        # print(now_chld_prt_tokens)
        # print(now_positions)
        scores = (token_embs * this_token_embs).sum(dim=-1)
        scores = scores.masked_fill(mask=padding_mask, value=-99999999)
        scores = nn.functional.softmax(scores)
        scores = scores.unsqueeze(-1)
        res = (scores * token_embs).sum(1)
        # print(scores)
        # print(scores.size())
        # exit()

        if len(chld_prt_tokens.size()) == 3:
            gened_emb = res.view(bsz, num, self.emb_dim)
        else:
            gened_emb = res.view(num, self.emb_dim)
        # print(gened_emb.size())
        # print(gened_emb)
        return gened_emb

