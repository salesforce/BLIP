import torch
from torch import nn
class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, attn_dropout=0.1, out_dropout=0.1):
        super(BiMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert(
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim should be divisible by num_heads, got embed_dim={embed_dim} and num_heads={num_heads}!"
        self.scale = self.num_heads ** (-0.5)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_dropout = nn.Dropout(p=out_dropout)
        # vision query, key & value projection layers
        self.v2q = nn.Linear(self.v_dim, self.embed_dim)
        self.v2k = nn.Linear(self.v_dim, self.embed_dim)
        self.v2v = nn.Linear(self.v_dim, self.embed_dim)
        # language query, key & value projection layers
        self.l2q = nn.Linear(self.l_dim, self.embed_dim)
        self.l2k = nn.Linear(self.l_dim, self.embed_dim)
        self.l2v = nn.Linear(self.l_dim, self.embed_dim)
        # vision & language output projection layers
        self.v2out = nn.Linear(self.embed_dim, self.v_dim)
        self.l2out = nn.Linear(self.embed_dim, self.l_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform(self.v2q.weight)
        self.v2q.bia.data.fill_(0)
        nn.init.xavier_uniform(self.v2k.weight)
        self.v2k.bia.data.fill_(0)
        nn.init.xavier_uniform(self.v2v.weight)
        self.v2v.bia.data.fill_(0)
        nn.init.xavier_uniform(self.l2q.weight)
        self.l2q.bia.data.fill_(0)
        nn.init.xavier_uniform(self.l2k.weight)
        self.l2k.bia.data.fill_(0)
        nn.init.xavier_uniform(self.l2v.weight)
        self.l2v.bia.data.fill_(0)
        nn.init.xavier_uniform(self.v2out.weight)
        self.v2out.bia.data.fill_(0)
        nn.init.xavier_uniform(self.l2out.weight)
        self.l2out.bia.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None, mode='vl'):
        assert(
            mode in ['v', 'l', 'vl']
        ), f"forward mode must be one of v, l or vl, got {mode} instead."
        if mode == 'v':
            assert(v is not None), "vision input cannot be None under vision self attention mode!"
            B, N, C = v.size()
            query_states = self.v2q(v).reshape(B, N, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            key_states = self.v2k(v).reshape(B, N, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            value_states = self.v2v(v).reshape(B, N, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            attn = (query_states @ key_states.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            x = (attn @ value_states).transpose(1, 2).reshape(B, N, self.embed_dim)
            x = self.v2out(x)
            x = self.out_dropout(x)
            return x
        elif mode == 'l':
            assert(l is not None), "language input cannot be None under vision self attention mode!"
            B, N, C = l.size()
            query_states = self.l2q(l).reshape(B, N, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            key_states = self.l2k(l).reshape(B, N, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            value_states = self.l2v(l).reshape(B, N, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)

# TODO: for testing purpose only, delete the code below after layers completed
if __name__ == "__main__":
    bimhattn = BiMultiHeadAttention(v_dim=1024, l_dim=768, embed_dim=768, num_heads=12)

    batchsize = 16
    maximum_text_length = 77
    actual_text_length = 11
    patch = 14
    vision_input = torch.randn([batchsize, patch**2+1, 1024])
    language_input = torch.randn([batchsize, maximum_text_length, 768])
