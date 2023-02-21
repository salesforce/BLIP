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
        self.v2q.bias.data.fill_(0)
        nn.init.xavier_uniform(self.v2k.weight)
        self.v2k.bias.data.fill_(0)
        nn.init.xavier_uniform(self.v2v.weight)
        self.v2v.bias.data.fill_(0)
        nn.init.xavier_uniform(self.l2q.weight)
        self.l2q.bias.data.fill_(0)
        nn.init.xavier_uniform(self.l2k.weight)
        self.l2k.bias.data.fill_(0)
        nn.init.xavier_uniform(self.l2v.weight)
        self.l2v.bias.data.fill_(0)
        nn.init.xavier_uniform(self.v2out.weight)
        self.v2out.bias.data.fill_(0)
        nn.init.xavier_uniform(self.l2out.weight)
        self.l2out.bias.data.fill_(0)

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
            attn = self.attn_dropout(attn)
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
            attn = (query_states @ key_states.transpose(-2, -1)) * self.scale
            attn = self.attn_dropout(attn)
            if attention_mask_l is not None:
                attn = attn + attention_mask_l.reshape(1, 1, 1, -1)
            attn = attn.softmax(dim=-1)
            x = (attn @ value_states).transpose(1, 2).reshape(B, N, self.embed_dim)
            x = self.l2out(x)
            return x
        elif mode == 'vl':
            assert(
                v is not None and l is not None
            ), "both vision & language input cannot be None under VL cross attention mode!"
            assert(v.size(0) == l.size(0)), "inputs must have the same batch size!"
            B, vN, vC = v.size()
            _, lN, lC = l.size()
            query_states = self.v2q(v).reshape(B, vN, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            key_states = self.l2k(l).reshape(B, lN, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            vision_value_states = self.v2v(v).reshape(B, vN, self.num_heads,
                                                      self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            language_value_states = self.l2v(l).reshape(B, lN, self.num_heads,
                                                        self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
            attn = (query_states @ key_states.transpose(-2, -1)) * self.scale
            attn = self.attn_dropout(attn)
            attn_v = attn
            attn_l = attn.transpose(-2, -1)
            if attention_mask_l is not None:
                attn_v = attn_v + attention_mask_l.reshape(1, 1, 1, -1)
            attn_l = attn_l.softmax(dim=-1)
            attn_v = attn_v.softmax(dim=-1)
            xl = (attn_l @ vision_value_states).transpose(1, 2).reshape(B, lN, self.embed_dim)
            xv = (attn_v @ language_value_states).transpose(1, 2).reshape(B, vN, self.embed_dim)
            xl = self.l2out(xl)
            xv = self.v2out(xv)
            xv = self.out_dropout(xv)

            return xv, xl



# TODO: for testing purpose only, delete the code below after layers completed
if __name__ == "__main__":
    bimhattn = BiMultiHeadAttention(v_dim=1024, l_dim=768, embed_dim=768, num_heads=12)
    # define input shape
    batchsize = 16
    maximum_text_length = 77
    actual_text_length = 11
    patch = 14
    # virtual vision input
    vision_input = torch.randn([batchsize, patch**2+1, 1024])
    # virtual language input
    language_input = torch.randn([batchsize, maximum_text_length, 768])
    language_mask = torch.zeros([maximum_text_length])
    language_mask[actual_text_length:] = -1e10
    # vision self attention test
    output_v = bimhattn(vision_input, None, mode='v')
    # language self attention test
    output_l = bimhattn(None, language_input, language_mask, mode='l')
    fused_v, fused_l = bimhattn(vision_input, language_input, language_mask, mode='vl')

