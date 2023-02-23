import torch
from torch import nn
from timm.models.layers import trunc_normal_, DropPath
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

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CombinerLayer(nn.Module):
    def __init__(
            self,
            v_dim,
            l_dim,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_scale_init_values=0.1,
    ):
        super().__init__()
        self.norm1_v = norm_layer(v_dim)
        self.norm1_l = norm_layer(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim, l_dim=l_dim, embed_dim=dim, num_heads=num_heads, attn_dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_v = norm_layer(v_dim)
        self.norm2_l = norm_layer(l_dim)
        mlp_hidden_dim_v = int(v_dim * mlp_ratio)
        mlp_hidden_dim_l = int(l_dim * mlp_ratio)
        self.mlp_v = Mlp(
            in_features=v_dim,
            hidden_features=mlp_hidden_dim_v,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_l = Mlp(
            in_features=l_dim,
            hidden_features=mlp_hidden_dim_l,
            act_layer=act_layer,
            drop=drop,
        )

        self.gamma_1_v = \
            nn.Parameter(layer_scale_init_values * torch.ones((v_dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0
        self.gamma_1_l = \
            nn.Parameter(layer_scale_init_values * torch.ones((l_dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0
        self.gamma_2_v = \
            nn.Parameter(layer_scale_init_values * torch.ones((v_dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0
        self.gamma_2_l = \
            nn.Parameter(layer_scale_init_values * torch.ones((l_dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0

    def forward(self, vision_input, language_input, attention_mask_l=None, mode='vl'):
        assert (
                mode in ['v', 'l', 'vl']
        ), f"forward mode must be one of v, l or vl, got {mode} instead."
        if mode == 'v':
            vision_input = self.norm1_v(vision_input)
            vision_output = self.attn(vision_input, None, None, mode=mode)
            vision_output = self.drop_path(vision_input + self.gamma_1_v * vision_output)

            vision_output = vision_output + self.drop_path(self.gamma_2_v * self.mlp_v(self.norm2_v(vision_output)))
            return vision_output
        elif mode == 'l':
            language_input = self.norm1_l(language_input)
            language_output = self.attn(None, language_input, attention_mask_l, mode=mode)
            language_output = self.drop_path(language_input + self.gamma_1_l * language_output)

            language_output = language_output + self.drop_path(self.gamma_2_l * self.mlp_l(self.norm2_l(language_output)))
            return language_output
        elif mode == 'vl':
            vision_input, language_input = self.norm1_v(vision_input), self.norm1_l(language_input)
            fused_v, fused_l = self.attn(vision_input, language_input, attention_mask_l, mode='vl')
            fused_v = self.drop_path(vision_input + self.gamma_1_v * fused_v)
            fused_l = self.drop_path(language_input + self.gamma_1_l * fused_l)
            return self.norm2_v(fused_v), self.norm2_l(fused_l)

class CombinerBlock(nn.Module):
    def __init__(
            self,
            v_dim,
            l_dim,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_scale_init_values=0.1,
    ):
        super().__init__()
        self.layer = CombinerLayer(
            v_dim,
            l_dim,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_scale_init_values=0.1,
        )
    def forward(self, vision_input, language_input, language_mask):
        # fusion forward
        v, l = self.layer(vision_input, language_input, language_mask, mode='vl')
        v = self.layer(v, None, None, mode='v')
        l = self.layer(None, l, language_mask, mode='l')
        return v, l




# TODO: for testing purpose only, delete the code below after layers completed
if __name__ == "__main__":
    block = CombinerBlock(v_dim=1024, l_dim=768, dim=768, num_heads=12)
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
    fused_v, fused_l = block(vision_input, language_input, language_mask)
    v_cls = fused_v[:,-1,:]
    l_cls = fused_l[:,-1,:]
    v_proj = nn.Linear(1024, 768)
    l_proj = nn.Linear(768, 768)
    v_cls = v_proj(v_cls)
    l_cls = l_proj(l_cls)
    dummy = (v_cls + l_cls).mean()
    dummy.backward()
    print("test passed")


