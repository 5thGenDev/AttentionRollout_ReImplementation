# d_model = token-embedding size
# head_dim = token-embedding size for each head ONLY in the context of Scaling module
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

# Copied from https://github.com/robflynnyh/hydra-linear-attention/blob/main/hydra.py
class HydraAttention(nn.Module):
    def __init__(self, d_model, output_layer='linear', dropout=0.0):
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask=None):
        '''x: (B, T, D)'''
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Tis cosine similarity kernel instead of softmax
        q = q / q.norm(dim=-1, keepdim=True) 
        k = k / k.norm(dim=-1, keepdim=True)
        
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        kv = k * v
        if self.dropout.p > 0:
            kv = self.dropout(kv.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out = kv.sum(dim=-2, keepdim=True) * q
        return self.out(out)
