import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def rotate_half(x):
    # Split the last dimension into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rope(q, k, cos, sin):
    """
    q, k: [B, H, K, d]
    cos, sin: [1, 1, H, K]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        pos = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

class GPT(nn.Module): 
    def __init__(self, total_token, max_len, d, n_layers):
        super(GPT, self).__init__()
        self.vocab_size = total_token
        self.token_embedding = nn.Embedding(total_token, d)  # [B,T,D]
        self.rope = RoPE(d // 12, max_len)
        self.blocks = nn.ModuleList(
            [Transformer_Block(d, p=0.1, head=12) for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, self.vocab_size, bias=False)
        self.classifier.weight = self.token_embedding.weight # weight tying
        
    def forward(self, input_token): 
        B,T = input_token.shape
        cos, sin = self.rope(T, input_token.device)
        x = self.token_embedding(input_token) # [B,T,D]

        for block in self.blocks:
            x = block(x, cos, sin)
        
        logits = self.classifier(self.ln(x))
        
        return logits

class Transformer_Block(nn.Module): 
    def __init__(self, d, p, head): 
        super(Transformer_Block, self).__init__()

        # LayerNorm from scratch
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        # LayerNorm from nn
        self.ln2 = nn.LayerNorm(d)

        self.d_k = d // head
        self.h = head
        
        self.fc_Q = nn.Linear(d,head * self.d_k)
        self.fc_K = nn.Linear(d,head * self.d_k)
        self.fc_V = nn.Linear(d,head * self.d_k)
        
        self.attn_dropout = nn.Dropout(p)
        self.fc_head = nn.Linear(d, d) 

        self.fc1 = nn.Linear(d, 4*d)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4*d, d)
        self.mlp_dropout = nn.Dropout(p)
        
    def forward(self, x, cos, sin):
        B, T, D = x.shape
        x_res = x
        # --- Manual LayerNorm Implementation (for Interpretability)
        mean = x.mean(dim=-1, keepdim=True)   # [B,T,1]
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # [B,T,1]
        epsilon = 1e-5
        x_norm = (x - mean) / torch.sqrt(var + epsilon) #feature-wise

        gamma = self.gamma
        beta = self.beta
        
        out = gamma * x_norm + beta  
        
        q = self.fc_Q(out).view(B, T, self.h, self.d_k).transpose(1, 2).contiguous() # [B, h, T, d_k]
        k = self.fc_K(out).view(B, T, self.h, self.d_k).transpose(1, 2).contiguous() # [B, h, T, d_k]
        v = self.fc_V(out).view(B, T, self.h, self.d_k).transpose(1, 2).contiguous() # [B, h, T, d_k]

        q, k = rope(q, k, cos, sin)

        mask = torch.triu(torch.ones(T,T), diagonal = 1).unsqueeze(0).unsqueeze(0).bool().to(x.device)
        e = q @ k.transpose(-2,-1) / math.sqrt(self.d_k) # [B,h,T,T]
        e = e.masked_fill(mask, float('-inf')) # causal mask
        
        a = F.softmax(e, dim=-1) # [B,h,T,T]
        a = self.attn_dropout(a)
        
        attention = a @ v  # [B, h, T, d_k]
        attention = attention.transpose(1, 2).contiguous()  # [B, T, h, d_k]
        attention = attention.view(B, T, D)  # [B, T, d]
        
        attention = self.fc_head(attention)
        
        attention += x_res
        x_res = attention

        output = self.mlp_dropout(self.fc2(self.gelu(self.fc1(self.ln2(attention)))))
        output += x_res
        
        return output
