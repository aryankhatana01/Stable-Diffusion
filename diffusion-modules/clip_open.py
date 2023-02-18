import torch.nn as nn
import torch
import numpy as np

class CLIPMLP(nn.Module):
  def __init__(self):
    super(CLIPMLP, self).__init__()
    self.fc1 = nn.Linear(768, 3072)
    self.fc2 = nn.Linear(3072, 768)

  def forward(self, hidden_states):
    hidden_states = self.fc1(hidden_states)
    hidden_states = nn.GELU()(hidden_states)
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class CLIPAttention(nn.Module):
  def __init__(self):
    super(CLIPAttention, self).__init__()
    self.embed_dim = 768
    self.num_heads = 12
    self.head_dim = self.embed_dim // self.num_heads
    self.scale = self.head_dim**-0.5
    self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

  def _shape(self, tensor, seq_len: int, bsz: int):
    return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

  def forward(self, hidden_states, causal_attention_mask):
    bsz, tgt_len, embed_dim = hidden_states.shape

    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).reshape(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    src_len = key_states.shape[1]
    value_states = value_states.reshape(*proj_shape)

    attn_weights = query_states @ key_states.permute(0,2,1)

    attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
    attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = attn_weights.softmax()

    attn_output = attn_weights @ value_states

    attn_output = attn_output.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.permute(0,2,1,3)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)
    return attn_output

class CLIPEncoderLayer(nn.Module):
  def __init__(self):
    super(CLIPEncoderLayer, self).__init__()
    self.self_attn = CLIPAttention()
    self.layer_norm1 = nn.LayerNorm(768)
    self.mlp = CLIPMLP()
    self.layer_norm2 = nn.LayerNorm(768)

  def forward(self, hidden_states, causal_attention_mask):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states = self.self_attn(hidden_states, causal_attention_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states

class CLIPEncoder(nn.Module):
  def __init__(self):
    super(CLIPEncoder, self).__init__()
    self.layers = [CLIPEncoderLayer() for _ in range(12)]

  def forward(self, hidden_states, causal_attention_mask):
    for l in self.layers:
      hidden_states = l(hidden_states, causal_attention_mask)
    return hidden_states

class CLIPTextEmbeddings(nn.Module):
  def __init__(self):
    super(CLIPTextEmbeddings, self).__init__()
    self.position_ids = torch.empty(1, 77)  # what is this?
    self.token_embedding = {"weight": torch.empty(49408, 768)}
    self.position_embedding = {"weight": torch.empty(77, 768)}

  def forward(self, input_ids, position_ids):

    inputs = np.zeros((1, len(input_ids), 49408))
    positions = np.zeros((1, len(position_ids), 77))
    for i,x in enumerate(input_ids): inputs[0][i][x] = 1
    for i,x in enumerate(position_ids): positions[0][i][x] = 1
    inputs_embeds = torch.tensor(inputs, device=self.token_embedding['weight'].device) @ self.token_embedding['weight']
    position_embeddings = torch.tensor(positions, device=self.position_embedding['weight'].device) @ self.position_embedding['weight'] 
    return inputs_embeds + position_embeddings

class CLIPTextTransformer(nn.Module):
  def __init__(self):
    super(CLIPTextTransformer, self).__init__()
    self.embeddings = CLIPTextEmbeddings()
    self.encoder = CLIPEncoder()
    self.final_layer_norm = nn.LayerNorm(768)

  def forward(self, input_ids):
    x = self.embeddings(input_ids, list(range(len(input_ids))))
    causal_attention_mask = np.triu(np.ones((1,1,77,77), dtype=np.float32) * -np.inf, k=1)
    x = self.encoder(x, torch.tensor(causal_attention_mask, device=x.device))
    return self.final_layer_norm(x)
