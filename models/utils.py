
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn import LayerNorm

def default(val, d):
    """
    Return the value if it exists, otherwise return a default value.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        The value if it exists, otherwise the default value.
    """
    return val if val else d

def exists(val):
    """
    Check if the value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None

def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Query 仍然是多头的
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # Key 和 Value 只有一个（共享）
        self.k_proj = nn.Linear(embed_dim, embed_dim // num_heads)
        self.v_proj = nn.Linear(embed_dim, embed_dim // num_heads)

        # 输出层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size, seq_length, _ = query.shape

        # 计算 Q，仍然是 num_heads 份
        Q = self.q_proj(query).reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        # 计算 K, V（但只计算一次，供所有头共享）
        K = self.k_proj(key).unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        V = self.v_proj(value).unsqueeze(1)  # [batch, 1, seq_len, head_dim]

        # 计算 Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 计算最终的 Value 加权和
        attn_output = torch.matmul(attn_weights, V)

        # 变换回输出形状
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        output = self.out_proj(attn_output)
        return output

def SimpleFeedForward(dim: int, hidden_dim: int, dropout=0.1):
    """
    Feedforward neural network with LayerNorms and GELU activations


    Flow:
    layer_norm -> linear -> gelu -> linear -> dropout


    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        dropout (float): Dropout probability

    Usage:
    >>> model = SimpleFeedForward(768, 2048, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape
    torch.Size([1, 768])
    """
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        dropout=0.0,
        norm_context=False,
        cosine_sim=False,
        cosine_sim_scale=16,
    ):
        """
        CrossAttention module performs cross-attention mechanism between input tensor `x` and context tensor `context`.

        Args:
            dim (int): The dimension of the input tensor `x`.
            context_dim (int, optional): The dimension of the context tensor `context`. If not provided, it defaults to `dim`.
            dim_head (int, optional): The dimension of each head in the multi-head attention. Defaults to 64.
            heads (int, optional): The number of attention heads. Defaults to 8.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            norm_context (bool, optional): Whether to apply layer normalization to the context tensor. Defaults to False.
            cosine_sim (bool, optional): Whether to use cosine similarity for attention calculation. Defaults to False.
            cosine_sim_scale (int, optional): The scale factor for cosine similarity. Defaults to 16.
        """
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = (
            LayerNorm(context_dim) if norm_context else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, dim).
            context (torch.Tensor): The context tensor of shape (batch_size, context_length, context_dim).
            mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, dim).
        """
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (
            self.to_q(x),
            *self.to_kv(context).chunk(2, dim=-1),
        )

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(
            lambda t: repeat(t, "d -> b h 1 d", h=self.heads, b=b),
            self.null_kv.unbind(dim=-2),
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SKNetandLSTM_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.convs = nn.ModuleList([])
        features = 1
        M = 2
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(3 + i * 2,3 + i * 2),
                              stride=(1,1),
                              padding=1 + i
                              ), nn.BatchNorm2d(1)#,
                    #nn.ReLU(inplace=False))
                ))

        self.fc = nn.Linear(1, 32)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(32, 1))
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=768,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False
                            )

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768 * 2, 768),
        )



    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            fea = conv(tokens).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v.squeeze_(dim=1)
        fea_v = fea_v[:, -1, :]
        rnn_outputs, _ = self.lstm(raw_outputs.last_hidden_state)
        rnn_outputs = rnn_outputs[:, -1, :]
        out = torch.cat((fea_v, rnn_outputs),dim=1)
        out = self.block(out)
        return out

class SKNet_LSTM_Attention_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.convs = nn.ModuleList([])
        dim_att = 768
        self.key_layer = nn.Linear(dim_att, dim_att)
        self.query_layer = nn.Linear(dim_att, dim_att)
        self.value_layer = nn.Linear(dim_att, dim_att)
        self._norm_fact = 1 / math.sqrt(dim_att)

        features = 1
        M = 2
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(3 + i * 2,3 + i * 2),
                              stride=(1,1),
                              padding=1 + i
                              ), nn.BatchNorm2d(1)#,
                    #nn.ReLU(inplace=False))
                ))

        self.fc = nn.Linear(1, 32)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(32, 1))
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=768,
                            num_layers=1,
                            batch_first=True,
                            )
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768 * 2,768),

        )

        dims = 1288 * 8
        self.heads = 8
        drop_out_num = 0.0
        self.multi_head_attention = torch.nn.MultiheadAttention(embed_dim=dims, num_heads=self.heads)



    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state


        # attention_output = torch.mean(attention_output, dim=1)

        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)


        sknet_tokens = attention_output.unsqueeze(dim=1)

        for i, conv in enumerate(self.convs):
            fea = conv(sknet_tokens).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)

        fea_v.squeeze_(dim=1)


        rnn_tokens = attention_output
        rnn_outputs, _ = self.lstm(rnn_tokens)


        rnn_outputs = rnn_outputs[:, -1, :]
        fea_v = fea_v[:, -1, :]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # out = self.LAFF(fea_v.unsqueeze(1), rnn_outputs.unsqueeze(1))
        out = torch.cat((fea_v, rnn_outputs), 1)
        
        pout = self.block(out)
        return pout

class AxModel(nn.Module):
    def __init__(self, basemode):
        super().__init__()
        self.base_model = basemode
        self.proj = nn.Linear(768*3,768)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.GELU
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self,pos, neu, neg):
        pos = self.base_model(**pos).last_hidden_state
        neu = self.base_model(**neu).last_hidden_state
        neg = self.base_model(**neg).last_hidden_state
        # s  = torch.concat((pos,neu,neg),dim=-1)
        # out = self.proj(s)

        # putong +
        # out = pos + neu + neg
        # out = self.softmax(out)

        # weighted
        w = torch.softmax(self.weights, dim=0)
        out = w[0] * pos + w[1] * neu + w[2] * neg
        
        return out