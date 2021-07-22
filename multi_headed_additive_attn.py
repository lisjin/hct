""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn

from misc import generate_relative_positions_matrix,\
                            relative_matmul
from misc import aeq

class Additive_Attention(nn.Module):
    def __init__(self, model_dim):
        super(Additive_Attention, self).__init__()

        self.linear_concat = nn.Linear(model_dim*2,model_dim)
        self.linear_logit = nn.Linear(model_dim,1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_concat.weight, std=.02)
        nn.init.normal_(self.linear_logit.weight, std=.02)
        nn.init.constant_(self.linear_concat.bias, 0.)
        nn.init.constant_(self.linear_logit.bias, 0.)

    def forward(self, queries, keys, values, mask):
        """ Additive attention mechanism. This layer is implemented using a
            one layer feed forward neural network
        :param queries: A tensor with shape [batch, heads, length_q, depth_k]
        :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
        :param values: A tensor with shape [batch, heads, length_kv, depth_v]
        :param bias: A tensor
        :param concat: A boolean value. If ``concat'' is set to True, then
            the computation of attention mechanism is following $tanh(W[q, k])$.
            When ``concat'' is set to False, the computation is following
            $tanh(Wq + Vk)$
        :param keep_prob: a scalar in [0, 1]
        :param dtype: An optional instance of tf.DType
        :param scope: An optional string, the scope of this layer
        :returns: A dict with the following keys:
            weights: A tensor with shape [batch, length_q, length_kv]
        """
        length_q = queries.size(2)
        length_kv = keys.size(2)

        queries = queries.unsqueeze(dim=3) #[bs, 1, len_q, 1, size]
        keys = keys.unsqueeze(dim=2) # [bs, 1, 1, len_k, size]
        q = queries.expand(-1, -1, -1, length_kv, -1)
        k = keys.expand(-1, -1, length_q, -1, -1)

        combined = torch.tanh(self.linear_concat(torch.cat((q, k), dim=-1)))

        # shape: [batch, heads, length_q, length_kv]
        logits = self.linear_logit(combined).squeeze(-1)
        if mask is not None:
            mask = torch.logical_not(mask)
            mask.masked_fill_(mask.all(-1, keepdim=True), 0)
            if len(mask.shape) == len(logits.shape) - 1:
                mask = mask.unsqueeze(1)
        logits.masked_fill_(mask, -float('inf'))
        weights = nn.functional.softmax(logits, dim=-1)
        return weights.squeeze(dim=1)

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions
        self.additive_attention = Additive_Attention(model_dim)

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def reset_parameters(self):
        nn.init.normal_(self.linear_keys.weight, std=.02)
        nn.init.normal_(self.linear_values.weight, std=.02)
        nn.init.normal_(self.linear_query.weight, std=.02)
        nn.init.normal_(self.final_linear.weight, std=.02)
        nn.init.constant_(self.linear_keys.bias, 0.)
        nn.init.constant_(self.linear_values.bias, 0.)
        nn.init.constant_(self.linear_query.bias, 0.)
        nn.init.constant_(self.final_linear.bias, 0.)

    def forward(self, key, value, query, mask=None, type=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """
        assert self.head_count == 1, "We want a single attention distribution, \
                not multiple ones for multiple heads"

        batch_size = key.shape[0]
        final_shape = (batch_size, -1, self.head_count, self.dim_per_head)

        def shape(x):
            """Projection."""
            nonlocal final_shape
            return x.view(*final_shape).transpose(1, 2)

        # 1) Project key, value, and query.
        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)

        if self.max_relative_positions > 0 and type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            # 1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query) / math.sqrt(self.dim_per_head)
        key = shape(key)
        value = shape(value)
        return self.additive_attention(query, key, value, mask)
