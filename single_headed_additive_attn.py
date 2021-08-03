import math
import torch
import torch.nn as nn


def generate_relative_positions_matrix(length, max_relative_positions,
        cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
            min=-max_relative_positions, max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


class AdditiveAttention(nn.Module):
    def __init__(self, model_dim):
        super(AdditiveAttention, self).__init__()
        self.linear_concat = nn.Linear(model_dim * 2, model_dim)
        self.linear_logit = nn.Linear(model_dim, 1)
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
        queries = queries.unsqueeze(dim=2)  # [bs, len_q, 1, size]
        keys = keys.unsqueeze(dim=1)  # [bs, 1, len_k, size]
        q = queries.expand(-1, -1, keys.shape[2], -1)
        k = keys.expand(-1, queries.shape[1], -1, -1)
        combined = torch.tanh(self.linear_concat(torch.cat((q, k), dim=-1)))
        logits = self.linear_logit(combined).squeeze(-1)  # [bs, len_q, len_k]
        if mask is not None:
            mask = torch.logical_not(mask)
            mask.masked_fill_(mask.all(-1, keepdim=True), 0)  # prevent NaN
        logits.masked_fill_(mask, -float('inf'))
        weights = nn.functional.softmax(logits, dim=-1)
        return weights


class SingleHeadedAttention(nn.Module):
    """Applies linear projection over concatenated queries and keys instead of
    dot-product.
    Args:
       model_dim (int): query/key/value hidden size
       dropout (float): proportion of weights dropped out
    """
    def __init__(self, model_dim, max_relative_positions=0):
        super(SingleHeadedAttention, self).__init__()
        self.model_dim = model_dim
        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions
        self.additive_attention = AdditiveAttention(model_dim)

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                    vocab_size, self.model_dim)

            def reset_parameters(self):
                nn.init.normal_(self.linear_keys.weight, std=.02)
        nn.init.normal_(self.linear_values.weight, std=.02)
        nn.init.normal_(self.linear_query.weight, std=.02)
        nn.init.normal_(self.final_linear.weight, std=.02)
        nn.init.constant_(self.linear_keys.bias, 0.)
        nn.init.constant_(self.linear_values.bias, 0.)
        nn.init.constant_(self.linear_query.bias, 0.)
        nn.init.constant_(self.final_linear.bias, 0.)

    def forward(self, key, query, value=None, mask=None, type=None):
        """Compute the context vector and the attention vectors.
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
        key = self.linear_keys(key)
        value = self.linear_values(key if value is None else value)
        query = self.linear_query(query)

        if self.max_relative_positions > 0 and type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                    key_len, self.max_relative_positions,
                    cache=True if layer_cache is not None else False)
            # 1 or key_len x key_len x model_dim
            relations_keys = self.relative_positions_embeddings(
                    relative_positions_matrix.to(key.device))

        out_shape = (key.shape[0], -1, self.model_dim)
        query = query.view(*out_shape) / math.sqrt(self.model_dim)
        return self.additive_attention(query, key.view(*out_shape), value.view(*out_shape), mask)
