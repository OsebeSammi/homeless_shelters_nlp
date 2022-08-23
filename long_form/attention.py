import math
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
import sys
import json

param_path = str(sys.argv[1])
with open(param_path, "r") as file:
    parameters = json.load(file)

MODE = parameters["mode"]
POOL = parameters["pool"]
K = parameters["k"]
# MODE = 0


class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cross_sentence_2(
            self,
            hidden_states: torch.Tensor,
            sep_indices: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            cross_type: int,
            scale: Optional[bool] = False,
            roll_step: Optional[int] = -1
    ) -> Tuple[torch.Tensor]:

        means = torch.zeros_like(hidden_states)
        # separate Questions and Context
        if scale:
            for i, qc_pair in enumerate(hidden_states):
                sep = sep_indices[i][0]
                sep_2 = sep_indices[i][2]
                if POOL == "MIN":
                    means[i, :] = torch.min(hidden_states[i][:sep], 0).values
                elif POOL == "MAX":
                    means[i, :] = torch.max(hidden_states[i][:sep], 0).values
                elif POOL == "SUM":
                    norms = torch.linalg.norm(hidden_states[i][:sep], dim=1)
                    top_indices = torch.argsort(norms, descending=True)
                    for j in top_indices[:int(K * len(top_indices))]:
                        means[i, :].add(hidden_states[i][j])
                # elif POOL == "SUM":
                #     means[i, :] = torch.sum(hidden_states[i][:sep], 0)
                else:
                    means[i, :] = torch.mean(hidden_states[i][:sep], 0)

            hidden_states_diff = hidden_states - means
            hidden_states_add = hidden_states + means
            if cross_type == 0:
                key = self.key(hidden_states_diff)
            else:
                key = self.key(hidden_states_add)

        else:
            # question = torch.ones_like(hidden_states)
            context = torch.clone(hidden_states)
            for i, qc_pair in enumerate(context):
                sep = sep_indices[i][0]
                sep_2 = sep_indices[i][2]
                # question[i][:sep] = hidden_states[i][:sep]     # question
                # question[i][sep:] = hidden_states[i][hidden_states.shape[1]-1]
                # context[i][:sep_2-sep] = hidden_states[i][sep:sep_2]       # context
                # context[i][sep_2 - sep:] = hidden_states[i][hidden_states.shape[1]-1]
                context[i][:sep] = 0
                means[i, :] = torch.mean(hidden_states[i][:sep], 0)

            if cross_type == 0:
                key = self.key(context)
            elif cross_type == 1:
                similarity = torch.nn.functional.cosine_similarity(means, context)
                new_shape = (similarity.shape[0], 1, similarity.shape[1])
                context_cos = torch.mul(context, similarity.reshape(new_shape))
                key = self.key(context_cos)
            else:
                key = self.key(hidden_states)


        # torch.autograd.set_detect_anomaly(True)
        # similarity = torch.nn.functional.cosine_similarity(means, context)
        # similarity = torch.nn.functional.cosine_similarity(means, hidden_states)
        # new_shape = (similarity.shape[0], 1, similarity.shape[1])
        # context_cos = torch.mul(context, similarity.reshape(new_shape))
        # hidden_states_cos = torch.mul(hidden_states, similarity.reshape(new_shape))


        # transform
        query = self.query(hidden_states)
        value = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # to avoid vanishing scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # add mask
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        return torch.matmul(attention_probs, value_layer)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
            sep_indices: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        if MODE == -1:
            context_layer = torch.matmul(attention_probs, value_layer)
        elif MODE >= 0:
            context_layer = self.cross_sentence_2(hidden_states, sep_indices, attention_mask, MODE, parameters["scale"])
        else:
            context_layer = torch.matmul(attention_probs, value_layer) + self.cross_sentence_2(hidden_states,
                                                                                              sep_indices,
                                                                                              attention_mask, 1)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
