from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward
)

from roberta import (
    ROBERTA_START_DOCSTRING,
    RobertaPreTrainedModel,
    RobertaModel,
    ROBERTA_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CONFIG_FOR_DOC,
    get_qa_sep_indices
)

from dataclasses import dataclass

import sys
import json

param_path = str(sys.argv[1])
with open(param_path, "r") as file:
    parameters = json.load(file)

FLAG = True


@dataclass
class QAOutput(QuestionAnsweringModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    has_answer: torch.FloatTensor = None
    sep_indices: Tuple = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Classifier(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(Classifier, self).__init__()  # init call to super class
        self.hidden_reduce = nn.Linear(hidden_size, max_length)
        self.middle = nn.Linear(max_length, 1)
        self.token_reduce = nn.Linear(max_length, 1)

    def forward(self, inputs):
        x = torch.relu(self.hidden_reduce(inputs))
        x = torch.relu(self.middle(x))
        x = x.squeeze()
        x = torch.sigmoid(self.token_reduce(x))
        return x


@add_start_docstrings(
    """
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = Classifier(config.hidden_size, parameters["max_length"])
        # self.span_length = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="roberta-base",
        output_type=QAOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        synonyms: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QAOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        # move to most 'fastest' device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')
        # print("DEVICE", device)
        if device.type != 'cpu':
            self.to(device)
            if input_ids is not None:
                input_ids = input_ids.to(device)  # don't understand documentation says this should be inplace!!
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            if position_ids is not None:
                position_ids = position_ids.to(device)
            if head_mask is not None:
                head_mask = head_mask.to(device)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.to(device)
            if start_positions is not None:
                start_positions = start_positions.to(device)
            if end_positions is not None:
                end_positions = end_positions.to(device)
            if synonyms is not None:
                synonyms = synonyms.to(device)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            synonyms,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        class_logits = self.classifier(sequence_output)
        class_logits = class_logits.view(len(class_logits))
        # len_logits = self.span_length(sequence_output)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # classifier
            loss_bin = BCELoss()
            class_labels = start_positions + end_positions
            class_labels = (class_labels > 0).type(torch.FloatTensor).to(class_logits.device)
            class_loss = loss_bin(class_logits, class_labels)  # to be in similar range as start and end loss

            # span length predictor
            # span_labels = (end_positions - start_positions).to(len_logits.device)
            # span_loss = loss_fct(len_logits, span_labels.view(class_shape))

            # total_loss = (start_loss + end_loss) / 2
            # total_loss = (start_loss + end_loss) / 2 + class_loss
            # total_loss = (start_loss + end_loss) / 2 + class_loss + span_loss

            global FLAG
            if FLAG:
                total_loss = class_loss
                FLAG = False
            else:
                total_loss = (start_loss + end_loss) / 2
                FLAG = True

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # return QuestionAnsweringModelOutput(
        #     loss=total_loss,
        #     start_logits=start_logits,
        #     end_logits=end_logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        sep_indices = get_qa_sep_indices(input_ids)

        return QAOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_answer=class_logits,
            sep_indices=sep_indices,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
