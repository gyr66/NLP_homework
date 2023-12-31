import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForRelationExtraction(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = len(config.label2id)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size)
        self.classifier = nn.Linear(2 * config.hidden_size, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
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

        sequence_output = self.dropout(sequence_output)

        e1_start = torch.where(input_ids == self.config.e1_start_token_id)
        e2_start = torch.where(input_ids == self.config.e2_start_token_id)

        e1_hidden_states = sequence_output[e1_start[0], e1_start[1]]
        e2_hidden_states = sequence_output[e2_start[0], e2_start[1]]

        h = torch.cat((e1_hidden_states, e2_hidden_states), dim=-1)
        logits = self.classifier(self.layer_norm(h))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]  # Need to check outputs shape
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
