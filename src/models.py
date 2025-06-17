import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.utils import logging
from typing import Optional, Tuple, Union


logger = logging.get_logger(__name__)


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_vars=conf.n_vars,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "gpt2_sde":
        model = SDETransformer(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_vars=conf.n_vars,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "gpt2_ao":
        model = AttentionOnlyTransformer(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_vars=conf.n_vars,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "gpt2_ao_sde":
        model = SDEAttentionOnlyTransformer(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_vars=conf.n_vars,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "gpt2_mlp":
        model = MLPOnlyTransformer(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_vars=conf.n_vars,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "lstm":
        model = LSTMModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
        )
    elif conf.family == "lstm_sde":
        model = LSTMSDEModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
        )
    elif conf.family == "rnn":
        model = RNNModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
        )
    elif conf.family == "rnn_sde":
        model = RNNSDEModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
        )
    elif conf.family == "gru":
        model = GRUModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
        )
    elif conf.family == "gru_sde":
        model = GRUSDEModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
        )
    else:
        raise NotImplementedError

    return model


class AttentionOnlyGPT2Block(GPT2Block):
    def __init__(self, conf, layer_idx = None):
        super().__init__(conf, layer_idx)
        self.mlp = None

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        # skip MLP
        # residual connection
        hidden_states = residual + hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    

class MLPOnlyGPT2Block(GPT2Block):
    def __init__(self, conf, layer_idx = None):
        super().__init__(conf, layer_idx)
        self.attn = None
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        
        # residual connection
        # layer-norm is at beginning of block and after processing entire block
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class AttentionOnlyGPT2Model(GPT2Model):
    def __init__(self, conf):
        super().__init__(conf)
        self.h = nn.ModuleList([AttentionOnlyGPT2Block(conf, layer_idx=i) for i in range(conf.num_hidden_layers)])


class MLPOnlyGPT2Model(GPT2Model):
    def __init__(self, conf):
        super().__init__(conf)
        self.h = nn.ModuleList([MLPOnlyGPT2Block(conf, layer_idx=i) for i in range(conf.num_hidden_layers)])


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_vars, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions= (n_positions + 1) * n_vars + 1,    # (n_positions + 1) as we only have one counterfactual example, + 1 for index token Z
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_embd = n_embd
        self.n_positions = n_positions
        self.n_vars = n_vars
        self.n_dims = n_dims
        self.o_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)

    def forward(self, data, o_vars = None, inds = None, output_attentions=False, output_hidden_states=False):
        """
        o_vars:         number of variables used at current iteration: curriculum.n_vars_truncated          <= n_vars
        o_points:       number of in-context examples at current iteration: curriculum.n_points_truncated   <= n_points
        o_positions:    number of total data points (positions) at current iteration                        <= n_positions
                        o_positions = (o_points + 1) * o_vars + 1
        """
        if output_attentions and output_hidden_states: raise NotImplementedError

        if o_vars == None: o_vars = self.n_vars
        if inds is not None: raise NotImplementedError        

        if self.n_dims != self.n_embd: embeds = self._read_in(data)
        else: embeds = data
        if output_attentions or output_hidden_states:
            outputs = self._backbone(inputs_embeds = embeds,
                                     output_attentions=output_attentions, 
                                     output_hidden_states = output_hidden_states)
            output = outputs.last_hidden_state
            attentions = outputs.attentions if output_attentions else None
            hidden_states = outputs.hidden_states if output_hidden_states else None
        else: output = self._backbone(inputs_embeds=embeds).last_hidden_state
        if self.n_dims != self.n_embd: prediction = self._read_out(output)
        else: prediction = output
        
        pred = prediction[:, -2, :]
        gt = data[:, -1, :]

        if output_attentions: return pred, gt, attentions
        elif output_hidden_states: return pred, gt, hidden_states, self._read_out
        else: return pred, gt


class SDETransformer(TransformerModel):
    def __init__(self, n_dims, n_positions, n_vars, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions= 2 * n_positions * n_vars + 1,   # 2 * for counterfactual
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_sde_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_vars = n_vars
        self.n_dims = n_dims
        self.o_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)

    def forward(self, data, o_vars = None, inds = None, output_attentions = False):
        """
        o_vars:         number of variables used at current iteration: curriculum.n_vars_truncated          <= n_vars
        o_points:       number of in-context examples at current iteration: curriculum.n_points_truncated   <= n_points
        """
        if o_vars == None: o_vars = self.n_vars
        if inds is not None: raise NotImplementedError

        _, o_positions, _ = data.shape

        # block setup obs, obs, ..., obs, cf, cf, ..., cf
        assert ((o_positions - 1) / (2 * o_vars)).is_integer()
        o_points = int((o_positions - 1) / (2 * o_vars))
        gt_inds = torch.arange(o_points * o_vars + 1 + o_vars, o_positions)
        pred_inds = gt_inds - 1
        
        embeds = self._read_in(data)
        if output_attentions:
            outputs = self._backbone(inputs_embeds = embeds,
                                     output_attentions=output_attentions)
            output = outputs.last_hidden_state
            attentions = outputs.attentions if output_attentions else None
        else: output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        
        pred = prediction[:, pred_inds, :]
        gt = data[:, gt_inds, :]

        if output_attentions: return pred, gt, attentions
        else: return pred, gt


class AttentionOnlyTransformer(TransformerModel):
    def __init__(self, n_dims, n_positions, n_vars, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions= (n_positions + 1) * n_vars + 1,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_ao_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_vars = n_vars
        self.n_dims = n_dims
        self.o_dims = n_dims
        self.n_embd = n_embd
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = AttentionOnlyGPT2Model(configuration)     # Attention-only GPT2
        self._read_out = nn.Linear(n_embd, n_dims)


class SDEAttentionOnlyTransformer(SDETransformer):
    def __init__(self, n_dims, n_positions, n_vars, n_embd=128, n_layer=12, n_head=4):
        super(SDETransformer, self).__init__(n_dims = n_dims, n_positions = n_positions, n_vars = n_vars)
        configuration = GPT2Config(
            n_positions= 2 * n_positions * n_vars + 1,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_ao_sde_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_vars = n_vars
        self.n_dims = n_dims
        self.o_dims = n_dims
        self.n_embd = n_embd
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = AttentionOnlyGPT2Model(configuration)     # Attention-only GPT2
        self._read_out = nn.Linear(n_embd, n_dims)
        

class MLPOnlyTransformer(TransformerModel):
    def __init__(self, n_dims, n_positions, n_vars, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions= (n_positions + 1) * n_vars + 1,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_mlp_embd={n_embd}_layer={n_layer}"

        self.n_positions = n_positions
        self.n_vars = n_vars
        self.n_dims = n_dims
        self.o_dims = n_dims
        self.n_embd = n_embd
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = MLPOnlyGPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)


class LSTMModel(nn.Module):
    def __init__(self, n_dims, n_embd, n_layer):
        super(LSTMModel, self).__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_dims = n_dims

        self.name = f"lstm_embd={n_embd}_layer={n_layer}"

        self._backbone = nn.LSTM(
            input_size=n_dims,
            hidden_size=self.n_embd,
            num_layers=self.n_layer,
            batch_first=True,
            dropout=0.0,        # as we have fresh data every iteration
            bidirectional=False,
        )
        self._read_out = nn.Linear(self.n_embd, n_dims)


    def forward(self, data, o_vars):
        h0 = torch.zeros(self.n_layer, data.size(0), self.n_embd).to(data.device)
        c0 = torch.zeros(self.n_layer, data.size(0), self.n_embd).to(data.device)

        out, _ = self._backbone(data[:, :-1, :], (h0, c0))
        pred = self._read_out(out[:, -1, :])
        gt = data[:, -1, :]
        
        return pred, gt
    

class LSTMSDEModel(LSTMModel):
    def forward(self, data, o_vars):
        n_thetas, o_positions, _ = data.shape

        assert ((o_positions - 1) / (2 * o_vars)).is_integer()
        o_points = int((o_positions - 1) / (2 * o_vars))
        gt_inds = torch.arange(o_points * o_vars + 1 + o_vars, o_positions)
        pred_inds = gt_inds - 1

        h0 = torch.zeros(self.n_layer, n_thetas, self.n_embd).to(data.device)
        c0 = torch.zeros(self.n_layer, n_thetas, self.n_embd).to(data.device)

        out, _ = self._backbone(data, (h0, c0))
        pred = self._read_out(out[:, pred_inds, :])
        gt = data[:, gt_inds, :]
        
        return pred, gt
    

class RNNModel(nn.Module):
    def __init__(self, n_dims, n_embd, n_layer):
        super(RNNModel, self).__init__()
        self.name = f"rnn_embd={n_embd}_layer={n_layer}"

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_dims = n_dims

        self._backbone = nn.RNN(
            input_size=n_dims,
            hidden_size=self.n_embd,
            num_layers=n_layer,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        
        self._read_out = nn.Linear(self.n_embd, n_dims)

    def forward(self, data, o_vars = None, output_hidden_states=False):
        h0 = torch.zeros(self.n_layer, data.size(0), self.n_embd).to(data.device)
        
        out, _ = self._backbone(data[:, :-1, :], h0)
        pred = self._read_out(out[:, -1, :])
        gt = data[:, -1, :]
        
        return pred, gt


class RNNSDEModel(RNNModel):
    def forward(self, data, o_vars = None, output_hidden_states=False):
        n_thetas, o_positions, _ = data.shape
        assert ((o_positions - 1) / (2 * o_vars)).is_integer()
        o_points = int((o_positions - 1) / (2 * o_vars))
        gt_inds = torch.arange(o_points * o_vars + 1 + o_vars, o_positions)
        pred_inds = gt_inds - 1

        h0 = torch.zeros(self.n_layer, n_thetas, self.n_embd).to(data.device)
        
        out, _ = self._backbone(data, h0)
        pred = self._read_out(out[:, pred_inds, :])
        gt = data[:, gt_inds, :]
        
        return pred, gt


class GRUModel(nn.Module):
    def __init__(self, n_dims, n_embd, n_layer):
        super(GRUModel, self).__init__()

        self.name = f"rnn_embd={n_embd}_layer={n_layer}"

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_dims = n_dims

        self._backbone = nn.GRU(
            input_size = n_dims,
            hidden_size=self.n_embd,
            num_layers=self.n_layer,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        self._read_out = nn.Linear(self.n_embd, self.n_dims)

    def forward(self, data, o_vars):
        h0 = torch.zeros(self.n_layer, data.size(0), self.n_embd).to(data.device)

        out, _ = self._backbone(data[:, :-1, :], h0)
        pred = self._read_out(out[:, -1, :])
        gt = data[:, -1, :]

        return pred, gt


class GRUSDEModel(GRUModel):
    def forward(self, data, o_vars):
        n_thetas, o_positions, _ = data.shape

        assert ((o_positions - 1) / (2 * o_vars)).is_integer()
        o_points = int((o_positions - 1) / (2 * o_vars))
        gt_inds = torch.arange(o_points * o_vars + 1 + o_vars, o_positions)
        pred_inds = gt_inds - 1

        h0 = torch.zeros(self.n_layer, n_thetas, self.n_embd).to(data.device)

        out, _ = self._backbone(data, h0)
        pred = self._read_out(out[:, pred_inds, :])
        gt = data[:, gt_inds, :]

        return pred, gt