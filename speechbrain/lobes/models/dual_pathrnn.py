import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.block_models.modularity import RIM

# from speechbrain.lobes.models.block_models.modularity import SCOFF


# from speechbrain.nnet.RNN import LSTM


EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    """

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )

    def forward(self, x, init_params=True):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    """
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x, init_params=True):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Dual_RNN_Block(nn.Module):
    """
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    """

    def __init__(
        self,
        out_channels,
        hidden_channels,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_spks=2,
    ):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels,
            out_channels,
        )
        self.inter_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels,
            out_channels,
        )

    def forward(self, x):
        """
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        """

        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(
            intra_rnn.contiguous().view(B * S * K, -1)
        ).view(B * S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(
            inter_rnn.contiguous().view(B * S * K, -1)
        ).view(B * K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Module):
    """
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_layers=4,
        K=200,
        num_spks=2,
    ):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_rnn.append(
                Dual_RNN_Block(
                    out_channels,
                    hidden_channels,
                    rnn_type=rnn_type,
                    norm=norm,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x, init_params=True):
        """
           x: [B, N, L]
        """
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Dual_Transformer_Block(nn.Module):
    """
#            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
    """

    def __init__(
        self,
        out_channels,
        num_layers=6,
        nhead=8,
        d_ffn=2048,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        return_attention=False,
        num_modules=1,
        use_group_comm=False,
        norm="ln",
    ):
        super(Dual_Transformer_Block, self).__init__()
        # RNN model
        self.intra_mdl = TransformerEncoder(
            num_layers,
            nhead,
            d_ffn,
            kdim,
            vdim,
            dropout,
            activation,
            return_attention,
            num_modules,
            use_group_comm,
        )
        # self.intra_rnn = LSTM(out_channels, bidirectional=True)

        self.inter_mdl = TransformerEncoder(
            num_layers,
            nhead,
            d_ffn,
            kdim,
            vdim,
            dropout,
            activation,
            return_attention,
            num_modules,
            use_group_comm,
        )

        # getattr(nn, rnn_type)(
        #    out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # self.inter_rnn = getattr(nn, rnn_type)(
        #    out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = Linear(out_channels)
        self.inter_linear = Linear(out_channels)

    def forward(self, x, init_params=True):
        """
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]
        intra_rnn = self.intra_mdl(intra_rnn, init_params=init_params)
        # intra_rnn = self.intra_rnn(intra_rnn, init_params=init_params)
        # [BS, K, N]
        intra_rnn = self.intra_linear(
            intra_rnn.contiguous().view(B * S * K, -1), init_params=init_params
        ).view(B * S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x
        # out = intra_rnn

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter_rnn = self.inter_mdl(inter_rnn, init_params=init_params)
        # [BK, S, N]
        inter_rnn = self.inter_linear(
            inter_rnn.contiguous().view(B * S * K, -1), init_params=init_params
        ).view(B * K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_Transformer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_tf_layers=6,
        num_layers=1,
        nhead=8,
        d_ffn=2048,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        return_attention=False,
        num_modules=1,
        use_group_comm=False,
        norm="ln",
        K=200,
        num_spks=2,
    ):
        super(Dual_Path_Transformer, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                Dual_Transformer_Block(
                    out_channels,
                    num_layers=num_tf_layers,
                    nhead=nhead,
                    d_ffn=d_ffn,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    return_attention=return_attention,
                    num_modules=num_modules,
                    use_group_comm=use_group_comm,
                    norm=norm,
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x, init_params=True):
        """
           x: [B, N, L]
        """
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, init_params=init_params)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Dual_RIM_Block(nn.Module):
    """
#            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
    """

    def __init__(
        self,
        device,
        in_channels,
        out_channels,
        hidden_channels,
        num_units,
        k,
        rnn_type="GRU",
        norm="ln",
        dropout=0.0,
        num_layers=1,
        bidirectional=True,
        version=1,
        attention_out=32,
        num_rules=0,
        rule_time_steps=0,
    ):
        super(Dual_RIM_Block, self).__init__()
        # RNN model
        self.intra_mdl = RIM(
            device,
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_units=num_units,
            k=k,
            rnn_cell=rnn_type,
            n_layers=num_layers,
            bidirectional=bidirectional,
            version=version,
            attention_out=attention_out,
            num_rules=num_rules,
            rule_time_steps=rule_time_steps,
            batch_first=True,
            dropout=dropout,
        )

        self.inter_mdl = RIM(
            device,
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_units=num_units,
            k=k,
            rnn_cell=rnn_type,
            n_layers=num_layers,
            bidirectional=bidirectional,
            version=version,
            attention_out=attention_out,
            num_rules=num_rules,
            rule_time_steps=rule_time_steps,
            batch_first=True,
            dropout=dropout,
        )

        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        self.intra_linear = Linear(out_channels)
        self.inter_linear = Linear(out_channels)

    def forward(self, x, init_params=True):
        """
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_mdl(intra_rnn, init_params=init_params)
        # intra_rnn = self.intra_rnn(intra_rnn, init_params=init_params)
        # [BS, K, N]
        intra_rnn = self.intra_linear(
            intra_rnn.contiguous().view(B * S * K, -1), init_params=init_params
        ).view(B * S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x
        # out = intra_rnn

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_mdl(inter_rnn, init_params=init_params)
        # [BK, S, N]
        inter_rnn = self.inter_linear(
            inter_rnn.contiguous().view(B * S * K, -1), init_params=init_params
        ).view(B * K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RIM(nn.Module):
    def __init__(
        self,
        device,
        in_channels,
        out_channels,
        hidden_channels,
        num_rim_layers=1,
        num_layers=6,
        num_units=4,
        k=4,
        rnn_type="GRU",
        norm="ln",
        dropout=0.0,
        bidirectional=True,
        version=1,
        K=200,
        num_spks=2,
    ):
        super(Dual_Path_RIM, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                Dual_RIM_Block(
                    device,
                    out_channels,
                    out_channels,
                    hidden_channels,
                    num_units=num_units,
                    k=k,
                    rnn_type=rnn_type,
                    norm=norm,
                    dropout=dropout,
                    num_layers=num_rim_layers,
                    bidirectional=bidirectional,
                    version=version,
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x, init_params=True):
        """
           x: [B, N, L]
        """
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, init_params=init_params)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input