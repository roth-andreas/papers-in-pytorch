import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    """
    Implementation of the Conv LSTM Operation following the paper
    'Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting'
    (https://arxiv.org/abs/1506.04214)

    Similar to a regular convolution, but the gates are defined as convolutions.
    The hidden state is 2 dimensional. Information is only acquired from the receptive field of the conv operations.
    """
    def __init__(self, in_channels, hidden_size=10, kernel_size=3, return_sequences=False):
        """
        Initializes the Convolutional LSTM Block
        Args:
            in_channels: Number of input channels
            hidden_size: Size of the hidden dimension of the LSTM
            kernel_size: kernel size for the Convolution
            return_sequences: Whether to return sequences
        """
        super(ConvLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        # Combine all weights into one big convolution for more efficient computation
        self.gates = nn.Conv2d(in_channels + hidden_size, hidden_size * 4, kernel_size, 1, kernel_size // 2, bias=True)
        self.return_sequences = return_sequences

    def forward(self, inputs, prev_state=None):
        """
        Defines the forward pass through the Conv LSTM
        Args:
            inputs: Sequential Inputs with shape (b, seq_len, channels, w, h)
            prev_state: Optionally pass a previous state as the initial hidden state

        Returns: A single output if return_sequences is False, otherwise returns a sequence of outputs

        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        spatial_size = inputs.shape[3:]
        if prev_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, *spatial_size).to(inputs.device)
            cell_state = torch.zeros(batch_size, self.hidden_size, *spatial_size).to(inputs.device)
        else:
            hidden_state, cell_state = prev_state
        if self.return_sequences:
            hidden_seq = []

        for time_step in range(seq_len):
            input_t = inputs[:, time_step, :]
            stacked = torch.cat((input_t, hidden_state), dim=1)
            gates = self.gates(stacked)
            input_gate, forget_gate, output_gate, cell_gate = gates.chunk(4, 1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            output_gate = torch.sigmoid(output_gate)
            cell_state = forget_gate * cell_state + input_gate * torch.tanh(cell_gate)
            hidden_state = output_gate * torch.tanh(cell_state)

            if self.return_sequences:
                hidden_seq.append(hidden_state.unsqueeze(dim=0))

        if self.return_sequences:
            hidden_seq = torch.cat(hidden_seq, dim=0)
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()
            return hidden_seq, (hidden_state, cell_state)
        else:
            return hidden_state, cell_state
