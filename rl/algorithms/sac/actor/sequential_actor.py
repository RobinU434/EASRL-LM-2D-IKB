import torch
import torch.nn as nn


class SequentialActor(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()

        self.lstm_cell = nn.LSTMCell(
            input_size=5
            hidden_size=2
            )

    def forward(self, x):
        target = x[:, :2]
        arm_end = x[:, 2:4]
        angles = x[:, 4:]

        batch_size, seq_len, _ = x.size()
        # allocate memory for output
        output = torch.zeros((batch_size, seq_len, self.lstm_cell.hidden_size))
        # iterate through sequence
        for idx in range(seq_len):
            # unsqueeze x
            x_input = x[:, idx, :]
            hx = self.lstm_cell(x_input, hx)
            output[:, idx, :] = hx[0]


