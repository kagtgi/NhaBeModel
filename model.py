import torch
import torch.nn as nn
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialPositionalEmbedding
from huggingface_hub import PyTorchModelHubMixin

class ConvLSTM2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super(ConvLSTM2D, self).__init__()
        self.hidden_dim = hidden_dim
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTMModel(
    nn.Module,
    PyTorchModelHubMixin, 
    repo_url="https://github.com/kagtgi/ACOMPA_weather",
    license="mit",
):
    def __init__(self):
        super(ConvLSTMModel, self).__init__()
        
        self.conv_lstm1 = ConvLSTM2D(input_dim=5, hidden_dim=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv_lstm2 = ConvLSTM2D(input_dim=32, hidden_dim=16, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv_lstm3 = ConvLSTM2D(input_dim=16, hidden_dim=5, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(5)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv3d_1 = nn.Conv3d(in_channels=5, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(16)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        
        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=5, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn5 = nn.BatchNorm3d(5)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        
        self.axial_pos_emb = AxialPositionalEmbedding(dim=5, shape=(128, 128))
        self.axial_attn = AxialAttention(dim=5, heads=5, dim_index=1)
        self.layer_norm = nn.LayerNorm((5,128,128))
        self.conv3d_j = nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(10)
        self.final_conv3d = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.out = nn.Sigmoid()
        
       
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Initialize hidden state for first ConvLSTM layer
        h1, c1 = self.conv_lstm1.init_hidden(batch_size, (height, width))
        x, (h1, c1) = self.conv_lstm1(x, (h1, c1))
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Initialize hidden state for second ConvLSTM layer
        h2, c2 = self.conv_lstm2.init_hidden(batch_size, (height, width))
        x, (h2, c2) = self.conv_lstm2(x, (h2, c2))
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Initialize hidden state for third ConvLSTM layer
        h3, c3 = self.conv_lstm3.init_hidden(batch_size, (height, width))
        x, (h3, c3) = self.conv_lstm3(x, (h3, c3))
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
         # Add a singleton dimension to represent depth for Conv3d
        x1 = x.unsqueeze(2)  # shape becomes (batch_size, 5, 1, 128, 128)
        
        x = self.conv3d_1(x1)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        
        x = self.conv3d_2(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        
        # Remove the singleton depth dimension
        x = x.squeeze(2)  # shape becomes (batch_size, 5, 128, 128)
        # Apply positional embedding and axial attention
        x = self.axial_pos_emb(x)
        x = self.axial_attn(x)
        x = self.layer_norm(x)
        x = self.relu5(x)
        
        # Add a singleton dimension to represent depth for final Conv3d
        x2 = x.unsqueeze(2)  # shape becomes (batch_size, 5, 1, 128, 128)
        
        # Concatenate x1 and x2 along the channel dimension
        x_concat = torch.cat([x1, x2], dim=1)  # shape becomes (batch_size, 10, 1, 128, 128)
        
        # Apply final Conv3d to reduce channels to 1
        x = self.conv3d_j(x_concat)
        x = self.bn6(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        
        x = self.final_conv3d(x)
        x = self.out(x)
        # Remove the singleton depth dimension
        x = x.squeeze(2)  # shape becomes (batch_size, 1, 128, 128)

        return x

