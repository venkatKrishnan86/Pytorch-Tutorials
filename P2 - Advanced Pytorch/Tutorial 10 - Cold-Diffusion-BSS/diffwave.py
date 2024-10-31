import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.swish_beta = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        
    def forward(self, x):
        return x * torch.sigmoid(self.swish_beta * x)

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
            Arguments
            ---------
            x: torch.Tensor
                The input tensor - Shape: (batch_size, in_channels, audio_length)
                
            Returns
            -------
            out: torch.Tensor
                The output tensor - Shape: (batch_size, out_channels, audio_length)
        """
        return self.norm(self.conv(x))

class BiDilatedConv1d(nn.Module):
    def __init__(
        self, 
        dilation,
        in_channels,
        kernel_size = 3, 
        stride = 1
    ):
        super(BiDilatedConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels*2, kernel_size, stride, padding = "same", dilation=dilation)
    
    def forward(self, x):
        """
            Arguments
            ---------
            x: torch.Tensor
                The input tensor - Shape: (batch_size, channels, audio_length)
            
            Returns
            -------
            out: torch.Tensor
                The output tensor - Shape: (batch_size, 2*channels, audio_length)
        """
        out = self.conv1(x)
        return out


class ResidualLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        time_embedding_dim,
        layer_number,
        num_groups = 10
    ):
        super(ResidualLayer, self).__init__()
        self.time_fc = nn.Linear(time_embedding_dim, in_channels)
        self.bi_dilated_conv = BiDilatedConv1d(dilation = (layer_number%num_groups)+1, in_channels = in_channels)

        # No condition in our case
        self.residual_conv = Conv1x1(in_channels, in_channels)
        self.skip_conv = Conv1x1(in_channels, in_channels)
    
    def forward(self, x, time_embedding):
        """
            Arguments
            ---------
            x: torch.Tensor
                The input tensor - Shape: (batch_size, channels, audio_length)
            
            time_embedding: torch.Tensor
                Transformed time step indices - Shape: (batch_size, embedding_dim)
            
            Returns
            -------
            out: torch.Tensor
                The output tensor - Shape:
            
            skip: torch.Tensor
                The skip connection tensor - Shape:
        """
        time_emb = self.time_fc(time_embedding)[:, :, None].repeat(1, 1, x.shape[-1])
        out = x + time_emb                                  # Shape: (batch_size, channels, audio_length)
        out = self.bi_dilated_conv(out)                     # Shape: (batch_size, 2*channels, audio_length)
        out1 = torch.tanh(out[:, :out.shape[1]//2, :])      # Shape: (batch_size, channels, audio_length)
        out2 = torch.sigmoid(out[:, out.shape[1]//2:, :])   # Shape: (batch_size, channels, audio_length)
        out = torch.mul(out1, out2)                         # Shape: (batch_size, channels, audio_length)

        residual = self.residual_conv(out)                  # Shape: (batch_size, channels, audio_length)
        skip = self.skip_conv(out)                          # Shape: (batch_size, channels, audio_length)
        
        return x + residual, skip
    


class DiffWave(nn.Module):
    def __init__(
        self,
        time_embedding_dim,
        in_channels,
        mid_channels,
        residual_input_dim = 512,
        n_residual_layers = 30,
        device = "cpu"
    ):
        super(DiffWave, self).__init__()
        self.embedding_dim = time_embedding_dim
        self.n_residual_layers = n_residual_layers
        self.device = device

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.relu = nn.ReLU()

        self.time_fc1 = nn.Linear(self.embedding_dim, residual_input_dim, device=device)
        self.swish1 = Swish()
        self.time_norm1 = nn.LayerNorm(residual_input_dim)
        self.time_fc2 = nn.Linear(residual_input_dim, residual_input_dim, device=device)
        self.swish2 = Swish()

        self.residual_layers = [
            ResidualLayer(
                mid_channels, 
                residual_input_dim, 
                layer_num, 
                num_groups=10
            ).to(device) for layer_num in range(n_residual_layers)
        ]

        self.end_conv1 = Conv1x1(mid_channels, mid_channels)
        self.end_conv2 = Conv1x1(mid_channels, in_channels)
    
    def pos_encoding(
        self, 
        t: torch.Tensor,
        F = 4
    ):
        """
            Arguments
            ---------
            t: torch.Tensor
                The time step indices - Shape: (batch_size, )
            
            channels: int
                The number of channels in the input tensor/audio signal
        """
        t = t[:, None].repeat(1, self.embedding_dim//2).to(self.device)                             # Shape: (batch_size, embedding_dim//2)
        freqs = 10**((torch.arange(0, self.embedding_dim//2).float() * F)/(self.embedding_dim//2))  # Shape: (embedding_dim//2, )
        freqs = freqs[None, :].to(self.device)                                                      # Shape: (1, embedding_dim//2)
        pos_enc_sin = torch.sin(torch.mul(t, freqs))                                                # Shape: (batch_size, embedding_dim//2)
        pos_enc_cos = torch.cos(torch.mul(t, freqs))                                                # Shape: (batch_size, embedding_dim//2)
        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)                                     # Shape: (batch_size, embedding_dim)
        return pos_enc

    def forward(self, x, t):
        """
            Arguments
            ---------
            x: torch.Tensor
                The input tensor - Shape: (batch_size, in_channels, audio_length)
            
            t: torch.Tensor
                The time step indices - Shape: (batch_size, )

            Returns
            -------
            pred_noise: torch.Tensor
                The output tensor - Shape: (batch_size, in_channels, audio_length)
        """ 
        # INPUT: (batch_size, in_channels, audio_length)
        x = self.conv1(x)
        x = self.relu(x)                                                       # (batch_size, mid_channels, audio_length)

        # Time embeddings
        t_emb = self.pos_encoding(t).to(x.device)                                            # (batch_size, embedding_dim)
        t_emb = self.swish1(self.time_fc1(t_emb))
        residual_input = self.swish2(self.time_fc2(self.time_norm1(t_emb)))     # (batch_size, residual_input_dim)

        # Residual layers
        skip_connections = torch.zeros((self.n_residual_layers, x.shape[0], x.shape[1], x.shape[2])).to(x.device)
        for layer_num in range(self.n_residual_layers):
            x, skip = self.residual_layers[layer_num](x, residual_input)
            skip_connections[layer_num] = skip

        skip_connections = torch.sum(skip_connections, dim=0)                  # (batch_size, mid_channels, audio_length)
        x = torch.relu(self.end_conv1(skip_connections))                       # (batch_size, mid_channels, audio_length)
        x = self.end_conv2(x)                                                  # (batch_size, in_channels, audio_length)
        
        return x
    
if __name__ == "__main__":
    batch_size = 4
    in_channels = 2
    audio_length = 16000*4
    time_embedding_dim = 128
    mid_channels = 64
    residual_input_dim = 512
    n_residual_layers = 30
    noise_steps = 20
    device = "cpu"

    model = DiffWave(
        time_embedding_dim=time_embedding_dim,
        in_channels=in_channels,
        mid_channels=mid_channels,
        residual_input_dim=residual_input_dim,
        n_residual_layers=n_residual_layers,
        device=device
    ).to(device)

    x = torch.rand((batch_size, in_channels, audio_length)).to(device)
    t = torch.randint(0, noise_steps, (batch_size,)).to(device)

    with torch.no_grad():
        pred_noise = model(x, t)
    
    print(pred_noise.shape)
    print("Number of parameters:", sum([p.numel() for p in model.parameters()]))