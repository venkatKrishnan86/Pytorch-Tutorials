import torch
from torch import nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(
        self, 
        ma_model: nn.Module, 
        current_model: nn.Module
    ):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(
        self, 
        ema_model: nn.Module,
        model: nn.Module
    ):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(
        self, 
        channels: int, 
        img_size: int,
        ffn_dim: int = 512
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.W_Q = nn.Linear(channels, channels, bias=False)
        self.W_K = nn.Linear(channels, channels, bias=False)
        self.W_V = nn.Linear(channels, channels, bias=False)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, channels),
        )

    def forward(self, x):
        """
            Self-Attention her occurs by converting the entire 2D image to a 1D sequence of pixels 
            and then performing multi-head attention.

            Arguments -
            -----------
            x: Input hidden image; Shape - (batch_size, num_channels, img_size, img_size)

            Returns -
            --------
            attention: Output hidden state; Shape - (batch_size, num_channels, img_size, img_size)

        """
        x = x.view(-1, self.channels, self.img_size * self.img_size).swapaxes(1, 2)     # (batch_size, img_size*img_size, num_channels)
        x_ln = self.ln(x)       # LayerNorm on channels (batch_size, img_size*img_size, num_channels)
        q = self.W_Q(x_ln)                                                  # q: (batch_size, img_size*img_size, num_channels)
        k = self.W_K(x_ln)                                                  # k: (batch_size, img_size*img_size, num_channels)
        v = self.W_V(x_ln)                                                  # v: (batch_size, img_size*img_size, num_channels)
        attention_value, _ = self.mha(q, k, v)                              # attention_value: (batch_size, img_size*img_size, num_channels)
        attention_value = attention_value + x                               # attention_value: (batch_size, img_size*img_size, num_channels)
        attention_value = self.ff_self(attention_value) + attention_value   # attention_value: (batch_size, img_size*img_size, num_channels)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.img_size, self.img_size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
            Arguments -
            -----------
            x: Input hidden image; Shape - (batch_size, in_channels, img_size, img_size)

            Returns -
            --------
            double_conv: Output hidden convolved state; Shape - (batch_size, out_channels, img_size, img_size)

        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        """
            Arguments -
            -----------
            x: Input hidden image; Shape - (batch_size, in_channels, img_size, img_size)
            t: Time step encodings; Shape - (batch_size, embed_dim)

            Returns -
            --------
            x + emb: Output downsampled hidden image; Shape - (batch_size, out_channels, img_size//2, img_size//2)
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None]           # (batch_size, out_channels, 1, 1)
        emb = emb.repeat(1, 1, x.shape[-2], x.shape[-1])    # (batch_size, out_channels, img_size//2, img_size//2)
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, mid_channels = in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        """
            Arguments -
            -----------
            x: Input hidden image; Shape - (batch_size, in_channels, img_size, img_size)
            skip_x: Skip connection coming from ENCODER; Shape - (batch_size, in_channels, img_size*2, img_size*2)
            t: Time step encodings; Shape - (batch_size, embed_dim)

            Returns -
            --------
            x + emb: Output downsampled hidden image; Shape - (batch_size, out_channels, img_size*2, img_size*2)
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None]           # (batch_size, out_channels, 1, 1)
        emb = emb.repeat(1, 1, x.shape[-2], x.shape[-1])    # (batch_size, out_channels, img_size*2, img_size*2)
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # ENCODER
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        
        # BOTTLENECK
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # DECODER
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(self.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq).to(self.device)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq).to(self.device)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
            Arguments -
            -----------
            x: Input image                  ; Shape - (batch_size, num_channels, img_size, img_size)
            t: Time steps input             ; Shape - (batch_size, )

            Returns -
            --------
            output: Predicted noise image   ; Shape - (batch_size, num_channels, img_size, img_size)
        """
        t = t.unsqueeze(-1).type(torch.float).to(self.device)   # t: (batch_size, 1)
        t = self.pos_encoding(t, self.time_dim)                 # t: (batch_size, embed_dim)

        x1 = self.inc(x)                # x1: (batch_size, 64, img_size, img_size)
        x2 = self.down1(x1, t)          # x2: (batch_size, 128, img_size//2, img_size//2)
        x2 = self.sa1(x2)               # x2: (batch_size, 128, img_size//2, img_size//2)
        x3 = self.down2(x2, t)          # x3: (batch_size, 256, img_size//4, img_size//4)
        x3 = self.sa2(x3)               # x3: (batch_size, 256, img_size//4, img_size//4)
        x4 = self.down3(x3, t)          # x4: (batch_size, 256, img_size//8, img_size//8)
        x4 = self.sa3(x4)               # x4: (batch_size, 256, img_size//8, img_size//8)

        x4 = self.bot1(x4)              # x4: (batch_size, 512, img_size//8, img_size//8)
        x4 = self.bot2(x4)              # x4: (batch_size, 512, img_size//8, img_size//8)
        x4 = self.bot3(x4)              # x4: (batch_size, 256, img_size//8, img_size//8)

        x = self.up1(x4, x3, t)         # x: (batch_size, 128, img_size//4, img_size//4)
        x = self.sa4(x)                 # x: (batch_size, 128, img_size//4, img_size//4)
        x = self.up2(x, x2, t)          # x: (batch_size, 64, img_size//2, img_size//2)
        x = self.sa5(x)                 # x: (batch_size, 64, img_size//2, img_size//2)
        x = self.up3(x, x1, t)          # x: (batch_size, 64, img_size, img_size)
        x = self.sa6(x)                 # x: (batch_size, 64, img_size, img_size)
        output = self.outc(x)           # output: (batch_size, 3, img_size, img_size)
        return output

# CFG: Classfier Free Guidance
class UNetConditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            # Keeping the same dimensions as time for adding it to the time embeddings itself!
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        """
            Arguments -
            -----------
            x: Input image                  ; Shape - (batch_size, num_channels, img_size, img_size)
            t: Time steps input             ; Shape - (batch_size, )
            y: Class label value            ; Shape - (batch_size, )

            Returns -
            --------
            output: Predicted noise image   ; Shape - (batch_size, num_channels, img_size, img_size)
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    net = UNetConditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)