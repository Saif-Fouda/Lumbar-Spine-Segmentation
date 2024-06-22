import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, middle_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Correction: the input to conv1 should be 'middle_channels' + 'in_channels' (from skip connection)
        self.conv1 = nn.Conv3d(middle_channels + middle_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_x):
        x = self.upconv(x)
        
        # Crop or pad x to ensure dimensions match for skip_x
        # (You might need to adjust this based on your actual dimensions if they do not match)
        if x.size() != skip_x.size():
            _, _, D, H, W = skip_x.size()
            x = F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=True)
        
        # Concatenate along the channel dimension
        print(f"Upsampled size: {x.size()}, Skip connection size: {skip_x.size()}")
        x = torch.cat([x, skip_x], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x



class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder1 = Encoder(in_channels, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.bottleneck = BottleNeck(256, 512)
        self.decoder3 = Decoder(512, 256, 256)
        self.decoder2 = Decoder(256, 128, 128)
        self.decoder1 = Decoder(128, 64, 64)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False))
        self.output = nn.Conv3d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x = self.bottleneck(x3)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        x = self.final_upsample(x)
        x = self.output(x)
        return x