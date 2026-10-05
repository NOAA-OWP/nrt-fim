import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def forward(self, x):
        """Swish forward"""
        return x * torch.sigmoid(x)


class CustomDecoderBlock(nn.Module):
    """Custom decoder block with explicit channel counts"""

    def __init__(self, in_channels, up_channels, skip_channels, out_channels, device="cpu"):
        super().__init__()
        self.device = device

        # Here we explicitly define the upsampling output channels
        self.upsample = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2).to(device)

        # Calculate concatenated channels
        concat_channels = up_channels + skip_channels

        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(out_channels).to(device)
        self.swish1 = Swish().to(device)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(out_channels).to(device)
        self.swish2 = Swish().to(device)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, skip):
        """Decoder Block forward function"""
        x_up = self.upsample(x)
        # print(f"After upsample: {x_up.shape}")

        if x_up.size()[2:] != skip.size()[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=True)
            # print(f"After interpolate: {x_up.shape}")

        x = torch.cat([x_up, skip], dim=1)
        # print(f"After concat: {x.shape}")

        x = self.swish1(self.bn1(self.conv1(x)))
        x = self.swish2(self.bn2(self.conv2(x)))

        return x


class FModel(nn.Module):
    """cnn structure defined here"""

    def __init__(self, num_classes=1, in_channels=5, device="cpu"):
        super().__init__()

        self.device = device
        self.to(device)

        # Initialize encoder as before
        self.encoder = efficientnet_b1(weights=None).to(device)
        original_conv = self.encoder.features[0][0]
        self.encoder.features[0][0] = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        ).to(device)
        self.initial_conv = self.encoder.features[0].to(device)

        # Create encoder blocks as before
        self.encoder_blocks = nn.ModuleList().to(device)
        self.encoder_blocks.append(self.encoder.features[1].to(device))
        self.encoder_blocks.append(self.encoder.features[2].to(device))
        self.encoder_blocks.append(self.encoder.features[3].to(device))
        self.encoder_blocks.append(self.encoder.features[4].to(device))
        self.encoder_blocks.append(
            nn.Sequential(
                self.encoder.features[5],
                self.encoder.features[6],
                self.encoder.features[7],
            ).to(device)
        )

        # Updated skip channels to match your actual encoder outputs
        self.skip_channels = [16, 24, 40, 80, 320]  # Correct channels from log output
        self.bottleneck = self.encoder.features[8].to(device)

        # Create custom decoder blocks with precise channel counts
        self.decoder_blocks = nn.ModuleList().to(device)

        # Decoder 1: Takes bottleneck (1280) → outputs 320
        self.decoder_blocks.append(
            CustomDecoderBlock(
                in_channels=1280,  # Bottleneck output
                up_channels=640,  # After upsampling
                skip_channels=320,  # From skip connection
                out_channels=320,  # Output channels
                device=device,
            )
        )

        # Decoder 2: Takes 320 → outputs 112 (matches with skip 80)
        self.decoder_blocks.append(
            CustomDecoderBlock(
                in_channels=320,  # From previous decoder
                up_channels=160,  # After upsampling
                skip_channels=80,  # From skip connection
                out_channels=112,  # Output channels
                device=device,
            )
        )

        # Decoder 3: Takes 112 → outputs 40
        self.decoder_blocks.append(
            CustomDecoderBlock(
                in_channels=112,
                up_channels=56,
                skip_channels=40,
                out_channels=40,
                device=device,
            )
        )

        # Decoder 4: Takes 40 → outputs 24
        self.decoder_blocks.append(
            CustomDecoderBlock(
                in_channels=40,
                up_channels=20,
                skip_channels=24,
                out_channels=24,
                device=device,
            )
        )

        # Decoder 5: Takes 24 → outputs 32
        self.decoder_blocks.append(
            CustomDecoderBlock(
                in_channels=24,
                up_channels=12,
                skip_channels=16,
                out_channels=32,
                device=device,
            )
        )

        # Final output layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

        # Apply Xavier Glorot initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights using Xavier Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Cnn forward

        :param x: inputs
        :return: outputs prediction
        """
        skip_features = []
        _, _, H, W = x.shape
        x = self.initial_conv(x)
        #  print(f"After initial conv: {x.shape}")

        for _i, block in enumerate(self.encoder_blocks):
            x = block(x)
            #  print(f"Encoder block {i} output: {x.shape}")
            skip_features.append(x)

        x = self.bottleneck(x)
        #  print(f"Bottleneck output: {x.shape}")

        for i, block in enumerate(self.decoder_blocks):
            #  print(f"Before decoder {i}: x={x.shape}, skip={skip_features[4-i].shape}")
            x = block(x, skip_features[4 - i])
            #  print(f"After decoder {i}: {x.shape}")

        ## upsample to the original input size
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x
