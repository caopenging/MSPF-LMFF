import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Sequential):
    def __init__(self, C_in, C_out, kernel_size=1):
        super(ConvBN, self).__init__(
            nn.Conv1d(C_in, C_out, kernel_size),
            nn.BatchNorm1d(C_out),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, C_in, C_out, kernel_size=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv1d(C_in, C_out, kernel_size),
            nn.BatchNorm1d(C_out),
            nn.ReLU(inplace=True)
        )


class ResnetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8):
        super().__init__()
        self.heads = heads
        assert in_dim % heads == 0, "in_dim must be divisible by heads for equal distribution"

        self.query_conv = nn.Conv1d(in_dim, in_dim, 1)  # Ensuring the output is in_dim * heads
        self.key_conv = nn.Conv1d(in_dim, in_dim, 1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Conv1d(in_dim, in_dim, 1)

    def forward(self, x):
        B, C, N = x.size()
        # Ensure C is divisible by heads
        q = self.query_conv(x).view(B, self.heads, C // self.heads, N)
        k = self.key_conv(x).view(B, self.heads, C // self.heads, N)
        v = self.value_conv(x).view(B, self.heads, C // self.heads, N)

        energy = torch.einsum('bhcn,bhcm->bhnm', q, k)  # [B, heads, N, N]
        attention = self.softmax(energy)
        out = torch.einsum('bhnm,bhcm->bhcn', attention, v).contiguous().view(B, C, N)

        out = self.fc(out) + x  # Apply residual connection
        return out


class PointCloudEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, 1)
        self.attention1 = MultiHeadSelfAttention(64)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = ResnetBasicBlock(64, 64)
        self.attention2 = MultiHeadSelfAttention(64)
        self.conv3 = ConvBNReLU(64, 128, 1)
        self.attention3 = MultiHeadSelfAttention(128)
        self.conv4 = ResnetBasicBlock(128, 128)
        self.attention4 = MultiHeadSelfAttention(128)
        self.conv5 = ConvBNReLU(256, 256, 1)  # Adjusted input channel count
        self.attention5 = MultiHeadSelfAttention(256)
        self.conv6 = ResnetBasicBlock(256, 256)
        self.attention6 = MultiHeadSelfAttention(256)
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, xyz):
        np = xyz.size()[2]
        x = self.conv1(xyz)
        x = self.attention1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        x = self.conv3(x)
        x = self.attention3(x)
        x = self.conv4(x)
        x = self.attention4(x)
        global_feat = F.adaptive_max_pool1d(x, 1)
        x = torch.cat((x, global_feat.repeat(1, 1, np)), dim=1)
        x = self.conv5(x)
        x = self.attention5(x)
        x = self.conv6(x)
        x = self.attention6(x)
        x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
        embedding = self.fc(x)
        return embedding

class PointCloudDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3 * n_pts)

    def forward(self, embedding):
        bs = embedding.size(0)
        out = F.relu(self.fc1(embedding))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out_pc = out.view(bs, -1, 3)
        return out_pc


class GSENet2(nn.Module):
    def __init__(self, emb_dim=512, n_pts=1024):
        super(GSENet2, self).__init__()
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = PointCloudDecoder(emb_dim, n_pts)

    def forward(self, in_pc, emb=None):
        """
        If an embedding is not provided, it computes it using the input point cloud.
        Args:
            in_pc: (B, N, 3) - Batch of N-point clouds, each with 3 coordinates.
            emb: (B, emb_dim) - Optional precomputed embeddings.

        Returns:
            emb: (B, emb_dim) - Embeddings for each point cloud.
            out_pc: (B, n_pts, 3) - Decoded point clouds.
        """
        if emb is None:
            xyz = in_pc.permute(0, 2, 1)  # Rearrange to (B, 3, N)
            emb = self.encoder(xyz)
        out_pc = self.decoder(emb)
        return emb, out_pc


# Example usage
if __name__ == "__main__":
    model = GSENet2()
    in_pc = torch.rand(10, 1024, 3)  # 10 point clouds, each with 1024 points
    emb, out_pc = model(in_pc)
    print("Embeddings shape:", emb.shape)
    print("Output Point Cloud shape:", out_pc.shape)

