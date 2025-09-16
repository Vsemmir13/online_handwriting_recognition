import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        pos_x = torch.arange(seq_len, device=x.device).float().unsqueeze(0)
        pos_y = torch.arange(seq_len, device=y.device).float().unsqueeze(0)
        freqs_x = torch.einsum("b n, d -> b n d", pos_x, self.inv_freq)
        freqs_y = torch.einsum("b n, d -> b n d", pos_y, self.inv_freq)
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)
        emb_x = torch.cat((torch.sin(emb_x), torch.cos(emb_x)), dim=-1)
        emb_y = torch.cat((torch.sin(emb_y), torch.cos(emb_y)), dim=-1)
        pos_emb = emb_x + emb_y
        return pos_emb[:, :, : self.dim]


class Conv1DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv2DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BiGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_output))
        return x


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRUCell(embedding_dim + hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoded_features: torch.Tensor,
        prev_chars: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = prev_chars.shape
        if hidden_states is None:
            hidden_states = torch.zeros(
                batch_size, self.hidden_size, device=encoded_features.device
            )
        char_embeddings = self.word_embedding(prev_chars)
        if hidden_states.dim() == 2:
            hidden_for_attention = hidden_states.unsqueeze(1).expand(-1, seq_len, -1)
            if hidden_for_attention.shape[-1] == char_embeddings.shape[-1]:
                query_input = char_embeddings + hidden_for_attention
            else:
                query_input = char_embeddings
        else:
            query_input = char_embeddings
        attn_output, _ = self.attention(
            query_input,
            encoded_features,
            encoded_features,
        )
        gru_input = torch.cat([char_embeddings, attn_output], dim=-1)
        output_hidden_states = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_size,
            device=encoded_features.device,
            dtype=encoded_features.dtype,
        )
        current_hidden = hidden_states
        for t in range(seq_len):
            current_input = gru_input[:, t, :]
            current_hidden = self.gru(current_input, current_hidden)
            output_hidden_states[:, t, :] = current_hidden
        hidden_states = output_hidden_states
        output_logits = self.output_projection(self.dropout(hidden_states))
        output_probs = F.softmax(output_logits, dim=-1)
        return output_probs, hidden_states


class PointToSpatialAlignment(nn.Module):
    def __init__(self, feature_dim: int, num_transformer_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_transformer_layers = num_transformer_layers
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(feature_dim) for _ in range(num_transformer_layers)]
        )
        self.pos_embedding = RotaryPositionalEmbedding2D(feature_dim)

    def forward(
        self, trajectory_features: torch.Tensor, trajectory_coords: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = trajectory_features.shape
        x_coords = trajectory_coords[:, :, 0]
        y_coords = trajectory_coords[:, :, 1]
        pos_emb = self.pos_embedding(x_coords, y_coords)
        features_with_pos = trajectory_features + pos_emb
        aligned_features = features_with_pos
        for transformer in self.transformer_layers:
            aligned_features = transformer(aligned_features)
        return aligned_features


class PTOHWR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 320,
        image_height: int = 32,
        trajectory_channels: int = 3,
        num_conv1d_layers: int = 6,
        num_conv2d_layers: int = 5,
        num_bigru_layers: int = 2,
        num_transformer_layers: int = 3,
        alignment_weight: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.image_height = image_height
        self.trajectory_channels = trajectory_channels
        self.alignment_weight = alignment_weight
        self.conv1d_layers = nn.ModuleList()
        in_channels = trajectory_channels
        for i in range(num_conv1d_layers):
            if i == num_conv1d_layers - 1:
                out_channels = feature_dim
            else:
                out_channels = min(feature_dim, trajectory_channels * (2 ** (i + 1)))
            stride = 2 if i < 3 else 1
            self.conv1d_layers.append(
                Conv1DBlock(in_channels, out_channels, stride=stride)
            )
            in_channels = out_channels
        self.bigru_1d = BiGRU(feature_dim, feature_dim // 2, num_bigru_layers, dropout)
        self.bigru_1d_updated = BiGRU(
            feature_dim, feature_dim // 2, num_bigru_layers, dropout
        )
        self.proj_1d = nn.Linear(feature_dim, feature_dim)
        self.proj_2d = nn.Linear(feature_dim, feature_dim)
        self.conv2d_layers = nn.ModuleList()
        in_channels = 1
        for i in range(num_conv2d_layers):
            if i == num_conv2d_layers - 1:
                out_channels = feature_dim
            else:
                out_channels = min(feature_dim, in_channels * 2)
            stride = 2 if i < 3 else 1
            self.conv2d_layers.append(
                Conv2DBlock(in_channels, out_channels, stride=stride)
            )
            in_channels = out_channels
        self.bigru_2d = BiGRU(feature_dim, feature_dim // 2, num_bigru_layers, dropout)
        self.p2sa = PointToSpatialAlignment(feature_dim, num_transformer_layers)
        self.decoder_1d = AttentionDecoder(
            feature_dim, vocab_size, embedding_dim=feature_dim, dropout=dropout
        )
        self.decoder_2d = AttentionDecoder(
            feature_dim, vocab_size, embedding_dim=feature_dim, dropout=dropout
        )
        self.feature_interpolation = nn.AdaptiveAvgPool2d((1, 1))

    def forward_1d_encoder(self, trajectory: torch.Tensor) -> torch.Tensor:
        x = trajectory.transpose(1, 2)
        for conv in self.conv1d_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        if x.shape[-1] != self.feature_dim:
            x = self.proj_1d(x)
        return x

    def forward_1d_encoder_with_gru(self, trajectory: torch.Tensor) -> torch.Tensor:
        conv_features = self.forward_1d_encoder(trajectory)
        gru_features = self.bigru_1d(conv_features)
        gru_features = self.proj_1d(gru_features)
        return gru_features

    def forward_2d_encoder(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        for conv in self.conv2d_layers:
            x = conv(x)
        self.last_2d_features = x
        x = self.feature_interpolation(x)
        x = x.squeeze(-1).squeeze(-1).unsqueeze(1)
        if x.shape[-1] != self.feature_dim:
            x = self.proj_2d(x)
        return x

    def forward_2d_encoder_with_gru(self, image: torch.Tensor) -> torch.Tensor:
        conv_features = self.forward_2d_encoder(image)
        gru_features = self.bigru_2d(conv_features)
        gru_features = self.proj_2d(gru_features)
        return gru_features

    def forward_p2sa(
        self, trajectory_features: torch.Tensor, trajectory_coords: torch.Tensor
    ) -> torch.Tensor:
        if trajectory_coords.shape[1] != trajectory_features.shape[1]:
            batch_size, feat_seq_len, feat_dim = trajectory_features.shape
            orig_seq_len = trajectory_coords.shape[1]
            indices = torch.linspace(0, orig_seq_len - 1, feat_seq_len).long()
            resampled_coords = trajectory_coords[:, indices, :]
            return self.p2sa(trajectory_features, resampled_coords)
        else:
            return self.p2sa(trajectory_features, trajectory_coords)

    def forward(
        self,
        trajectory: torch.Tensor,
        image: torch.Tensor,
        trajectory_coords: torch.Tensor,
        target_chars: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> dict:
        batch_size = trajectory.shape[0]
        f_1d_conv = self.forward_1d_encoder(trajectory)
        f_1d_gru = self.forward_1d_encoder_with_gru(trajectory)
        f_2d_conv = self.forward_2d_encoder(image)
        f_2d_gru = self.forward_2d_encoder_with_gru(image)
        f_p2s = self.forward_p2sa(f_1d_conv, trajectory_coords)
        f_1d_updated = f_1d_conv + f_p2s
        f_1d_gru_star = self.bigru_1d_updated(f_1d_updated)
        outputs = {
            "f_1d_gru": f_1d_gru,
            "f_2d_gru": f_2d_gru,
            "f_1d_gru_star": f_1d_gru_star,
            "f_p2s": f_p2s,
            "f_2d_conv": f_2d_conv,
        }

        if training and target_chars is not None:
            probs_1d, _ = self.decoder_1d(f_1d_gru_star, target_chars)
            loss_1d = F.cross_entropy(
                probs_1d.view(-1, self.vocab_size),
                target_chars.view(-1),
                ignore_index=-1,
            )
            probs_2d, _ = self.decoder_2d(f_2d_gru, target_chars)
            loss_2d = F.cross_entropy(
                probs_2d.view(-1, self.vocab_size),
                target_chars.view(-1),
                ignore_index=-1,
            )
            feat_seq_len = f_p2s.shape[1]
            orig_seq_len = trajectory_coords.shape[1]
            if feat_seq_len != orig_seq_len:
                indices = torch.linspace(0, orig_seq_len - 1, feat_seq_len).long()
                resampled_coords = trajectory_coords[:, indices, :]
            else:
                resampled_coords = trajectory_coords
            f_2d_sample = self._interpolate_features(
                self.last_2d_features, resampled_coords
            )
            f_2d_sample_sg = f_2d_sample.detach()
            loss_align = F.mse_loss(f_p2s, f_2d_sample_sg)
            total_loss = loss_1d + loss_2d + self.alignment_weight * loss_align
            outputs.update(
                {
                    "loss_1d": loss_1d,
                    "loss_2d": loss_2d,
                    "loss_align": loss_align,
                    "total_loss": total_loss,
                    "probs_1d": probs_1d,
                    "probs_2d": probs_2d,
                }
            )

        return outputs

    def _interpolate_features(
        self, features: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        coords_normalized = coords.clone()
        coords_normalized[:, :, 0] = coords_normalized[:, :, 0] / width
        coords_normalized[:, :, 1] = coords_normalized[:, :, 1] / height
        features_flat = features.view(batch_size, channels, -1)
        return features.mean(dim=(2, 3)).unsqueeze(1).expand(-1, coords.shape[1], -1)

    def inference(
        self, trajectory: torch.Tensor, trajectory_coords: torch.Tensor
    ) -> torch.Tensor:
        f_1d_conv = self.forward_1d_encoder(trajectory)
        f_p2s = self.forward_p2sa(f_1d_conv, trajectory_coords)
        f_1d_updated = f_1d_conv + f_p2s
        f_1d_gru_star = self.bigru_1d_updated(f_1d_updated)
        f_1d_gru_star = self.proj_1d(f_1d_gru_star)
        return f_1d_gru_star
    
    def decode_iterative(
        self, 
        encoded_features: torch.Tensor, 
        max_length: int = 128,
        blank_idx: int = None
    ) -> torch.Tensor:
        if blank_idx is None:
            blank_idx = self.vocab_size - 1
        batch_size = encoded_features.shape[0]
        device = encoded_features.device
        seq_len = min(encoded_features.shape[1], max_length)
        encoded_flat = encoded_features[:, :seq_len, :].reshape(-1, encoded_features.shape[-1])
        char_logits = self.decoder_1d.output_projection(encoded_flat)
        char_logits = char_logits.reshape(batch_size, seq_len, self.vocab_size)
        char_probs = F.softmax(char_logits, dim=-1)
        predicted_chars = torch.argmax(char_probs, dim=-1)
        return predicted_chars
