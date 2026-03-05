import torch
import torch.nn as nn
import torch.nn.functional as F


class MIEstimator(nn.Module):
    """Mutual Information Estimator"""

    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        pos_logits = self.net(torch.cat([z1, z2], dim=1))
        pos_labels = torch.ones_like(pos_logits)
        neg_indices = torch.randperm(batch_size, device=z1.device)
        neg_logits = self.net(torch.cat([z1, z2[neg_indices]], dim=1))
        neg_labels = torch.zeros_like(neg_logits)
        loss = (self.loss_fn(pos_logits, pos_labels) + self.loss_fn(neg_logits, neg_labels)) / 2
        return loss, None


class Processor(nn.Module):
    """Cross-view Prediction Processor"""

    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.mlp(x)


class ViewSpecificEncoder(nn.Module):
    """View-specific Encoder Component"""

    def __init__(self, embed_size, hidden_size, dropout_prob, num_layers, view_name):
        super().__init__()
        self.view_name = view_name

        # View-specific adapter layer
        self.embed_adapter = nn.Linear(embed_size, hidden_size)

        # View-specific attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )

        # View-specific layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # View-specific LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # View-specific dropout and fully connected layer
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size)

        # View-specific feature transformation layer (enhance view differences)
        self.view_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Use different initialization strategies for different views
        if self.view_name == 'orig':
            # Original view: standard initialization
            init_gain = 1.0
        elif self.view_name == 'crop':
            # Crop view: slightly enhanced initialization
            init_gain = 1.2
        else:  # reverse
            # Reverse view: stronger initialization difference
            init_gain = 0.8

        # Apply different initialization
        nn.init.xavier_normal_(self.embed_adapter.weight, gain=init_gain)
        nn.init.constant_(self.embed_adapter.bias, 0.1)

        nn.init.xavier_normal_(self.fc1.weight, gain=init_gain)
        nn.init.constant_(self.fc1.bias, 0.1)

        # LSTM weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param, gain=init_gain)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=init_gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, emb, lengths):
        """Forward propagation for single view processing"""
        # View-specific embedding adaptation
        emb = self.embed_adapter(emb)

        # View-specific attention processing
        attn_input = emb.permute(1, 0, 2)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)
        attn_res = self.layer_norm(attn_output + emb)

        # View-specific LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(attn_res, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = self.dropout(hidden)
        hidden = self.fc1(hidden[-1, :, :])

        # View-specific feature transformation
        hidden = self.view_transform(hidden)

        return hidden


class Encoder(nn.Module):
    """Optimized Trajectory Encoder - Supports View-specific Encoding"""

    def __init__(self, embed_size, hidden_size, dropout_prob, num_layers, embeddings, output_user_size, device):
        super().__init__()
        self.device = device

        # Shared base embedding layer
        self.base_embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=True).to(device)

        # Create independent encoders for each view
        self.orig_encoder = ViewSpecificEncoder(embed_size, hidden_size, dropout_prob, num_layers, 'orig')
        self.crop_encoder = ViewSpecificEncoder(embed_size, hidden_size, dropout_prob, num_layers, 'crop')
        self.reverse_encoder = ViewSpecificEncoder(embed_size, hidden_size, dropout_prob, num_layers, 'reverse')

        # Information bottleneck layer, reduce dimension then restore
        self.ib_mu = nn.Linear(hidden_size * 3, hidden_size)
        self.ib_logvar = nn.Linear(hidden_size * 3, hidden_size)

        # Cross-view prediction processors
        self.processor_orig2crop = Processor(in_dim=hidden_size, out_dim=hidden_size)
        self.processor_orig2rev = Processor(in_dim=hidden_size, out_dim=hidden_size)
        self.processor_crop2orig = Processor(in_dim=hidden_size, out_dim=hidden_size)
        self.processor_rev2orig = Processor(in_dim=hidden_size, out_dim=hidden_size)

        # Mutual information estimator
        self.mi_estimator = MIEstimator(z_dim=hidden_size)

        # View contrastive learning loss weight
        self.contrastive_weight = 0.1

        self._init_weights()

    def _init_weights(self):
        """Initialize information bottleneck layer weights"""
        nn.init.xavier_normal_(self.ib_mu.weight)
        nn.init.constant_(self.ib_mu.bias, 0.1)
        nn.init.xavier_normal_(self.ib_logvar.weight)
        nn.init.constant_(self.ib_logvar.bias, 0.1)

    def _process_single_view(self, seq, lengths, view_type='orig'):
        """Process single view using view-specific encoder"""
        # Shared base embedding
        emb = self.base_embedding(seq)

        # Select corresponding encoder based on view type
        if view_type == 'orig':
            return self.orig_encoder(emb, lengths)
        elif view_type == 'crop':
            return self.crop_encoder(emb, lengths)
        elif view_type == 'reverse':
            return self.reverse_encoder(emb, lengths)
        else:
            raise ValueError(f"Unknown view type: {view_type}")

    def forward(self, multi_view_inputs):
        # Process each view using view-specific encoders
        orig_emb = self._process_single_view(*multi_view_inputs['orig'], view_type='orig')
        crop_emb = self._process_single_view(*multi_view_inputs['crop'], view_type='crop')
        reverse_emb = self._process_single_view(*multi_view_inputs['reverse'], view_type='reverse')

        combined = torch.cat([orig_emb, crop_emb, reverse_emb], dim=1)

        mu = self.ib_mu(combined)
        logvar = self.ib_logvar(combined)

        std = torch.exp(0.5 * logvar)
        fused_emb = mu + torch.randn_like(std) * std

        # L2 regularization to keep mu close to 0, avoid drift
        l2_reg = torch.mean(mu.pow(2))

        mi_loss_oc, _ = self.mi_estimator(orig_emb, crop_emb)
        mi_loss_or, _ = self.mi_estimator(orig_emb, reverse_emb)
        mi_loss_cr, _ = self.mi_estimator(crop_emb, reverse_emb)

        pred_oc = self.processor_orig2crop(orig_emb)
        pred_or = self.processor_orig2rev(orig_emb)
        pred_co = self.processor_crop2orig(crop_emb)
        pred_ro = self.processor_rev2orig(reverse_emb)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        mse_loss = (F.mse_loss(pred_oc, crop_emb) +
                    F.mse_loss(pred_or, reverse_emb) +
                    F.mse_loss(pred_co, orig_emb) +
                    F.mse_loss(pred_ro, orig_emb))

        # Total loss includes contrastive learning loss
        total_loss = (kl_loss + 0.5 * (mi_loss_oc + mi_loss_or + mi_loss_cr) +
                      mse_loss + 0.001 * l2_reg)

        return fused_emb, total_loss


class Model(nn.Module):
    def __init__(self, encoder, hidden_size, output_user_size):
        super().__init__()
        self.encoder = encoder
        self.out = nn.Linear(hidden_size, output_user_size)

        # Random initialization of prototypes with zero mean and small variance
        # Prototype initialization with zero mean and small standard deviation
        self.register_buffer("prototypes", torch.randn(output_user_size, hidden_size) * 0.01)

        self.prototype_momentum = 0.9  # Momentum value
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_prototypes(self, z, labels, scale_factor=1):
        """
        Incrementally update clustering prototypes. Add scale_factor to reduce update step size
        """
        for i in range(z.size(0)):
            label = labels[i]
            # Calculate prototype update using momentum method, reduce step size
            self.prototypes[label] = (
                    self.prototype_momentum * self.prototypes[label]
                    + (1 - self.prototype_momentum) * z[i] * scale_factor  # Add scaling factor
            )

    def inter_cluster_distance_loss(self):
        """
        Calculate inter-cluster distance maximization loss to increase inter-cluster distances
        """
        dist_matrix = torch.cdist(self.prototypes, self.prototypes)  # [num_classes, num_classes]
        dist_matrix = dist_matrix + torch.eye(dist_matrix.size(0)).to(self.device) * 1e9  # Avoid self-distance
        min_dist = dist_matrix.min(dim=1)[0]  # Find minimum inter-cluster distance for each prototype
        max_dist = dist_matrix.max(dim=1)[0]  # Find maximum inter-cluster distance for each prototype

        # Limit maximum inter-cluster distance loss to prevent numerical explosion
        max_dist = torch.clamp(max_dist, min=0.0, max=10.0)  # Limit maximum inter-cluster distance
        min_dist = torch.clamp(min_dist, min=0.0, max=10.0)  # Limit minimum inter-cluster distance

        return max_dist.mean() - min_dist.mean()  # Difference between max and min inter-cluster distances as regularization

    def cluster_loss(self, z, labels):
        """
        Calculate clustering loss, including intra-cluster distance minimization and inter-cluster distance maximization
        """
        # Intra-cluster distance minimization
        intra_loss = 0
        for i in range(z.size(0)):
            label = labels[i]
            intra_loss += F.mse_loss(z[i], self.prototypes[label])

        # Inter-cluster distance maximization
        inter_loss = self.inter_cluster_distance_loss()

        return intra_loss / z.size(0) + 0.1 * inter_loss  # Intra-cluster loss + inter-cluster loss

    def forward(self, batch_data):
        """
        Forward propagation: extract features from input data and perform classification.
        """
        # Prepare multi-view input
        multi_view_inputs = {
            'orig': (batch_data['orig_seq'], batch_data['orig_length']),
            'crop': (batch_data['crop_seq'], batch_data['crop_length']),
            'reverse': (batch_data['reverse_seq'], batch_data['reverse_length'])
        }

        # Encoding process
        z, kl_loss = self.encoder(multi_view_inputs)

        # Pass to classifier
        logits = self.out(z)

        return logits, kl_loss, z