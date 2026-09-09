# -*- coding: utf-8 -*-
"""
PIBTUL strict implementation, faithful to the paper
"Identifying Human Mobility via Prototype-guided Information Bottleneck".

Key equations implemented:
  Eq.(10)-(11): trajectory truncating / reversal augmentation (in utils.py)
  Eq.(12)-(14): self-attention + LayerNorm residual -> LSTM -> final hidden state
  Eq.(15)-(16): per-view Bayesian sampling -> z, z', z'' (three latent variables)
  Eq.(9):       L_T,T' = 0.5*DKL(p||p') + 0.5*DKL(p'||p) - ((eta+zeta)/2) * I(z; z')
  Eq.(17):      L_MobCL = L_T,T' + L_T,T'' + L_T',T''
  Eq.(18)-(21): prototypes ~ N(0, 0.01^2 I), momentum update nu=0.9,
                L_intra = MSE(z_hat, r_u), L_inter = mean(max D - min D)
  Eq.(22):      z_hat = [z_hat^(1) || z_hat^(2) || z_hat^(3)] (pairwise fusion concat)
  Eq.(23)-(24): softmax classifier + cross entropy
  Eq.(25):      L = L_u + gamma * L_MobCL + lambda * (L_intra + L_inter)

Paper leaves the following unspecified; our documented choices:
  - "integrating the two latent representations" in Eq.(22) context -> average.
  - eta, zeta (MI term weight) -> single hyperparameter mi_weight = (eta+zeta)/2, default 1.0.
  - I(z; z') estimation -> MINE (Donsker-Varadhan lower bound) with a critic
    network trained in alternation (critic updated on detached latents).
  - Prototype space dimension is d (Eq. 18); pairwise fused representations
    z_hat^(k) are d-dim, so prototype losses / updates act on the average of
    the three pairwise fusions (each d-dim), not on the 3d concatenation.
  - At inference: the raw trajectory is used for all three views and the
    posterior mean mu is used (deterministic), no augmentation, no sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """Shared trajectory encoder: Eq.(12)-(14).

    Self-attention + LayerNorm residual, then LSTM; the trajectory encoding
    h_T is the final hidden state of the (single) LSTM layer (Eq. 14 with L=1).
    """

    def __init__(self, embed_size, hidden_size, num_layers, embeddings, device,
                 use_attn=True, pretrained_emb=True):
        super().__init__()
        self.device = device
        self.use_attn = use_attn
        # w/o PEmb: use a randomly-initialized learnable embedding instead of the
        # frozen pre-trained (node2vec) POI embedding.
        if pretrained_emb:
            self.base_embedding = nn.Embedding.from_pretrained(
                embeddings, padding_idx=0, freeze=True).to(device)
        else:
            self.base_embedding = nn.Embedding(
                embeddings.shape[0], embed_size, padding_idx=0).to(device)
        self.embed_adapter = nn.Linear(embed_size, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        ).to(device)
        self.num_layers = num_layers

    def forward(self, seq, lengths):
        emb = self.embed_adapter(self.base_embedding(seq))
        # Eq.(12): h_i^attn = M(A(v(c_i)) + v(c_i))   (w/o Attn: skip attention)
        if self.use_attn:
            attn_input = emb.permute(1, 0, 2)
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            attn_output = attn_output.permute(1, 0, 2)
            h_attn = self.layer_norm(attn_output + emb)
        else:
            h_attn = emb
        # Eq.(13)-(14): LSTM, average of final hidden states over layers
        packed = nn.utils.rnn.pack_padded_sequence(
            h_attn, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)          # hidden: [L, B, d]
        h_T = hidden.mean(dim=0)                    # Eq.(14): (1/L) * sum_l h_n^(l)
        return h_T


class BayesianHead(nn.Module):
    """Eq.(15)-(16): map h_T to a Gaussian and sample z = mu + sigma * eps."""

    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, h_T, sample=True):
        mu = self.fc_mu(h_T)
        logvar = torch.clamp(self.fc_logvar(h_T), min=-10.0, max=10.0)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar


class MINE(nn.Module):
    """Mutual information estimator for the term I(z; z') in Eq.(9).

    JS / Deep-InfoMax style lower bound with a critic network T(x, y):
        I(X;Y) >= E_joint[-softplus(-T)] - E_marginal[softplus(T)]
    The bound is bounded above by log 4, which keeps the estimator stable
    (an unbounded Donsker-Varadhan estimator was observed to diverge and
    destroy the encoder when the -I(z;z') term dominates the objective).
    The critic is trained to maximise the bound (minimise its negation) on
    detached latents; the encoder objective subtracts the bound, i.e. it
    maximises I(z; z') as required by Eq.(9).
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(2 * latent_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1),
        )

    def mi_estimate(self, z1, z2):
        # JS / Deep-InfoMax style bound: I(X;Y) >= E_joint[-softplus(-T)]
        # - E_marginal[softplus(T)], bounded above by log 4 -> stable training.
        joint = self.critic(torch.cat([z1, z2], dim=1))
        idx = torch.randperm(z2.size(0), device=z2.device)
        marginal = self.critic(torch.cat([z1, z2[idx]], dim=1))
        mi = (-F.softplus(-joint)).mean() - F.softplus(marginal).mean()
        return mi

    def critic_loss(self, z1, z2):
        # maximise the DV bound <=> minimise its negation (on detached inputs)
        return -self.mi_estimate(z1.detach(), z2.detach())


def gaussian_kl(mu1, logvar1, mu2, logvar2):
    """DKL(N(mu1, sigma1^2) || N(mu2, sigma2^2)), mean over batch."""
    kl = logvar2 - logvar1 + (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp() - 1.0
    return 0.5 * kl.sum(dim=1).mean()


class PIBTUL(nn.Module):
    """Full PIBTUL model: MobCL (multi-view IB) + PGO (prototype-guided opt.)."""

    def __init__(self, embed_size, hidden_size, latent_dim, num_layers,
                 embeddings, output_user_size, device,
                 nu=0.9, mi_weight=1.0, inter_mode='paper', init_mode='gaussian',
                 views='OTR', use_attn=True, pretrained_emb=True):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        # Table IV ablation: active views (subset of O/T/R/S; S=Substitution).
        self.views = views.upper()
        self.act = [c for c in 'OTRS' if c in self.views]
        if not self.act:
            self.act = ['O']
        self.n_act = len(self.act)
        self.mi_weight = mi_weight  # (eta + zeta) / 2 in Eq.(9); paper unspecified
        # 'paper'  = literal Eq.(21): mean(max D - min D)
        # 'intent' = paper's stated intent ("sufficient margins"): -mean(min D)
        self.inter_mode = inter_mode

        # C2-3e variant: three fully independent encoders (one per view),
        # mirroring the exp1/my_TUL engineering version. fuse/prototype-768/
        # warmup stay exactly as in C2. The substitution view (S, revision
        # R1.1/R2.5) gets its own encoder/head too; names of the O/T/R
        # modules are unchanged so earlier checkpoints still load.
        self.encoder_orig = TrajectoryEncoder(embed_size, hidden_size, num_layers,
                                              embeddings, device, use_attn=use_attn,
                                              pretrained_emb=pretrained_emb)
        self.encoder_crop = TrajectoryEncoder(embed_size, hidden_size, num_layers,
                                              embeddings, device, use_attn=use_attn,
                                              pretrained_emb=pretrained_emb)
        self.encoder_rev = TrajectoryEncoder(embed_size, hidden_size, num_layers,
                                             embeddings, device, use_attn=use_attn,
                                             pretrained_emb=pretrained_emb)
        # per-view Bayesian heads: p_theta(z|T), p_psi(z'|T'), p_phi(z''|T'')
        self.head_orig = BayesianHead(hidden_size, latent_dim)
        self.head_crop = BayesianHead(hidden_size, latent_dim)
        self.head_rev = BayesianHead(hidden_size, latent_dim)
        if 'S' in self.act:
            self.encoder_subst = TrajectoryEncoder(embed_size, hidden_size,
                                                   num_layers, embeddings, device,
                                                   use_attn=use_attn,
                                                   pretrained_emb=pretrained_emb)
            self.head_subst = BayesianHead(hidden_size, latent_dim)

        self.mine_oc = MINE(latent_dim)
        self.mine_or = MINE(latent_dim)
        self.mine_cr = MINE(latent_dim)
        if 'S' in self.act:
            self.mine_os = MINE(latent_dim)
            self.mine_ts = MINE(latent_dim)
            self.mine_rs = MINE(latent_dim)

        # active view pairs for MobCL (Eq.17 sums over all view pairs)
        _mine_for = {('O', 'T'): 'mine_oc', ('O', 'R'): 'mine_or',
                     ('T', 'R'): 'mine_cr', ('O', 'S'): 'mine_os',
                     ('T', 'S'): 'mine_ts', ('R', 'S'): 'mine_rs'}
        self.pairs = [(a, b, getattr(self, _mine_for[(a, b)]))
                      for a, b in [('O', 'T'), ('O', 'R'), ('T', 'R'),
                                   ('O', 'S'), ('T', 'S'), ('R', 'S')]
                      if a in self.act and b in self.act]

        # Eq.(23): classifier on the concatenated view embedding (n_act views)
        self.out = nn.Linear(self.n_act * latent_dim, output_user_size)

        # Variant-C: prototype space lives in the fused z_hat space
        # (n_act*latent): PGO acts directly on z_hat, no averaged z_proto.
        self.proto_dim = self.n_act * latent_dim
        # Table V ablation: prototype initialization.
        #   gaussian  -> N(0, 0.01^2 I)      (paper default)
        #   uniform   -> U[-0.01, 0.01]
        #   classmean -> zeros here; main.py fills each user's proto with the
        #                mean of that user's z_hat over the training set.
        self.init_mode = init_mode
        if init_mode == 'uniform':
            self.prototypes = nn.Parameter(
                (torch.rand(output_user_size, self.proto_dim) * 2 - 1) * 0.01)
        elif init_mode == 'classmean':
            self.prototypes = nn.Parameter(
                torch.zeros(output_user_size, self.proto_dim))
        else:  # 'gaussian' default
            self.prototypes = nn.Parameter(
                torch.randn(output_user_size, self.proto_dim) * 0.01)
        self.nu = nu  # momentum trade-off, Eq.(19)

    # ---------------- MobCL ----------------
    def encode_views(self, multi_view_inputs, sample=True):
        """Returns {letter: (z, mu, logvar)} for the active views present in
        multi_view_inputs ('orig'->O, 'crop'->T, 'reverse'->R, 'subst'->S)."""
        out = {}
        if 'orig' in multi_view_inputs:
            h = self.encoder_orig(*multi_view_inputs['orig'])
            out['O'] = self.head_orig(h, sample)
        if 'crop' in multi_view_inputs:
            h = self.encoder_crop(*multi_view_inputs['crop'])
            out['T'] = self.head_crop(h, sample)
        if 'reverse' in multi_view_inputs:
            h = self.encoder_rev(*multi_view_inputs['reverse'])
            out['R'] = self.head_rev(h, sample)
        if 'subst' in multi_view_inputs:
            h = self.encoder_subst(*multi_view_inputs['subst'])
            out['S'] = self.head_subst(h, sample)
        return out

    @staticmethod
    def _pair_fuse(z1, z2):
        # "integrating the two latent representations" (paper, after Eq.9) -> average
        return 0.5 * (z1 + z2)

    def mobcl_pair_loss(self, va, vb, mine):
        """Eq.(9): symmetric posterior KL minus weighted mutual information.
        Returns the raw (KL, MI) so main.py can weight them by beta_kl/beta_mi."""
        _, mu_a, lv_a = va
        _, mu_b, lv_b = vb
        sym_kl = 0.5 * gaussian_kl(mu_a, lv_a, mu_b, lv_b) + \
                 0.5 * gaussian_kl(mu_b, lv_b, mu_a, lv_a)
        mi = mine.mi_estimate(va[0], vb[0])
        return sym_kl, mi

    def mobcl_loss(self, views):
        """Eq.(17): L_MobCL summed over all active view pairs. views is the
        dict from encode_views. Returns raw (KL_sum, MI_sum); main.py weights
        by beta_kl / beta_mi. Single active view -> no pairs -> zeros."""
        kl_sum = 0.0
        mi_sum = 0.0
        n = 0
        for (a, b, mine) in self.pairs:
            if a in views and b in views:
                kl, mi = self.mobcl_pair_loss(views[a], views[b], mine)
                kl_sum = kl_sum + kl
                mi_sum = mi_sum + mi
                n += 1
        if n == 0:
            z = torch.zeros((), device=self.device, requires_grad=True)
            return z, z
        return kl_sum, mi_sum

    def mine_critic_loss(self, views):
        losses = []
        for (a, b, mine) in self.pairs:
            if a in views and b in views:
                losses.append(mine.critic_loss(views[a][0], views[b][0]))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return sum(losses) / len(losses)

    def critic_modules(self):
        """All MINE critics (for the separate critic optimizer in main.py)."""
        return [m for _, _, m in self.pairs]

    # ---------------- view fusion, Eq.(22) ----------------
    def fuse(self, views):
        # ABLATION no-pairfuse (2026-09-04): feed the raw view latents as the
        # channels instead of pairwise averages. z_proto is mathematically
        # unchanged: (0.5(z+z')+0.5(z+z'')+0.5(z'+z''))/3 == (z+z'+z'')/3.
        # Table IV: concatenate only the active views' latents.
        z_hat = torch.cat([views[c][0] for c in self.act], dim=1)  # n_act*dim
        return z_hat, z_hat    # Variant-C: PGO on the concatenated z_hat

    # ---------------- PGO, Eq.(18)-(21) ----------------
    def update_prototypes(self, z_proto, labels):
        """Eq.(19): r_uj <- nu * r_uj + (1 - nu) * mean of the user's batch reprs.

        Applied via .data so the momentum rule coexists with the gradient
        updates the Parameter receives from L_intra / L_inter (called after
        optimizer.step(), so no autograd version conflict)."""
        with torch.no_grad():
            for label in torch.unique(labels):
                mask = labels == label
                batch_mean = z_proto[mask].mean(dim=0)
                self.prototypes.data[label] = \
                    self.nu * self.prototypes.data[label] + \
                    (1 - self.nu) * batch_mean

    def prototype_loss(self, z_proto, labels):
        """Eq.(20)-(21): intra-class compactness + inter-class separation."""
        # Eq.(20): L_intra = (1/B) * sum_i ||z_hat_i - r_ui||^2
        # v4: keep the Eq.(20) sum-form but scale by 1/latent_dim. The literal
        # x256 version was proven untrainable twice (v2 with two-sided grad,
        # v3 with prototype stop-grad both collapsed to single-class
        # predictions): at lambda=1 it starves the CE gradient before any
        # discriminative structure forms. Scaling by 1/256 keeps the formula's
        # shape while matching the effective weight the paper's lambda=1 can
        # actually sustain. Prototype side stays stop-grad (paper's text:
        # "guiding z_hat to approach their prototype centers"); L_inter keeps
        # its gradient through the learnable prototypes (v2 fix, review pt.1).
        l_intra = (z_proto - self.prototypes[labels].detach()).pow(2).sum(dim=1).mean() \
            / self.proto_dim
        # Eq.(21): L_inter = (1/M) * sum_m (max_{j!=m} D_mj - min_{j!=m} D_mj)
        dmat = torch.cdist(self.prototypes, self.prototypes)
        eye = torch.eye(dmat.size(0), device=dmat.device)
        min_d = (dmat + eye * 1e9).min(dim=1)[0]   # exclude self-distance
        if self.inter_mode == 'intent':
            # paper's stated intent variant: maximise the nearest-prototype margin
            l_inter = -min_d.mean()
        else:
            max_d = (dmat - eye * 1e9).max(dim=1)[0]   # exclude self-distance
            l_inter = (max_d - min_d).mean()
        return l_intra, l_inter

    # ---------------- forward ----------------
    def forward(self, multi_view_inputs, sample=None):
        if sample is None:
            sample = self.training
        views = self.encode_views(multi_view_inputs, sample=sample)
        z_hat, z_proto = self.fuse(views)
        logits = self.out(z_hat)                       # Eq.(23)
        return logits, views, z_proto
