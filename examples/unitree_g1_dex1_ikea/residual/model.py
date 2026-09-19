# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The residual head.

Small on purpose. The baseline already carries the skill; what is being learned
here is a correction that is zero almost everywhere, from 12,887 corrective
frames. A large head on that much data would memorise the interventions.

Inputs are the three things the correction can depend on: what the cameras saw
(the frozen backbone tokens, attention-pooled), where the robot is (the state the
baseline also reads), and what the baseline intends to do (its own chunk, which is
what the correction is relative to).

The last layer is zero-initialised, so an untrained head outputs exactly zero and
the deployed policy is exactly the baseline. Training moves away from that only
where the data asks it to.
"""

import torch
from torch import nn


class ResidualHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        state_dim: int,
        horizon: int,
        action_dim: int,
        width: int = 512,
        heads: int = 8,
        clip: float = 0.1,
    ):
        super().__init__()
        self.horizon, self.action_dim, self.clip = horizon, action_dim, clip
        self.tok = nn.Linear(token_dim, width)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.state = nn.Sequential(nn.Linear(state_dim, width), nn.GELU(), nn.Linear(width, width))
        self.base = nn.Sequential(
            nn.Linear(horizon * action_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.out = nn.Linear(width, horizon * action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor, state: torch.Tensor, a_base: torch.Tensor):
        b = feats.shape[0]
        tok = self.tok(feats)
        q = self.query.expand(b, -1, -1)
        vis, _ = self.pool(q, tok, tok, need_weights=False)
        z = torch.cat([vis.squeeze(1), self.state(state), self.base(a_base.flatten(1))], dim=-1)
        delta = self.out(self.trunk(z)).view(b, self.horizon, self.action_dim)
        # A runaway residual must not be able to leave the baseline's neighbourhood.
        # tanh rather than clamp so the bound stays differentiable everywhere.
        return self.clip * torch.tanh(delta / self.clip)
