import logging
from typing import Any, KeysView, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Losses"]

logger = logging.getLogger(__name__)


class DPOLoss(nn.Module):
    """
    Implementation based upon
    https://github.com/eric-mitchell/direct-preference-optimization
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        beta: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

        Returns:
            DPO loss
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        # logits are maximized when losses a
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(beta * logits)

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (
            beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class Losses:
    """Losses factory."""

    _losses = {
        "DPOLoss": DPOLoss,
    }

    @classmethod
    def names(cls) -> KeysView:
        return cls._losses.keys()

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Losses.

        Args:
            name: losses name
        Returns:
            A class to build the Losses
        """
        return cls._losses.get(name)
