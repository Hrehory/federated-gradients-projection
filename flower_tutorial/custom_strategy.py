from typing import Iterable, Tuple, Optional, Dict

import torch

from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord



class CustomFedAdagrad(FedAdagrad):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.1 every 4 rounds
        if server_round % 4 == 0 and server_round > 0:
            config["lr"] *= 0.1
            print("LR decreased to:", config["lr"])
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)


def proj_identity(t: torch.Tensor) -> torch.Tensor:
    return t

class GradientFedAvg(FedAvg):
    """Clients send gradients; server applies SGD step: w <- w - lr * g."""

    def __init__(
        self,
        lr: float,
        projector=proj_identity,          # replace later with cone projection
        project_each_client: bool = False,
        project_agg: bool = False,
    ):
        super().__init__()
        self.lr = float(lr)
        self.projector = projector
        self.project_each_client = project_each_client
        self.project_agg = project_agg

        self._current_arrays: Optional[ArrayRecord] = None
        self.last_client_grads: Dict[int, Dict[str, torch.Tensor]] = {}


    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # Cache current global model for aggregate_train
        self._current_arrays = arrays

        # Send anything you want
        config["lr"] = self.lr
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies = list(replies)
        if self._current_arrays is None or len(replies) == 0:
            return None, None

        # Current global weights (parameters + buffers)
        w = self._current_arrays.to_torch_state_dict()

        # Canonical gradient keys (what the client sends)
        g0 = replies[0].content["arrays"].to_torch_state_dict()
        grad_keys = list(g0.keys())

        # Sanity: ensure keys exist in w
        missing = [k for k in grad_keys if k not in w]
        if missing:
            raise KeyError(f"Gradient keys missing in server weights: {missing[:5]} (total {len(missing)})")

        # Init aggregated gradient dict
        g_agg: Dict[str, torch.Tensor] = {k: torch.zeros_like(w[k]) for k in grad_keys}

        self.last_client_grads.clear()
        W = 0.0
        losses = []

        for msg in replies:
            g = msg.content["arrays"].to_torch_state_dict()
            n = float(msg.content["metrics"]["num-examples"])
            W += n
            losses.append(float(msg.content["metrics"]["train_loss"]))

            # (Optional) per-client projection
            if self.project_each_client:
                g = {k: self.projector(g[k]) for k in grad_keys}

            # Keep for inspection (optional)
            self.last_client_grads[msg.metadata.src_node_id] = {k: g[k].detach().cpu() for k in grad_keys}

            for k in grad_keys:
                g_agg[k] += n * g[k]

        if W <= 0:
            return None, None

        for k in grad_keys:
            g_agg[k] /= W

        # (Optional) aggregate projection
        if self.project_agg:
            g_agg = {k: self.projector(g_agg[k]) for k in grad_keys}

        # Apply server SGD update
        new_w = dict(w)
        for k in grad_keys:
            new_w[k] = (w[k] - self.lr * g_agg[k]).detach()
            

        # Diagnostics
        with torch.no_grad():
            agg_grad_norm = torch.sqrt(sum((g_agg[k].float() ** 2).sum() for k in grad_keys)).item()
            max_abs_grad = max(g_agg[k].abs().max().item() for k in grad_keys)

            delta_w_norm = torch.sqrt(sum(((new_w[k] - w[k]).float() ** 2).sum() for k in grad_keys)).item()
            max_abs_delta = max((new_w[k] - w[k]).abs().max().item() for k in grad_keys)

        avg_loss = float(sum(losses) / max(1, len(losses)))

        print(
            f"[round {server_round}] "
            f"train_loss={avg_loss:.6f} "
            f"agg_grad_norm={agg_grad_norm:.6e} max|g|={max_abs_grad:.6e} "
            f"delta_w_norm={delta_w_norm:.6e} max|dw|={max_abs_delta:.6e} "
            f"lr={self.lr}"
        )

        return ArrayRecord(new_w), MetricRecord({"train_loss": avg_loss})


    