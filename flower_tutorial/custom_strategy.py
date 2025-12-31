from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message


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
