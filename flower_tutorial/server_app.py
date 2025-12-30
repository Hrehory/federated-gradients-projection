"""flower-tutorial: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower_tutorial.task import Net, load_centralized_dataset, test

from flower_tutorial.custom_strategy import CustomFedAdagrad

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    print(f"Starting training for {num_rounds} rounds with initial lr={lr}")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    if context.run_config["strategy"] == "default":
        print("Using default FedAvg strategy.")
        # Initialize FedAvg strategy
        fraction_train: float = context.run_config["fraction-train"]
        strategy = FedAvg(fraction_train=fraction_train)
    else:
        print("Using CustomFedAdagrad strategy.")
        fraction_train: float = context.run_config["fraction-custom-train"]
        fraction_evaluate: float = context.run_config["fraction-custom-evaluate"]

        strategy = CustomFedAdagrad(
            fraction_train=fraction_train,
            #fraction_evaluate=fraction_evaluate,
            # min_train_nodes=20,  # Optional config
            # min_evaluate_nodes=20,  # Optional config
            # min_available_nodes=20,  # Optional config
        )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    print(10*'=' + ' GLOBAL EVALUATE ' + 10*'=') 

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
    