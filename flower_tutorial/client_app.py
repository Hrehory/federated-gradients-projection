"""flower-tutorial: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_tutorial.task import Net, NetBN, load_data
from flower_tutorial.task import test as test_fn
from flower_tutorial.task import local_train_and_return_delta  # <-- you add this in task.py

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train locally (SGD), convert delta to an 'effective gradient', send to server."""

    # 1) Build model
    if context.run_config["model-type"] == "bn":
        print("Client: Using BatchNorm model")
        model = NetBN()
    else:
        print("Client: Using standard model")
        model = Net()

    # 2) Load global weights from server
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) Load local data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # 4) Local train (SGD) and get delta = w_local - w_global for PARAMETERS
    local_epochs = int(context.run_config["local-epochs"])
    lr_local = float(msg.content["config"]["lr"])  # comes from server config

    print(10 * "=" + f" LOCAL TRAIN (delta->grad), lr_local={lr_local} " + 10 * "=")

    delta_dict, avg_loss, steps = local_train_and_return_delta(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr_local,
        device=device,
    )

    # 5) Convert delta to "effective gradient": g_eff = -delta / lr_local
    #    (So the server can still do: w <- w - lr_server * g_eff)
    if lr_local <= 0:
        raise ValueError(f"lr_local must be > 0, got {lr_local}")

    g_eff = {k: (-v / lr_local) for k, v in delta_dict.items()}

    # 6) Send to server
    content = RecordDict(
        {
            "arrays": ArrayRecord(g_eff),
            "metrics": MetricRecord(
                {
                    "train_loss": float(avg_loss),
                    "num-examples": len(trainloader.dataset),
                    "steps": int(steps),
                    "lr_local": lr_local,
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local validation data."""
    print(10 * "=" + " LOCAL EVALUATE " + 10 * "=")

    # Build model
    if context.run_config["model-type"] == "bn":
        print("Client eval: Using BatchNorm model")
        model = NetBN()
    else:
        print("Client eval: Using standard model")
        model = Net()

    # Load weights
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load local validation data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Evaluate
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Reply
    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
