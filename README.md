# flower-tutorial: A Flower / PyTorch app


## Badania

### 30.12.2025

Przy: flwr run . --run-config="strategy='custom' lr=0.01 num-server-rounds=3"

Wszystko jest OK , natomiast:

flwr run . --run-config="strategy='custom' lr=0.1 num-server-rounds=8"

Model się nie trenuje dla przypadku bez BatchNorm.

Co ciekawe, przy podmianie CustomFedAdagrad(FedAvg) -> CustomFedAdagrad(FedAdagrad), model z BatchNorm powoduje błąd:

TypeError: Invalid arguments for Array. Expected either a PyTorch tensor, a NumPy ndarray, or explicit dtype/shape/stype/data values.


## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `flower-tutorial` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
