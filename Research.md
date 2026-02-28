## Badania

### 9.02.2026

Przerobienie kodu na przesyłanie gradientów i placeholder (projet_identity) na przyszłe badania dot. projekcji (dual cone)

flwr run . --run-config="strategy='custom' lr=0.02 num-server-rounds=100 model-type='bn'"



### 31.12.2025

Brak badań, poprawki, by działał Fedadagrad i by można było podać w cmd model-type='standard'/'bn':

flwr run . --run-config="strategy='custom' lr=0.01 num-server-rounds=3 model-type='standard'"



### 30.12.2025

Przy: flwr run . --run-config="strategy='custom' lr=0.01 num-server-rounds=3"

Wszystko jest OK , natomiast:

flwr run . --run-config="strategy='custom' lr=0.1 num-server-rounds=8"

Model się nie trenuje dla przypadku bez BatchNorm.

Co ciekawe, przy podmianie CustomFedAdagrad(FedAvg) -> CustomFedAdagrad(FedAdagrad), model z BatchNorm powoduje błąd:

TypeError: Invalid arguments for Array. Expected either a PyTorch tensor, a NumPy ndarray, or explicit dtype/shape/stype/data values.
