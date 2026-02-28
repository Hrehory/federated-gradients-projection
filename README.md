# flower-tutorial: A Flower / PyTorch app

## Install dependencies and run

You can just type:

```bash
uv run --active flwr run . --run-config="strategy='custom' lr=0.01 num-server-rounds=3 model-type='standard'" 
```

It will explicitly create a needed .venv (**more than 7GB!**).

Once the `.venv` has been created - either by the command above or by `uv sync`, you can run the project using `launch.json`:


```
{
  "version": "0.2.0",
  "configurations": [

    {
      "name": "Launch flwr run",
      "type": "python",
      "request": "launch",
      "python": "${workspaceFolder}/flower-tutorial/.venv/bin/python",
      "program": "${workspaceFolder}/flower-tutorial/.venv/bin/flwr",
      "args": [
        "run",
        ".",
        "--run-config",
        "strategy='custom' lr=0.01 num-server-rounds=3 model-type='standard'"
      ],
      "cwd": "${workspaceFolder}/flower-tutorial",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/flower-tutorial"
      }
    },



    {
      "name": "Attach to Flower Server",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "justMyCode": false
    }

  ]
}

```

The expected structure of the project is as follows:

```
├── flower-tutorial
│   ├── final_model.pt
│   ├── flower_tutorial
│   │   ├── client_app.py
│   │   ├── custom_strategy.py
│   │   ├── __init__.py
│   │   ├── server_app.py
│   │   └── task.py
│   ├── .gitignore
│   ├── pyproject.toml
│   ├── .rayignore
│   ├── README.md
│   └── Research.md
└── .vscode
    └── launch.json
```


To debug the application, copy and paste the following snippet into `venv/lib/python3.13/site-packages/flwr/simulation/run_simulation.py`:


```
import debugpy
print("DEBUGPY: starting listen")
debugpy.listen(("0.0.0.0", 5678))
print("DEBUGPY: waiting for client")
debugpy.wait_for_client()
print("DEBUGPY: attached")
```

Next, install the debugger by running `debugpy` by typing `uv pip install debugpy` and launch the application using `uv run --active ...`. Finally, attach the debugger to the Flower Server in VS Code by pressing `CTRL+SHIFT+D` and selecting your "Attach" profile.