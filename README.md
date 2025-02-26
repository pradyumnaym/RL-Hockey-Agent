### RL Agent for the Laser Hockey Environment

### Installation Instructions
The codebase has been tested with python3.10+. We recommend using a virtual environment:

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

To train an agent, use the `train.py` script. The training configuration is managed by Hydra and can be found in the `configs/` directory.

You can modify the training parameters by editing the configuration files in `configs/td3_v1.yaml` or `configs/sac_v1.yaml`.

**Example Command:**
```bash
python train.py --config-name=td3_v1
```
You can override specific parameters using Hydra's syntax:
```bash
python train.py --config-name=td3_v1 parameter=value
```

## Evaluation

To evaluate a trained agent, use the `eval.py` script. This will run 1000 episodes and report the average reward and win/loss rates.

**Example Command:**
```bash
python eval.py --model-path=path/to/trained/model --opponent-type=weak
```
For a custom opponent:
```bash
python eval.py --model-path=path/to/trained/model --opponent-type=custom --opponent-path=path/to/custom/opponent
```

## Testing and Visualization

The `test.py` script allows you to visualize the agent's gameplay by generating GIF animations. It will create 10 game episodes and save them as GIFs.

**Example Command:**
```bash
python test.py --model-path=path/to/trained/model --opponent-type=strong
```
For a custom opponent:
```bash
python test.py --model-path=path/to/trained/model --opponent-type=custom --opponent-path=path/to/custom/opponent
```

## Arguments

### eval.py and test.py
- `--model-path`: Path to the trained model file (required)
- `--opponent-type`: Type of opponent - 'weak', 'strong', or 'custom' (required)
- `--opponent-path`: Path to the custom opponent model file (required if opponent-type is 'custom')

### train.py
- `--config-name`: Name of the configuration file to use (default: td3_v1)
- Additional configuration options can be overridden via command line using Hydra's syntax

### Configuring Opponents
To configure the agent to play against a single or multiple opponents, you can adjust the probabilities in the YAML configuration files located in the `configs/` directory. The fields `weak_prob`, `strong_prob`, and `self_prob` determine the likelihood of facing each type of opponent during training or evaluation.

- **Single Opponent**: Set the probability of the desired opponent type to 1.0 and the others to 0.0. For example, to always face a weak opponent:
  ```yaml
  opponent_pooler:
      weak_prob: 1.0
      strong_prob: 0.0
      self_prob: 0.0
  ```

- **Multiple Opponents**: Distribute the probabilities among the opponent types to reflect the desired frequency of facing each type. For example, to face weak opponents 20% of the time, strong opponents 30% of the time, and self 50% of the time:
  ```yaml
  opponent_pooler:
      weak_prob: 0.2
      strong_prob: 0.3
      self_prob: 0.5
  ```

- **Curriculum Learning**: The opponent pooler also supports curriculum learning, where the agent is trained at increasingly difficult opponents. To provide a curriculum, simply provide a list of probabilities for each type of opponent, and the `max_episodes` variable from the hydra configuration (required to evenly space the different stages).
  ```yaml
  opponent_pooler:
    weak_prob:      [0.1, 0.1, 0.2, 0.2, 0.1, 0.0]
    strong_prob:    [0.7, 0.5, 0.2, 0.2, 0.1, 0.0]
    self_prob:      [0.0, 0.2, 0.3, 0.2, 0.5, 0.9]
    custom_prob:    [0.2, 0.2, 0.3, 0.4, 0.3, 0.1]
    custom_weight_paths: ['checkpoints/model_best_sac_14.pth',                      # best sac agent        
                          'outputs/2025-02-21/23-13-30/model_best.pth',             # best strong agent
                          'outputs/2025-02-20/22-41-32/model_best.pth',             # best weak agent
                          ]
    max_episodes: ${max_episodes}
    update_self_opponent_freq: 1000
  ```

Adjust these probabilities according to your training or evaluation strategy.

## Notes
- The evaluation script runs 1000 episodes by default
- The test script generates 10 GIF animations by default
- Maximum steps per episode is set to 250

The generated GIFs will be saved in the `gifs/` directory.
