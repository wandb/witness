# Finding The Witness Puzzle Patterns

This repo is example code and data for training a CNN to identify patterns from the video game "The Witness", using python3 and tensorflow 1.13.

![Witness deep learning](readme-image.jpg)

## To train the model:

```sh
# Optional: create a virtualenv
pyenv virtualenv 3.6.4 witness-3.6
pyenv local witness-3.6
echo witness-3.6 > .python-version

# Install requirements.
pip install -U -r requirements.txt

# Install tensorflow GPU support if needed
pip install tensorflow-gpu

# Link to W&B -- this will save your run results.
wandb init

# Process data from /data/all into a training set in /data/train and validation set in /data/valid.
./process.py

# Train your model, and save the best one into the /model folder!
./train.py
```

## To visualize output:

```sh
# Run a prediction on all entries in your validation set.
./predict.py

# Generate a visualization of every layer in the model.
./visualize.py
```

## To submit to W&B:

Weights & Biases is running a public benchmark, to which anyone can share their model improvements. To submit your results, follow these steps.

* [Create a W&B account](https://app.wandb.ai)
* Follow the above instructions to run a run in your personal project.
* You can submit a run to the benchmark from the runs table:

![Submit a run](https://camo.githubusercontent.com/132cdb4665ced6f9f303a7f4ea464c03b94b68bd/68747470733a2f2f6170702e77616e64622e61692f7374617469632f6d656469612f7375626d69745f62656e63686d61726b5f72756e2e65323836646130642e706e67)
