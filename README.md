# Learning structured sparsity for efficient basecalling

This repository contains the code to train basecaller neural networks while
learning structured sparsity in the LSTM layers. By penalizing nonzero mask
entries on the input-to-hidden and hidden-to-hidden weight matrices, the model
regularizes itself, leading to a sparse model that can be converted to smaller
(standard) LSTM layers for faster and more efficient inference.

The masked LSTM layer that's used during training to learn structured sparsity
can be found in [src/layers/masked_lstm.py](./src/layers/masked_lstm.py)

## Installation & dependencies
This code has been tested on Python 3.6 and 3.8-3.10. Using python 3.6, you
may run into issues with dependencies; installing an older version usually
does the trick.

A working CUDA installation and a GPU with about 16 GB of memory is required,
more is recommended: it allows increasing the batch size to speed up training.
For the default batch size 64, --starting-lr 0.001 is the default. For batch
size 128, we used --starting-lr 0.0015. On an A100, 5 epochs of training takes
around 23 to 24 hours.

```sh
git clone https://github.com/meesfrensel/efficient-basecallers.git
cd efficient-basecallers
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you get dependency conflicts, try to manually install `seqdist==0.0.3` first
and then manually upgrade to `numpy==1.21.5`, as was found as a solution in
this [issue](https://github.com/marcpaga/basecalling_architectures/issues/4#issuecomment-1645361245).

## Training
The model can be trained on whatever data you like, but do note that it must be
preprocessed (resquiggled, converted to npz files). For the paper, we refer you
to https://github.com/marcpaga/nanopore_benchmark. This repository contains
scripts to download the standard datasets, train/test splits and scripts for
evaluating basecaller performance.

Training is done using [scripts/train.py](./scripts/train.py):

```sh
python ./scripts/train.py
    --data-dir ./demo_data/nn_input
    --output-dir ./test_output
```

Study the other available parameters with `python ./scripts/train.py --help`.

## Basecalling & evaluation
Basecall you test data with [scripts/basecall.py](./scripts/basecall.py):

```sh
python ./scripts/basecall.py
    --fast5-dir ./demo_data/fast5
    --checkpoint ./demo_data/model_checkpoint.pt
    --output demo_basecalls.fastq
```

Again, study the other parameters with `python ./scripts/basecall.py --help`.

Evaluation is done with the scripts and code from
https://github.com/marcpaga/nanopore_benchmark.

## License
This work is available under the Apache 2.0 license. See the [LICENSE](./LICENSE)
file for more information.

---

Copyright 2024 Mees Frensel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
