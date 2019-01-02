# APE-NPI
Source code of paper "Automatic Post-Editing of Machine Translation: A Neural Programmer-Interpreter Approach" - EMNLP2018

## Dependencies

* [TensorFlow 1.2+ for Python 3](https://www.tensorflow.org/get_started/os_setup.html)
* YAML and Matplotlib modules for Python 3: `sudo apt-get install python3-yaml python3-matplotlib`
* A recent NVIDIA GPU

## How to use

Train a model (CONFIG is a YAML configuration file, such as `config/default.yaml`):

    ./seq2seq.sh CONFIG --train -v 


Translate text using an existing model:

    ./seq2seq.sh CONFIG --decode FILE_TO_TRANSLATE --output OUTPUT_FILE
or for interactive decoding:

    ./seq2seq.sh CONFIG --decode

## APE experiments
* Training scripts and configuration for all experiments in the paper can be found under `./experiments` folder

## Credits

* This project is implemented based on [seq2seq project](https://github.com/eske/seq2seq/)
