# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset:
- French
- Kinyarwanda
- Italian
- English

# Results

| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| French | 2020-06-22 | train_fr.yaml | No | 5.36 | 15.87 | 6.54 | 17.70 | [model](https://drive.google.com/drive/folders/13i7rdgVX7-qZ94Rtj6OdUgU-S6BbKKvw?usp=sharing) | 2xV100 16GB |
| Kinyarwanda | 2020-06-22 | train_rw.yaml | No | 7.30 | 21.36 | 9.55 | 24.27 | Not Avail. | 2xV100 16GB |
| Italian | 2020-06-22 | train_it.yaml | No | 5.14 | 15.59 | 5.40 | 16.61 | [model](https://drive.google.com/drive/folders/1asxPsY1EBGHIpIFhBtUi9oiyR6C7gC0g?usp=sharing) | 2xV100 16GB |
| English | 2020-06-22 | train_it.yaml | No | 8.66 | 20.16 | 12.93 | 24.89 | Not Avail. | 2xV100 16GB |
