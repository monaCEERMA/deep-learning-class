# Code originally from https://github.com/SeanNaren/deepspeech.pytorch

NEED TO BE UPDATE...

$ cd ~
$ git clone https://github.com/pytorch/audio.git
cd audio
python setup.py install
pip install git+https://github.com/pytorch/tnt.git@master
$ unzip MIT.zip –d pytorch-deepspeaker/data/
cd pytorch-deepspeaker/data
$ mv MIT\ Mobile\ Device\ Speaker\ Verification\ Corpus/ mit_dataset
$ python mit.py
$ python mit_speaker_identification.py
cd ..
pip install -r requirements.txt
./train.sh

