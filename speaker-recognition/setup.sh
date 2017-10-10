# NEED TO BE VERIFIED...
# Need to have MIT.zip in home dir

cd ~
git clone https://github.com/pytorch/audio.git
cd audio
python setup.py install
cd ~
pip install git+https://github.com/pytorch/tnt.git@master
unzip MIT.zip -d deep-learning-class/speaker-recognition/data/
cd deep-learning-class/speaker-recognition/data
mv MIT\ Mobile\ Device\ Speaker\ Verification\ Corpus/ mit_dataset
python mit.py
python mit_speaker_identification.py
cd ..
pip install -r requirements.txt
