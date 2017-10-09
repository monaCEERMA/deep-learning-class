import argparse

import fnmatch
import io
import os

from utils import update_progress, _order_files

parser = argparse.ArgumentParser(description='Processes timit for speech recognition task.')
parser.add_argument('--target_dir', default='timit_dataset/', help='Path to dataset')
args = parser.parse_args()

def create_manifest(data_path, tag, ordered=True):
    manifest_path = '%s_manifest.csv' % tag
    file_paths = []
    wav_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(data_path)
                 ### Modified from utils.py to change from "wav" to "WAV"...
                 for f in fnmatch.filter(files, '*.WAV')]
    size = len(wav_files)
    counter = 0
    for file_path in wav_files:
        file_paths.append(file_path.strip())
        counter += 1
        update_progress(counter / float(size))
    print('\n')
    if ordered:
        _order_files(file_paths)
    counter = 0
    with io.FileIO(manifest_path, "w") as file:
        for wav_path in file_paths:
            ### Modified from utils.py to remove a replace step...
            transcript_path = wav_path.replace('.WAV', '.TXT')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
            counter += 1
            update_progress(counter / float(size))
    print('\n')

def main():
    train_path = args.target_dir + '/TRAIN/'
    test_path = args.target_dir + '/TEST/'
    print ('\n', 'Creating manifests...')
    create_manifest(train_path, 'timit_train')
    create_manifest(test_path, 'timit_val')


if __name__ == '__main__':
    main()
