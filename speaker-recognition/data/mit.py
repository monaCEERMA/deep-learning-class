import argparse

import fnmatch
import io
import os

from utils import update_progress, _order_files


parser = argparse.ArgumentParser(description='Processes mit.')
parser.add_argument('--target_dir', default='mit_dataset/', help='Path to dataset')
args = parser.parse_args()

def create_manifest(data_path, tag, ordered=True):
    manifest_path = '%s_manifest.csv' % tag
    file_paths = []
    wav_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(data_path)
                 for f in fnmatch.filter(files, '*.wav')]
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
            ### Modified from utils.py to remove a replace step and add "_16k"...
            transcript_path = wav_path.replace('_16k.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
            counter += 1
            update_progress(counter / float(size))
    print('\n')

def main():
    train_path = args.target_dir + '/Enroll_Session1/'
    test_path = args.target_dir + '/Enroll_Session2/'
    imposter_path = args.target_dir + '/Imposter/'
    print ('\n', 'Creating manifests...')
    create_manifest(train_path, 'mit_train')
    create_manifest(test_path, 'mit_val')
    create_manifest(imposter_path, 'mit_imposter')


if __name__ == '__main__':
    main()
