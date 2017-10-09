import argparse
import os

parser = argparse.ArgumentParser(description='Processes timit redistribution in preparation for speaker recognition.')
parser.add_argument('--target_dir', default='timit_dataset/', help='Path to dataset')
args = parser.parse_args()

def process_manifests(lines, train_file, val_file, path):
    speakers = {}
    for line in range(len(lines)):
        speaker = lines[line].split(',')[0].strip(os.path.abspath(path)).split('/')[1]
        if speaker not in speakers:
            speakers[speaker] = 1
        else:
            speakers[speaker] += 1
        if speakers[speaker] % 5 != 4:
            train_file.write(lines[line])
        else:
            val_file.write(lines[line])
    print(speakers)


def main():
    name = 'timit'
    train_path = args.target_dir + '/TRAIN/'
    test_path = args.target_dir + '/TEST/'
    with open(name + "_train_redistributed_manifest.csv", 'w') as train_file:
        with open(name + "_val_redistributed_manifest.csv", 'w') as val_file:
            with open(name + "_train_manifest.csv", 'r') as file:
                lines = file.readlines()
                process_manifests(lines, train_file, val_file, train_path)
            with open(name + "_val_manifest.csv", 'r') as file:
                lines = file.readlines()
                process_manifests(lines, train_file, val_file, test_path)


if __name__ == '__main__':
    main()
