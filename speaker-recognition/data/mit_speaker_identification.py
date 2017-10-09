import argparse
import os

parser = argparse.ArgumentParser(description='Processes mit to speaker identification task.')
parser.add_argument('--target_dir', default='mit_dataset', help='Path to dataset')
args = parser.parse_args()


def process_manifests(lines, file, path):
    label = 0
    for line in range(len(lines)):
        speaker = lines[line].split(',')[0].strip(os.path.abspath(args.target_dir)).split('/')[1]
        #speaker = lines[line].split('/')[8] # Hardcoded... Diferent for each instalation...
        lines[line] = lines[line].strip("\n")
        if speaker not in speakers:
            speakers[speaker] = label
            label += 1
        file.write(lines[line] + "," + str(speakers[speaker]) + "\n")
    print(speakers)


def main():
    name = 'mit'
    global speakers
    speakers = {}
    train_path = args.target_dir + 'Enroll_Session1'
    test_path = args.target_dir + 'Enroll_Session2'
    with open(name + "_train_speaker_identification.csv", 'w') as train_file:
        with open(name + "_train_manifest.csv", 'r') as file:
            lines = file.readlines()
            process_manifests(lines, train_file, train_path)
    with open(name + "_val_speaker_identification.csv", 'w') as val_file:
        with open(name + "_val_manifest.csv", 'r') as file:
            lines = file.readlines()
            process_manifests(lines, val_file, test_path)


if __name__ == '__main__':
    main()
