"""
Script to create a pkl file with this format:
mypickle = {"Filenames": list_of_file_paths, "Labels": class_condition_labels}
"""
import argparse
import os
import glob
import pickle


def do_parsing():
    parser = argparse.ArgumentParser(description='Create dataset metadata in pickle format')
    parser.add_argument('--images_root_dir', type=str, required=True,
                        help='Images root directory, subdirectories must be related to classes')
    parser.add_argument('--pkl_filepath', type=str, required=True,
                        help='Output pkl filepath to be used for dataset creation')
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    walker = os.walk(args.images_root_dir)

    _, dirs, _ = next(walker)

    classes = sorted(dirs)

    metadata = dict()
    metadata["Labels"] = list()
    metadata["Filenames"] = list()

    for idx, class_name in enumerate(classes):
        filenames = glob.glob(os.path.join(args.images_root_dir, class_name) + "/*.jpg")
        for filename in filenames:
            metadata["Labels"].append(idx)
            metadata["Filenames"].append(os.path.join(class_name, os.path.basename(filename)))

    os.makedirs(os.path.dirname(args.pkl_filepath), exist_ok=True)
    with open(args.pkl_filepath, "wb") as fp:
        pickle.dump(metadata, fp)
    print(f"pkl file saved in {args.pkl_filepath}")


if __name__ == "__main__":
    main()
