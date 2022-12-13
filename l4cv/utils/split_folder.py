import os
import argparse

import numpy as np


def split_folder(folder, test_proportion=0.1):

    dir_list = os.listdir(folder)

    folder_split = os.path.split(folder)
    folder_name = folder_split[-1]
    folder_lead = folder_split[0]

    test_folder = os.path.join(folder_lead, f"test_{folder_name}")

    make_test_cmd = f"mkdir {test_folder}"

    print(make_test_cmd)
    os.system(make_test_cmd)

    for subfolder in dir_list:

        subfolder_path = os.path.join(f"{test_folder}", subfolder)
        make_subfolder_cmd = f"mkdir {subfolder_path}"

        print(make_subfolder_cmd)
        os.system(make_subfolder_cmd)

        file_listdir = os.listdir(os.path.join(folder, subfolder))
        for filepath in file_listdir:
            if np.random.rand() < test_proportion:
                
                move_from = os.path.join(folder,subfolder,filepath)
                move_to = os.path.join(f"{test_folder}", subfolder, filepath)
                mv_cmd = f"mv {move_from} {move_to}"

                print(mv_cmd)
                os.system(mv_cmd)

    print("done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default="data/trees")

    args = parser.parse_args()

    folder = args.input_folder

    split_folder(folder)

