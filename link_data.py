import os
import numpy as np
import tqdm
original_folder = "/data/laiyan/datasets/ScanNet/"
extracted_folder = "/data/laiyan/datasets/ScanNet/extracted/"
cases = ['scans', 'scans_test']
# Iterate over each subfolder in the source folder
for case in cases:
    case_folder = os.path.join(extracted_folder, case)
    scan_folders = os.listdir(case_folder)
    scan_folders.sort()
    for scan_folder in tqdm.tqdm(scan_folders):
        org_files=os.listdir(os.path.join(original_folder,case,scan_folder))
        ext_files=os.listdir(os.path.join(extracted_folder,case,scan_folder))
        org_files.sort()
        ext_files.sort()
        for file in org_files:
            if os.path.exists(os.path.join(extracted_folder,case,scan_folder,file)):
                os.remove(os.path.join(extracted_folder,case,scan_folder,file))

            print(os.path.join(original_folder,case,scan_folder,file),'-------->',os.path.join(extracted_folder,case,scan_folder,file))
            # os.symlink(os.path.join(original_folder,case,scan_folder,file),os.path.join(extracted_folder,case,scan_folder,file))
            exit()

