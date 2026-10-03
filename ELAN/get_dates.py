import os
import time

# Change to the ELAN-annotation directory
with open('./dataset_dates.txt', 'w') as f:
    os.chdir('/srv/projects2/captarlibras_finep/sign-to-text/dataset/captar-libras_16-10/raw_data/train')
    
    files = os.listdir('.')
    file_info = []

    for file in files:
        # Get the full path of the file
        full_path = os.path.join('.', file)
        # Get the modification time and format it
        mod_time = time.ctime(os.path.getmtime(full_path))
        file_info.append((file, mod_time, os.path.getmtime(full_path)))

    # Sort file_info by file name
    file_info.sort(key=lambda x: x[2])  
    for file, mod_time, _ in file_info:
            f.write(f"{file} - {mod_time}\n")
