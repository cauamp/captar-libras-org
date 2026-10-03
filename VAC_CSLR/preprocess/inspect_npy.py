import numpy as np
import glob
import os
from dotenv import load_dotenv

load_dotenv("./config.env")

DATASET = os.getenv('DATASET')
# Locate all .npy files in specified directories
files = glob.glob(f'./{DATASET}/*.npy')

# Dictionary to store metadata and preview from .npy files
data_list = {}

for file_path in files:
    try:
        # Load the .npy file with allow_pickle=True
        data = np.load(file_path, allow_pickle=True)
        
        # Collect metadata and preview safely
        shape = getattr(data, 'shape', 'N/A')  # Use 'getattr' for non-NumPy objects
        dtype = getattr(data, 'dtype', 'N/A')
        size = getattr(data, 'size', 'N/A')

        # Preview content safely
        if isinstance(data, np.ndarray):
            if data.ndim > 0:
                preview = data[:10]
            else:
                preview = data  # 0-dimensional array, show it directly
        else:
            preview = data  # Non-NumPy object (e.g., list or dictionary)
                    
        # Store information in dictionary
        data_list[file_path] = {
            "shape": shape,
            "dtype": dtype,
            "size": size,
            "preview": preview
        }
        
        # Display basic information
        print(f"File: {file_path}")
        print(f"Type: {type(data)}")
        print(f"Shape: {shape}")
        print(f"Data Type: {dtype}")
        print(f"Size: {size} elements")
        print(f"Content Preview: {preview}\n")

    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

# Write collected data information to an output file
with open('./npy_inspect.txt', 'w') as f:
    for file_path, metadata in data_list.items():
        f.write(f"File: {file_path}\n")
        f.write(f"Shape: {metadata['shape']}\n")
        f.write(f"Data Type: {metadata['dtype']}\n")
        f.write(f"Size: {metadata['size']} elements\n")
        f.write(f"Content Preview:\n{metadata['preview']}\n")
        f.write("\n\n")
