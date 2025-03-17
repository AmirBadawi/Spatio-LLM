import os
import shutil

# Set the path to the parent directory containing the folders you want to process.
parent_dir = './Geolife Trajectories 1.3/Data'

# Iterate over all items in the parent directory.
for item in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, item)
    
    # Check if the item is a directory (folder).
    if os.path.isdir(folder_path):
        # Get the list of items in the folder.
        contents = os.listdir(folder_path)
        count = len(contents)
        
        # If the folder contains exactly one item, remove it.
        if count == 1:
            print(f"Removing folder: {folder_path} (contains {count} item)")
            shutil.rmtree(folder_path)
        else:
            print(f"Keeping folder: {folder_path} (contains {count} items)")
