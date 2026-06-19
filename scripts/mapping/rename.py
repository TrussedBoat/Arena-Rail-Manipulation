import os
import re

directory = 'dataset_1/results'
offset = 150

# Sort in reverse to avoid overwriting files if the target name already exists
files = sorted(os.listdir(directory), reverse=True)

# Regex pattern to match both depth and frame files, capturing the prefix, number, and extension
pattern = re.compile(r"^(depth|frame)_(\d{6})\.(png|jpg)$")

for filename in files:
    match = pattern.match(filename)
    if match:
        prefix, num_str, ext = match.groups()
        
        # Add the offset
        new_num = int(num_str) + offset
        
        # Construct the new filename with 6-digit zero-padding
        new_filename = f"{prefix}_{new_num:06d}.{ext}"
        
        # Get full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename
        os.rename(old_path, new_path)

print("Files successfully renamed!")
