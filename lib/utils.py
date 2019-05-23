#!/usr/bin python3
""" Utilities available across all scripts """

import logging, os
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]


def get_folder(path, make_folder=True):
    """ Return a path to a folder, creating it if it doesn't exist """
    output_dir = Path(path)
    if not make_folder and not output_dir.exists():
        logger.debug("%s does not exist", path)
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
    
    
def get_image_paths(directory):
    dir_contents = list()
    dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
    
    for file in dir_scanned:
        if any([file.name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS]):
            dir_contents.append(file.path)
            
    return dir_contents
