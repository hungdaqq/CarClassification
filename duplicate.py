from PIL import Image
import imagehash
import os
from collections import defaultdict

def calculate_image_hash(image_path, hash_size=8):
    img = Image.open(image_path)
    return imagehash.phash(img, hash_size=hash_size)

def find_duplicate_images(directory, hash_size=8):
    image_hashes = defaultdict(list)

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower():
                image_path = os.path.join(root, filename)
                image_hash = calculate_image_hash(image_path, hash_size=hash_size)
                image_hashes[image_hash].append(image_path)

    duplicate_images = {image_hash: image_list for image_hash, image_list in image_hashes.items() if len(image_list) > 1}
    return duplicate_images

def delete_duplicate_images(directory, hash_size=8):
    duplicate_images = find_duplicate_images(directory, hash_size=hash_size)

    for image_hash, image_list in duplicate_images.items():
        for image_path in image_list[1:]:
            try:
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            except OSError as e:
                print(f"Error deleting {image_path}: {e}")

if __name__ == "__main__":
    directory_to_check = input('Directory?\n')
    
    # Display the duplicate images first before proceeding to delete
    duplicate_images = find_duplicate_images(directory_to_check)
    if duplicate_images:
        print("Duplicate images found:")
        for image_hash, image_list in duplicate_images.items():
            print(f"Hash: {image_hash}")
            for image_path in image_list:
                print(f"- {image_path}")

        confirmation = input("Do you want to delete the duplicate images? (y/n): ").strip().lower()
        if confirmation == "y":
            delete_duplicate_images(directory_to_check)
        else:
            print("Operation aborted.")
    else:
        print("No duplicate images found.")
