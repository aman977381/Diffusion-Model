import os
import zipfile
import gdown
import shutil
from sklearn.model_selection import train_test_split

def download_and_extract_data(shared_link: str, download_dir: str) -> str:
    '''Fetch and extract data from the shared link'''
    os.makedirs(download_dir, exist_ok=True)
    zip_file_path = os.path.join(download_dir, "data.zip")

    file_id = shared_link.split("/")[-2]
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', zip_file_path, quiet=False)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    
    os.remove(zip_file_path)  # Clean up the zip file
    return download_dir

def list_directory_contents(directory: str) -> None:
    '''List contents of the directory'''
    for root, dirs, files in os.walk(directory):
        print(f"Found directory: {root}")
        for file in files:
            print(f"File: {file}")

def collect_images(src_dir: str) -> list:
    '''Collect all .jpg images from the directory and its subdirectories'''
    images = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                images.append(os.path.join(root, file))
    return images

def keep_random_subset(src_dir: str, dest_dir: str, keep_ratio: float = 0.2) -> None:
    '''Keep a random subset of images'''
    os.makedirs(dest_dir, exist_ok=True)

    # Collect all images
    images = collect_images(src_dir)
    print(f"Found {len(images)} images.")

    if not images:
        print("No images found in the directory.")
        return

    # Shuffle and select a subset of images
    selected_images, _ = train_test_split(images, train_size=keep_ratio, random_state=42)

    for img_path in selected_images:
        img_filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_dir, img_filename)
        shutil.copy(img_path, dest_path)

    print(f"Selected and copied {len(selected_images)} images to {dest_dir}")

    shutil.rmtree("./artifacts/cats")

def main():
    shared_link = "https://drive.google.com/file/d/1UWwdGu_z4HbiVZbqwCmW-XB2lirDSm_x/view?usp=sharing"
    base_dir = "./artifacts"
    dest_dir = os.path.join(base_dir, 'cat')

    extracted_dir = download_and_extract_data(shared_link, base_dir)
    print(f"Data extracted to {extracted_dir}")

    keep_random_subset(extracted_dir, dest_dir, keep_ratio=0.2)


if __name__ == "__main__":
    main()
