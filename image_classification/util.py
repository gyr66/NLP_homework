import os
import shutil


def move_images_to_folders(data_dir, classes):
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        for img_file in os.listdir(data_dir):
            if img_file.startswith(class_name):
                shutil.move(os.path.join(data_dir, img_file), class_dir)


if __name__ == "__main__":
    move_images_to_folders("data/small/val", ["cat", "dog"])