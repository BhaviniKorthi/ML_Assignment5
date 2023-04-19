import os
from sklearn.model_selection import train_test_split
from shutil import copyfile

classes = ["SHEEP", "BEAR"]
for type in classes:
    source_dir = "images/"+ type

    # Set the destination directories for the train and test sets
    train_dir = "bear_vs_sheep/train/"+type
    test_dir = "bear_vs_sheep/test/"+type


    image_filenames = os.listdir(source_dir)


    train_filenames, test_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)


    for filename in train_filenames:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(train_dir, filename)
        copyfile(source_path, dest_path)

    # Copy the test images to the test directory
    for filename in test_filenames:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(test_dir, filename)
        copyfile(source_path, dest_path)
    
print(len(os.listdir("bear_vs_sheep/test/SHEEP")))
print(len(os.listdir("bear_vs_sheep/train/SHEEP")))
print(len(os.listdir("bear_vs_sheep/test/BEAR")))
print(len(os.listdir("bear_vs_sheep/train/BEAR")))