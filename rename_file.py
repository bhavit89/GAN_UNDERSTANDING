import os 


directory = "/home/bhavit/Desktop/GAN/Cars Dataset/test/Toyota Innova"


files = os.listdir(directory)

for filename in  files:
    if not filename.startswith("innova_"):
        old_path = os.path.join(directory,filename)
        new_name = f"innova_{filename}"
        new_path = os.path.join(directory ,new_name)

        os.rename(old_path,new_path)


print("Renamed Successfull")