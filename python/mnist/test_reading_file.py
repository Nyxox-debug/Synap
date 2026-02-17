import struct
import numpy as np

def read_mnist_images(path):
    with open(path, "rb") as f:
        # Read header (big-endian integers)
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        print("Magic number:", magic)
        print("Number of images:", num_images)
        print("Image size:", rows, "x", cols)

        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)

    return images

images = read_mnist_images("t10k-images.idx3-ubyte")
print("First image shape:", images[0].shape)
print("First pixel value:", images[0][0][0])
