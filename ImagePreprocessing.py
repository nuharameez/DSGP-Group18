import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature

def preprocess_and_display(images):
    plt.figure(figsize=(15, 10))

    for i, image_path in enumerate(images, 1):
        # Load the X-ray image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Data cleaning and preprocessing
        # Example: Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Example: Histogram equalization for contrast enhancement
        equalized_image = cv2.equalizeHist(blurred_image)

        # Thresholding
        _, thresholded_image = cv2.threshold(equalized_image, 150, 255, cv2.THRESH_BINARY)

        # Feature extraction
        # Example: Canny edge detection
        edges = cv2.Canny(equalized_image, 30, 150)

        # Example: Local Binary Pattern (LBP) for texture features
        radius = 3
        num_points = 8 * radius
        lbp = feature.local_binary_pattern(equalized_image, num_points, radius, method='uniform')

        # Display the original and processed images
        plt.subplot(2, len(images), i)
        plt.imshow(image, cmap='gray')
        plt.title(f'Original Image {i}')

        plt.subplot(2, len(images), i + len(images))
        plt.imshow(thresholded_image, cmap='gray')
        plt.title(f'Thresholded Image {i}')

    plt.tight_layout()
    plt.show()


# Example usage with a list of image paths
image_paths = [r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\1\9613237R.png",
               r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\2\9755851L.png",
               r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\3\9002430L.png",
               r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\4\9057150L.png"]
preprocess_and_display(image_paths)

# import cv2
# import numpy as np
# from skimage import feature, exposure, filters
# from skimage.util import img_as_ubyte, img_as_float
# from skimage.morphology import reconstruction
# from scipy.ndimage import gaussian_filter
# from skimage.transform import resize
# import matplotlib.pyplot as plt
#
# def square(img):
#     diff = img.shape[0] - img.shape[1]
#     if diff % 2 == 0:
#         pad1 = int(np.floor(np.abs(diff) / 2))
#         pad2 = int(np.floor(np.abs(diff) / 2))
#     else:
#         pad1 = int(np.floor(np.abs(diff) / 2))
#         pad2 = int(np.floor(np.abs(diff) / 2)) + 1
#
#     if diff == 0:
#         return img
#     elif diff > 0:
#         return np.pad(img, [(0, 0), (pad1, pad2)], 'constant', constant_values=(0))
#     elif diff < 0:
#         return np.pad(img, [(pad1, pad2), (0, 0)], 'constant', constant_values=(0))
#
# def contrast(img):
#     img = img_as_ubyte(img / 255)
#     img = img_as_float(img)
#     img = gaussian_filter(img, 1)
#     h = 0.8
#     seed = img - h
#     mask = img
#     dilated = reconstruction(seed, mask, method='dilation')
#     img_dil_adapteq = exposure.equalize_adapthist(img - dilated, clip_limit=0.03)
#     return img_dil_adapteq
#
# def threshold(img):
#     img_threshold = filters.threshold_li(img)
#     img_new = np.ones(img.shape[:2], dtype="float")
#     img_new[(img < img_threshold) | (img > 250)] = 0
#     return img_new
#
# def resize_crop(img):
#     img = resize(img, (224, 224), anti_aliasing=True)
#     return img.flatten()
#
# def preprocess_and_display(images):
#     plt.figure(figsize=(15, 10))
#
#     for i, image_path in enumerate(images, 1):
#         # Load the X-ray image
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#         # Additional preprocessing using provided functions
#         image_original = image.copy()
#         image = square(image)
#         image = contrast(image)
#         image = threshold(image)
#         image = resize_crop(image)
#
#         # Display the original and processed images
#         plt.subplot(2, len(images), i)
#         plt.imshow(image_original, cmap='gray')
#         plt.title(f'Original Image {i}')
#
#         plt.subplot(2, len(images), i + len(images))
#         plt.imshow(image.reshape(224, 224), cmap='gray')
#         plt.title(f'Processed Image {i}')
#
#     plt.tight_layout()
#     plt.show()
#
#
#
# # Example usage with a list of image paths
# image_paths = [r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\1\9613237R.png",
#                r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\2\9755851L.png",
#                r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\3\9002430L.png",
#                r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW implementation testing\DATASETS\train\4\9057150L.png"]
# preprocess_and_display(image_paths)
