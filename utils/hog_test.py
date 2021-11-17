
"""
Histogram of Oriented Gradients

Based on:
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
"""

import matplotlib.pyplot as plt
from skimage import exposure, io
from skimage.feature import hog

image1 = io.imread("image.png")
fd, hog_image = hog(image1,
                    orientations=8,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1),
                    visualize=True,
                    multichannel=True)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))


plt.rcParams['figure.figsize'] = [10, 10]
f, axarr = plt.subplots(1, 2, figsize=(6, 6), sharex=True)
f.subplots_adjust(hspace=0.1, wspace=0.01)

axarr[0].axis('off')
axarr[0].imshow(image1, cmap='gray')
axarr[0].set_title('Input image')

axarr[1].axis('off')
axarr[1].imshow(hog_image_rescaled, cmap='gray')
axarr[1].set_title('HOG')


plt.imsave('hog_target.jpg', hog_image_rescaled, cmap='gray')
