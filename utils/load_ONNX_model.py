"""
Load ONNX models in order to extract
weights from the feature extractor.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnx
from onnx import numpy_helper

images = []
input_image = load_image("Misc_283.png")

MODEL_STR = "DD-v2.onnx"
model = onnx.load(MODEL_STR)

def load_image(image_path):
    """
    Load the image used to add the extracted weights onto
    """
    coloured_image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
    return grey_image




def convolve2d(image, kernel):
    """
    This function which takes an image & a kernel and
    returns the convolution of them.
    """
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    # Add zero padding to input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y : y + 3, x : x + 3]).sum()
    return output



W_PATH_ALL = (
    "StatefulPartitionedCall/center_net_resnet_v1fpn_feature_extractor/model_1/model/"
)

# "conv2_block1_2_conv/BiasAdd_weights_fused_bn"
# "conv3_block1_2_conv/BiasAdd_weights_fused_bn"
# "conv4_block1_2_conv/BiasAdd_weights_fused_bn"
# "conv5_block1_2_conv/BiasAdd_weights_fused_bn"

for i in range(2, 6):
    [tensor] = [
        t
        for t in model.graph.initializer
        if t.name
        == W_PATH_ALL + "conv" + str(i) + "_block1_2_conv/BiasAdd_weights_fused_bn"
    ]
    wghts = numpy_helper.to_array(tensor)

    num_maps, chnls, W_kernel = wghts[:, 0, 0, 0], wghts[0, :, 0, 0], wghts[0, 0, :, :]
    my_img = convolve2d(input_image, kernel=wghts[25, 40:, :, :])
    my_img = np.maximum(my_img, 0)
    images.append(my_img)

#    The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW),
#    where C is the number of channels, and kH and kW are the height and width of the kernel,
#    and M is the number of feature maps. For more than 2 dimensions,
#    the kernel shape will be (M x C/group x k1 x k2 x ... x kn),
#    where (k1 x k2 x ... kn) is the dimension of the kernel.

# plt.imshow(np.maximum(my_img, 0), cmap='jet')

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(images[0], cmap="Greys")
axs[0, 0].set_axis_off()
axs[0, 1].imshow(images[1], cmap="Greys")
axs[0, 1].set_axis_off()

axs[1, 0].imshow(images[2], cmap="Greys")
axs[1, 0].set_axis_off()
axs[1, 1].imshow(images[3], cmap="Greys")
axs[1, 1].set_axis_off()


plt.tight_layout()

plt.savefig(f"my_figs/{MODEL_STR}{str(i)}_wght.pdf")
plt.show()
