# Neural-Style-Transfer
- This project is based on an optimization technique used to take two images - content and style *reference* image and generating an output art image by blending the content image and style image together. The output image looks like the content image(input image) but painted in the style of style image.
- This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network.

## tools
- Python
- Tensorflow
- Keras
- Flask
- Waitress Serve
