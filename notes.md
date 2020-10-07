# Task 1
## Image Histograms
### Hypothesis
When an image is scaled or rotated, it will remain invariant to histograms.
This is because scaling or rotating an image should no change the proportion
between colors of the original image and the modified image.

For example, applying a 2 times scale on an image on both the height and width
should quadruple pixel count per pixel. While this would increase the number of
overall pixels for that color space, the proportions between the original image
and the scaled image remains unchanged.

Using a rotated image as an example, a 90 degrees rotation would contain the
same pixel arrangement as the original image, thus the proportions should stay
exactly the same.


### Results
The proportions of the modified image color spaces varies slightly, but not
enough to consider them variant, thus image rotation and scaling are invariant.


### Discussion and Issues
The hypothesis states that scale and rotation is invariant. The results shows
that the hypothesis is mostly correct. Scaling an image where the pixel values
are duplicated exactly would lead to the same proportions between the two
histograms, but some scaling algorithms, such as bilinear scaling, interpolates
the pixels such that it would smoothen the image after scaling. As a result,
this would lead to a difference in proportion between the two histograms.

Other issues include the rotation of an image. When rotating an image in steps
of 90 degrees, the histograms between the images maintains the same
proportions. However on other angles, black pixels are used to fill in the
space between the border and the image edge, thus introducing a large influx
of black pixels which skews the histograms towards the left.


## Harris Corner Detection
### Hypothesis
When an image is scaled or rotated, using the Harris corner detection will
lead to different results to the original image. This is because of how the
Harris corner algorithm checks the change of pixel intensity in changes
between the x-axis and y-axis.


### Results
Scaled images are variant, whereas rotated images are invariant.


### Discussion and Issues
The hypothesis states that the scale and rotation is variant. The results shows
that the hypothesis is incorrect. Harris corner identifies corners by detecting
changes in intensity of edges when shifting a viewport by the x-axis and
y-axis. As a result, a rotated image still contains the same edges from the
original image. Thus, Harris corner detection is still able to detect the
change in intensity of an edge no matter what orientation that edge is. Hence,
a rotated image is invariant.

However, a scaled image is variant because a change in scaling may misled
Harris corner detection to erroneously detect a scaled edge to be a corner,
in the case that an image is scaled down. It may also be unable to detect the
same corner of an upscaled image because the corner might be detected as an
edge.

One issue with the rotated Dugong image would be that the black background
fill interferes with the Harris corner detection, requiring a crop in order
to validate if a rotated image is invariant.

Other issues include having Harris corner incorrectly detecting the edges
on an image's border. For example, the black corner on the diamond card.


## Scale Invariant Feature Transform Key Points
### Hypothesis
When an image is scaled or rotated, using SIFT will be able to detect the
same features as the original image.


### Results
Though some of the key points within the groups differ between the original
and modified image, SIFT is capable of identifying the same features on
both images. Therefore, scaling and rotation is invariant to SIFT.


### Discussion and Issues
The hypothesis states that the scale and rotation is invariant. The results
shows that the hypothesis is correct. SIFT is able to detect the same features
on all of the different variants of scaling and rotation as the original
image. However, with extreme transformations such as a large scaling factor
or a large angle of rotation, SIFT's detection of key points differ to the
original image's key points coordinates, but are detecting the same features.


# Task 2
## Local Binary Patterns
### Main Steps
The main steps of Local Binary Patterns are as follows:
1. Convert image to grayscale.
2. Divide the image into many cells of size 16x16.
3. For each pixel in each cell in the image, compare the luminance of the
   pixel to the surrounding 8 pixels.
4. On the surrounding pixels, set 0 if the pixel value is less than the center
   pixel, otherwise set to 1.
5. For each pixel in each cell in the image, starting from the top left pixel
   in a clockwise direction, set the pixel value to the 8-bit binary number the
   surrounding pixels form.
6. For each cell in the image, calculate the histogram of the cell with a bin
   size of 256.
7. For each cell in the image, normalise the histogram.
8. Concatenate the histograms of each cell into a single vector, which forms
   the feature descriptors for the image.


Source: [https://www.bytefish.de/blog/local_binary_patterns.html](https://www.bytefish.de/blog/local_binary_patterns.html)
Accessed: 28/9/2020

Source: [https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/](https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)
Accessed: 28/9/2020

Source: [https://en.wikipedia.org/wiki/Local_binary_patterns](https://en.wikipedia.org/wiki/Local_binary_patterns)
Accessed: 29/9/2020


### Advantages
Local Binary Patterns algorithm is computationally light to perform. It's a
simple check for the intensity for all neighbouring pixels. Thus, it is very
fast to calculate.

Local Binary Patterns algorithm is very simple and does not require any image
filters, or rescaling.

Local Binary Patterns algorithm is invariant to photometric transformations.


### Disadvantages
The Local Binary Patterns algorithm is variant against rotation. When an image
is rotated, the binary encoding will be shifted, and thus creating a different
binary number when placed into the histogram, thus it is variant to rotation.

The Local Binary Patterns algorithm constructs a large histogram with a bin
size of 256, which can increase the computation time required to utilise this
histogram.

The Local Binary Patterns algorithm compares the center pixel to only it's
neighbouring pixels, thus would not be able to capture other dominant features
nearby. This can be remedied by applying an extension to Local Binary Patterns
that defines a variable neighbourhood size.


Source: [http://biomisa.org/uploads/2016/10/Lect-15.pdf](http://biomisa.org/uploads/2016/10/Lect-15.pdf)
Accessed: 29/9/2020

Source: [https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/](https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)
Accessed: 29/9/2020

Source: [https://www.computer.org/csdl/proceedings-article/icnc/2008/3304d115/12OmNx5Yvs6](https://www.computer.org/csdl/proceedings-article/icnc/2008/3304d115/12OmNx5Yvs6)
Accessed: 29/9/2020


## Histogram of Oriented Gradients
### Main Steps
The main steps of Histogram of Oriented Gradients are as follows:
1. Process the image by cropping it to a 1:2 aspect ratio.
2. Calculate the gradients along the x-axis and y-axis using the Sobel
   operator.
3. From the gradients, calculate the gradient angle and magnitude.
4. Divide the image into cells of equal sizes.
5. For each pixel in each cell, select an appropriate bin using the gradient
   direction.
6. If the angle matches a bin, record the entire magnitude for that bin,
   otherwise separate the magnitude to record in between the two bins.
7. Normalize the resulting histogram.
8. Calculate the resulting feature vector by concatenating every histograms.


Source: [https://www.learnopencv.com/histogram-of-oriented-gradients/](https://www.learnopencv.com/histogram-of-oriented-gradients/)
Accessed: 29/9/2020

Source: [http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/](http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/)
Accessed: 29/9/2020

Source: [https://chrisjmccormick.wordpress.com/2013/05/07/gradient-vectors/](https://chrisjmccormick.wordpress.com/2013/05/07/gradient-vectors/)
Accessed: 29/9/2020


### Advantages
Histogram of Oriented Gradients algorithm provides a good descriptor for
detecting a person.

Histogram of Oriented Gradients algorithm is computationally light to compute
when compared with Scale Invariant Feature Transform.

Histogram of Oriented Gradients algorithm is invariant to geometric and
photometric transformations.


### Disadvantages
Histogram of Oriented Gradients will produce a larger histogram on larger
images, which increases the computation time required to utilise the histogram.


## Scale Invariant Feature Transform
### Main Steps
The main steps of Scale Invariant Feature Transform are as follows:
1.  Create the scale space by creating multiple different octaves of the image.
2.  Each octaves contains the image of the same size, but with increasing
    scale, which is the kernel size for the Gaussian filter applied to the
    image.
3.  Calculate the difference of Gaussian for each octave by comparing the pixel
    value between the scales.
4.  Find all of the discrete extrema within the image by comparing a pixel with
    all of the surrounding neighbors, including the different scales.
5.  For each key points in the image, convert the discrete coordinates of the
    key point to a continuous coordinate by using the Taylor expansion of the
    image around the extrema.
6.  Remove any key points that contains low contrast or is an edge by using a
    technique similar to Harris corner detection.
7.  For each key points, perform Histogram of Oriented Gradients on the
    surrounding area of the key points, with a bin size of 36 for full 360
    degrees with 10 degrees per bin.
8.  For each bin in the histogram, if it is higher than 80% of the highest
    peak, create a new key point in the same location and with the same
    magnitude, but with the same direction as the highest peak.
9.  For each key points, generate a feature vector by extracting a 16x16 window
    with the key point at the center, dividing the window into 16 4x4 cells and
    perform a Histogram of Oriented Gradients with a bin size of 8 with 45
    degrees in each bin.
10. For each key points, concatenate and normalise each of the cells'
    histograms to form a vector of 128 values.


Source: [https://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/](https://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/)
Accessed: 29/9/2020

Source: [http://weitz.de/sift/index.html](http://weitz.de/sift/index.html)
Accessed: 29/9/2020


### Advantages
Scale Invariant Feature Transform performs very well, yielding high accuracy
when comparatively to other feature descriptors.

Scale Invariant Feature Transform is able to detect feature key points and
descriptors of a rotated or scaled image, within reason.


### Disadvantages
Scale Invariant Feature Transform is computationally expensive. It requires
constructing a scale space, which requires multiple Gaussian and downscaling
operations.

Scale Invariant Feature Transform is also variant to scale and rotation at
extreme values.


## Similarities
All three algorithms calculates the descriptor vector though concatenating
the calculated histograms in each cell.

All three algorithms are local algorithms.


## Differences
Scale Invariant Feature Transform is a feature descriptor and feature detector
whereas Local Binary Patterns and Histogram of Oriented Gradients are feature
descriptors only.

Scale Invariant Feature Transform creates a scale space to detect key points
whereas Local Binary Patterns and Histogram of Oriented Gradients operate in a
cell by cell process.


## Discussion and Issues
### Histogram of Oriented Gradients
Histogram of Oriented Gradients is unable to describe the same feature of an
image that is scaled or rotated. The difference in distance between the two
histograms created by the original image and the modified image not close.
Most distances are around 0.5 and the max distance being 1.48.

An issue encountered would be that rotated images have a black filler on the
border of the image. It is not clear if this affects the histogram calculated
by HOG. A crop was applied in order to remove the black border, but now brings
up the issue of whether the crop would also affect the resulting histogram.


### Scale Invariant Feature Transform
Scale Invariant Feature Transform is able to extract the same feature of an
image that is scaled or rotated. Images that are rotated slightly contains the
same feature key point and 

An issue encountered involved the diamond image rotated 20 degrees. The best
key point matches a diamond on the opposite side. This may be an issue with how
there are too many similar objects, which causes SIFT to match against the
opposite symbol. Cropping the image does not fix the issue. This issue also
appears on the diamond image upscaled to 1.5 times.


# Task 4
## Methods
The following methods are tested:
1. Original image
2. Red channel only
3. Red channel only with a Gaussian filter with kernel size 7

### Diamond
The Diamond image works best with method 1 with k value of 2. With methods 2
and 3, every channel except red is muted. Since the card's main color is red,
the samples passed into the kmeans function would end up being detected by one
cluster. However, method 3 is able to successfully segment the card symbols,
possibly due to the Gaussian filter.


### Dugong
The Dugong image works best with method 2 with k value of 2. With method 1, the
ocean was part of the segment that contains the seaweed and the dugong. With
method 3, while able to get similar results to method 2, contains the ocean as
part of the dugong extracted, surrounding it slightly.
