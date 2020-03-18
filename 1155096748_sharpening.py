#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 1
# Name : Wong Kai Long
# Student ID : 1155096748
# Email Addr : 1155096748@link.cuhk.edu.hk
#

import argparse
import numpy as np
import imageio

PI = 3.14


# Please DO NOT import other libraries!


def imread(path):
    """
    DO NOT MODIFY!
    :param path: image path to read, str format
    :return: image data in ndarray format, the scale for the image is from 0.0 to 1.0
    """
    assert isinstance(path, str), 'Please use str as your path!'
    assert (path[-3:] == 'png') or (path[-3:] == 'PNG'), 'This assignment only support PNG grayscale images!'
    im = imageio.imread(path)
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = im / 255.
    return im


def imwrite(im, path):
    """
    DO NOT MODIFY!
    :param im: image to save, ndarray format, the scale for the image is from 0.0 to 1.0
    :param path: path to save the image, str format
    """
    assert isinstance(im, np.ndarray), 'Please use ndarray data structure for your image to save!'
    assert isinstance(path, str), 'Please use str as your path!'
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = (im * 255.0).astype(np.uint8)
    imageio.imwrite(path, im)


def gaussian_kernel(size, sigma):
    """
    :param size: kernel size: size x size, int format
    :param sigma: standard deviation for gaussian kernel, float format
    :return: gaussian kernel in ndarray format
    """

    num=int(size/2)
    assert isinstance(size, int), 'Please use int for the kernel size!'
    assert isinstance(sigma, float), 'Please use float for sigma!'
    y,x = np.mgrid[-num:size-num, -num:size-num]

    value1 = np.exp(-(x**2+y**2)/(2*sigma**2))*(1/(2*PI*sigma**2))

    n=0
    for i in range(0,size+1):
        n = np.exp(-(i**2+i**2)/(2*sigma**2))*(1/(2*PI*sigma**2))+n

    kernel=value1/n

    # ##################### Implement this function here ##################### #
    #kernel = np.zeros(shape=[size, size], dtype=float)  # this line can be modified

    # ######################################################################## #
    assert isinstance(kernel, np.ndarray), 'please use ndarray as you kernel data format!'
    return kernel


def conv(im_in, kernel):
    """
    :param im_in: image to be convolved, ndarray format
    :param kernel: kernel use to convolve, ndarray format
    :return: result image, ndarray format
    """
    assert isinstance(im_in, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(kernel, np.ndarray), 'Please use ndarray data structure for your kernel!'

    im_in=np.asarray(im_in)
    size_of_im_1d=int(im_in.size**(1/2))
    kernel_size_1d=int(kernel.size**(1/2))
    size = size_of_im_1d-kernel_size_1d+1

    array = np.zeros(shape=[size, size], dtype=float)

    for i in range(0,size_of_im_1d-kernel_size_1d+1):
        for j in range(0,size_of_im_1d-kernel_size_1d+1):
            x=0
            for k in range(kernel_size_1d):
                for l in range(kernel_size_1d):
                    x=kernel[k,l]*im_in[i+k,j+l]+x
            array[i,j]=x
    return array




    #return im_in


    # ##################### Implement this function here ##################### #

    # ######################################################################## #
def sharpen(im_input, im_smoothed):
    """
    :param im_input: the original image, ndarray format
    :param im_smoothed: the smoothed image, ndarray format
    :return: sharoened image, ndarray format
    """
    assert isinstance(im_input, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(im_smoothed, np.ndarray), 'Please use ndarray data structure for your image!'

    # ##################### Implement this function here ##################### #

    k=int(im_smoothed.size**(1/2))

    n = int((im_input.size**(1/2)-im_smoothed.size**(1/2))/2)
    input_image_crop = np.zeros(shape=[k,k], dtype=float)
    y,x = im_input.shape
    startx = int(x//2-(k//2))
    starty = int(y//2-(k//2))

    input_image_crop = im_input[starty:starty+k, starty:starty+k]

    detail_map = input_image_crop-im_smoothed
    sharpened_image = input_image_crop + detail_map
    return sharpened_image

    # ######################################################################## #
#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 1
# Name : Wong Kai Long
# Student ID : 1155096748
# Email Addr : 1155096748@link.cuhk.edu.hk
#

import argparse
import numpy as np
import imageio

PI = 3.14


# Please DO NOT import other libraries!


def imread(path):
    """
    DO NOT MODIFY!
    :param path: image path to read, str format
    :return: image data in ndarray format, the scale for the image is from 0.0 to 1.0
    """
    assert isinstance(path, str), 'Please use str as your path!'
    assert (path[-3:] == 'png') or (path[-3:] == 'PNG'), 'This assignment only support PNG grayscale images!'
    im = imageio.imread(path)
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = im / 255.
    return im


def imwrite(im, path):
    """
    DO NOT MODIFY!
    :param im: image to save, ndarray format, the scale for the image is from 0.0 to 1.0
    :param path: path to save the image, str format
    """
    assert isinstance(im, np.ndarray), 'Please use ndarray data structure for your image to save!'
    assert isinstance(path, str), 'Please use str as your path!'
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = (im * 255.0).astype(np.uint8)

    imageio.imwrite(path, im)


def gaussian_kernel(size, sigma):
    """
    :param size: kernel size: size x size, int format
    :param sigma: standard deviation for gaussian kernel, float format
    :return: gaussian kernel in ndarray format
    """

    num=int(size/2)
    assert isinstance(size, int), 'Please use int for the kernel size!'
    assert isinstance(sigma, float), 'Please use float for sigma!'
    y,x = np.mgrid[-num:size-num, -num:size-num]
    kernel = np.zeros(shape=[size, size], dtype=float)
    kernel = np.exp(-(x**2+y**2))*(1/(2*PI*sigma**2))
    kernel = kernel / kernel.sum()


    # ##################### Implement this function here ##################### #
    #kernel = np.zeros(shape=[size, size], dtype=float)  # this line can be modified

    # ######################################################################## #
    assert isinstance(kernel, np.ndarray), 'please use ndarray as you kernel data format!'

    return kernel


def conv(im_in, kernel):
    """
    :param im_in: image to be convolved, ndarray format
    :param kernel: kernel use to convolve, ndarray format
    :return: result image, ndarray format
    """
    assert isinstance(im_in, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(kernel, np.ndarray), 'Please use ndarray data structure for your kernel!'

    im_in=np.asarray(im_in)
    size_of_im_1d=int(im_in.size**(1/2))
    kernel_size_1d=int(kernel.size**(1/2))
    size = size_of_im_1d-kernel_size_1d+1

    array = np.zeros(shape=[size, size], dtype=float)

    for i in range(0,size_of_im_1d-kernel_size_1d+1):
        for j in range(0,size_of_im_1d-kernel_size_1d+1):
            x=0
            for k in range(kernel_size_1d):
                for l in range(kernel_size_1d):
                    x=kernel[k,l]*im_in[i+k,j+l]+x
            array[i,j]=x
    #np.clip(array,0,1)

    return array

    #return im_in


    # ##################### Implement this function here ##################### #

    # ######################################################################## #
def sharpen(im_input, im_smoothed):
    """
    :param im_input: the original image, ndarray format
    :param im_smoothed: the smoothed image, ndarray format
    :return: sharoened image, ndarray format
    """
    assert isinstance(im_input, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(im_smoothed, np.ndarray), 'Please use ndarray data structure for your image!'

    # ##################### Implement this function here ##################### #

    k=int(im_smoothed.size**(1/2))

    n = int((im_input.size**(1/2)-im_smoothed.size**(1/2))/2)
    input_image_crop = np.zeros(shape=[k,k], dtype=float)
    y,x = im_input.shape
    startx = int(x//2-(k//2))
    starty = int(y//2-(k//2))

    input_image_crop = im_input[starty:starty+k, starty:starty+k]

    detail_map = input_image_crop-im_smoothed
    sharpened_image = input_image_crop + detail_map
    sharpened_image=np.clip(sharpened_image,0,1)
    return sharpened_image

def main():
    parser = argparse.ArgumentParser(description='Image Sharpening')
    parser.add_argument('--input', type=str, default='test_01.png', help='path of the input image')
    parser.add_argument('--kernel', type=int, default=5, help='the square kernel size')
    parser.add_argument('--sigma', type=float, default=1.5, help='the standard deviation in gaussian kernel')
    parser.add_argument('--output', type=str, default='output_01.png', help='the path of the output image')
    args = parser.parse_args()

    im = imread(args.input)

    kernel = gaussian_kernel(size=args.kernel, sigma=args.sigma)
    smoothed_im = conv(im_in=im, kernel=kernel)
    sharpened_im = sharpen(im_input=im, im_smoothed=smoothed_im)
    imwrite(im=sharpened_im, path=args.output)


if __name__ == '__main__':
    main()


def main():
    parser = argparse.ArgumentParser(description='Image Sharpening')
    parser.add_argument('--input', type=str, default='test_01.png', help='path of the input image')
    parser.add_argument('--kernel', type=int, default=5, help='the square kernel size')
    parser.add_argument('--sigma', type=float, default=1.5, help='the standard deviation in gaussian kernel')
    parser.add_argument('--output', type=str, default='output_01.png', help='the path of the output image')
    args = parser.parse_args()

    im = imread(args.input)

    kernel = gaussian_kernel(size=args.kernel, sigma=args.sigma)
    smoothed_im = conv(im_in=im, kernel=kernel)
    sharpened_im = sharpen(im_input=im, im_smoothed=smoothed_im)
    imwrite(im=sharpened_im, path=args.output)


if __name__ == '__main__':
    main()
