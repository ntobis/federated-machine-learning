import math
import os
import cv2
from PIL import Image
import numpy as np

from Scripts.Data_Loader_Functions import get_labels


def load_and_preprocess_image(path):
    """
    Utility function loading an image, converting it into greyscale, and performing histogram equilization

    :param path:                    string, path for an image
    :return:
    """

    img = cv2.imread(path, 0)  # Load image into greyscale
    img = cv2.equalizeHist(img)  # Histogram equilization
    return img


def bulk_process_images(inputpath, outputpath, extension):
    """
    Utility function processing all images in a given directory.

    :param inputpath:               string, input path
    :param outputpath:              string, output path
    :param extension:               string, extension of the images
    :return:
    """

    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath) + 1:])
        for file in filenames:
            if file.endswith(extension):
                src = os.path.join(dirpath, file)
                dest = os.path.join(structure, file)
                img = load_and_preprocess_image(src)
                cv2.imwrite(dest, img)


def bulk_augment_images(input_path, output_path, extension, augmentation, label_type, label_threshold=-1):
    """
    Utility function augmenting all images in an input path, copying them into an output path

    :param input_path:              string, input path
    :param output_path:             string, output path
    :param extension:               string, extension of the images
    :param augmentation:            string, type of augmentation, takes 'flip', 'original', 'rotate_crop'
    :param label_type:              int, specify if images of a certain label type should not be augmented
    :param label_threshold:         int, specify if images of a certain label type should not be augmented
    :return:
    """
    for dir_path, dir_names, filenames in os.walk(input_path):
        structure = os.path.join(output_path, dir_path[len(input_path) + 1:])
        for file in filenames:
            if file.endswith(extension):
                src = os.path.join(dir_path, file)
                label = get_labels([src], label_type)[0]
                if label > label_threshold:
                    img = cv2.imread(src, 0)
                    f_name, f_ext = os.path.splitext(file)
                    if augmentation == 'flip':
                        img = np.flip(img, axis=-1)
                        file = f_name + "_flipped" + f_ext
                    elif augmentation == 'original':
                        file = f_name + "_original" + f_ext
                    elif augmentation == 'rotate_crop':
                        rotation = np.random.choice((-10, 10))
                        img = rotate_and_crop_image(img, rotation)
                        file = f_name + "_rotated" + f_ext
                    else:
                        raise ValueError(
                            "Invalid value for 'augmentation'. Value can be 'flip', 'original', 'rotate_crop, "
                            "value was: {}".format(augmentation))
                    dest = os.path.join(structure, file)
                    cv2.imwrite(dest, img)


def bulk_rename_files(input_path, output_path, suffix, new_suffix):
    """
    Utility function renaming files in a given folder. Can also move files.

    :param input_path:              string, input path
    :param output_path:             string, output path
    :param suffix:                  string, suffix that should be looked for
    :param new_suffix:              string, new suffix that the old one should be changed to
    :return:
    """
    for dir_path, dir_names, filenames in os.walk(input_path):
        structure = os.path.join(output_path, dir_path[len(input_path) + 1:])
        for file in filenames:
            src = os.path.join(dir_path, file)
            f_name, ext = os.path.splitext(file)
            if not f_name.endswith(suffix):
                file = f_name + new_suffix + ext
                dest = os.path.join(structure, file)
                os.rename(src, dest)


def bulk_crop_images(input_path, output_path, dims, extension):
    """
    Utility function cropping images to a specified size

    :param input_path:              string, input path
    :param output_path:             string, output path
    :param dims:                    tuple, tuple of ints specifying the image dimensions
    :param extension:               string, extension of the images
    :return:
    """
    for dir_path, dir_names, filenames in os.walk(input_path):
        structure = os.path.join(output_path, dir_path[len(input_path) + 1:])
        for file in filenames:
            if file.endswith(extension):
                src = os.path.join(dir_path, file)
                width, height = Image.open(src).size
                if width > dims[0] or height > dims[1]:
                    img = cv2.imread(src, 0)
                    img = crop_around_center(img, dims[0], dims[1])
                    dest = os.path.join(structure, file)
                    cv2.imwrite(dest, img)


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matrices backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rotate_and_crop_image(img, degrees):
    """
    Utility function rotating and cropping an image.

    :param img:             numpy array
    :param degrees:         int
    :return:
        Cropped image
    """
    image_rotated = rotate_image(img, degrees)
    return crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            img.shape[0],
            img.shape[1],
            math.radians(degrees)
        )
    )
