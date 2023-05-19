from PIL import Image, ExifTags


def get_focal_length(image_path):

    """
    Get focal length from EXIF or calculate it using prior

    Parameters
    ----------
    image_path: str
        Image path

    Returns
    -------
    focal_length: float
        Focal length extracted from EXIF or calculated using prior
    """

    image = Image.open(image_path)
    image_longest_edge = max(image.size)

    focal_length = None
    exif = image.getexif()

    if exif is not None:

        focal_length_35mm = None

        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_length_35mm = float(value)

        if focal_length_35mm is not None:
            focal_length = focal_length_35mm / 35. * image_longest_edge

    if focal_length is None:
        prior_focal_length = 1.2
        focal_length = prior_focal_length * image_longest_edge

    return focal_length
