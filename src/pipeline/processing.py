import torch
import numpy as np
import pandas as pd
import pydicom
import os
from pydicom.filebase import DicomBytesIO
from nvidia.dali import fn, math, pipeline_def, types
from typing import Any, Tuple, Dict
from collections import defaultdict


def read_encoded_stream(filename:str) -> np.ndarray:
    """Read in a filename retrieving a NumPy array of bytes
    """
    dcmfile = pydicom.dcmread(filename)
    
    if dcmfile.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.90":
        # Read in JPEG2000
        offset = dcmfile.PixelData.find(b"\x00\x00\x00\x0C")
    else:
        # Read in JPG lossless
        offset = dcmfile.PixelData.find(b"\xff\xd8")

    buffer = np.array(bytearray(dcmfile.PixelData[offset:]), dtype=np.uint8)
    return buffer


def parse_window_element(element:Any) -> float:
    """Process the window element of a DICOM file as a single float. If multiple
    potential window elements are provided, we extract only the first one.
    """
    if isinstance(element, list):
        return float(element[0])
    
    if isinstance(element, str):
        return float(element)
    
    if isinstance(element, float):
        return element
    
    if isinstance(element, pydicom.dataelem.DataElement):
        try:
            return float(element[0])
        except TypeError as error:
            return float(element.value)


def read_parameters(filename:str) -> Tuple[bool, int, int]:
    """Retrieve DICOM parameters for parsing images. We determine if an image
    should be inverted and specify a lower and upper bound on image values. Any
    value less than the lower bound should be set to the lower boundm, and any
    value greater than the upper bound should be set to the upper bound.
    
    Args:
        filename (str): The file name of the DICOM file.
    
    Returns:
        invert (bool): Whether the image should be inverted
        lower (int): The computed lowest visible value of the DICOM image
        upper (int): The computed highest visible value of the DICOM image
    """
    dcmfile = pydicom.dcmread(filename, stop_before_pixels=True)
    
    try:
        invert = getattr(
            dcmfile,
            "PhotometricInterpretation",
            None
        ) == "MONOCHROME1"
    except:
        invert = False
    
    center = parse_window_element(dcmfile["WindowCenter"])
    width = parse_window_element(dcmfile["WindowWidth"])
    
    # Compute the lower and upper bounds of the DICOM image
    lower = center - width // 2
    upper = center + width // 2
    
    return (invert, lower, upper)


def read_parameters(filename:str) -> Tuple[bool, int, int]:
    """Retrieve DICOM parameters for parsing images. We determine if an image
    should be inverted and specify a lower and upper bound on image values. Any
    value less than the lower bound should be set to the lower boundm, and any
    value greater than the upper bound should be set to the upper bound.
    
    Args:
        filename (str): The file name of the DICOM file.
    
    Returns:
        invert (bool): Whether the image should be inverted
        lower (int): The computed lowest visible value of the DICOM image
        upper (int): The computed highest visible value of the DICOM image
    """
    dcmfile = pydicom.dcmread(filename, stop_before_pixels=True)
    
    # TODO: Determine underlying behavior of this try/except
    try:
        invert = getattr(
            dcmfile,
            "PhotometricInterpretation",
            None
        ) == "MONOCHROME1"
    except:
        invert = False
    
    center = parse_window_element(dcmfile["WindowCenter"])
    width = parse_window_element(dcmfile["WindowWidth"])
    
    # Compute the lower and upper bounds of the DICOM image
    lower = center - width // 2
    upper = center + width // 2
    
    return (invert, lower, upper)


@pipeline_def()
def pipeline():
    """Pipeline for preprocessing DICOM images. Parameters must be fed into the
    pipeline prior to running the pipeline.run() method using the
    pipeline.feed_input() method.
    
    Args:
        jpegs (list of strings): The byte array of the DICOM images to decode
        invert (list of bools or ints): Whether to invert each DICOM image
        lower (list of ints): The lower bound of each DICOM image
        upper (list of ints): The upper bound of each DICOM image
    
    Returns:
        images (DALI Tensor): The output image(s)
    """
    # These parameters must be fed into the pipeline prior to each pipeline.run()
    jpegs = fn.external_source(device="cpu", name="jpegs")
    invert = fn.external_source(device="cpu", name="invert")
    lower = fn.external_source(device="cpu", name="lower")
    upper = fn.external_source(device="cpu", name="upper")
    
    # Read in the JPEG images
    images = fn.experimental.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.ANY_DATA,
        dtype=types.UINT16
    )
    
    images = fn.cast(images, dtype=types.FLOAT)
    
    # Clip the pixel values between the lower and upper bound
    images = math.clamp(images, lower, upper)
    
    # Rescale the image between 0 and 1
    min_value = fn.reductions.min(images)
    max_value = fn.reductions.max(images)
    images = (images - min_value) / (max_value - min_value)
    
    # Invert the image if necessary
    images = images * (1 - invert) + invert * (1 - images)
    
    return images


def process_torch(image:torch.Tensor, device:torch.device) -> torch.Tensor:
    """Preprocess a torch tensor image
    
    Args:
        image (torch.Tensor): The input tensor
    
    returns:
        tensor (torch.Tensor): The processed tensor
    """
    image = image.reshape((image.shape[0], image.shape[1]))
    
    # Rescale the image to [0, 1]
    min_value = torch.min(image)
    max_value = torch.max(image)
    image = (image - min_value) / (max_value - min_value)
    
    # Remove whitespace
    rows_keep = torch.sum(image, axis=1) != 0
    cols_keep = torch.sum(image, axis=0) != 0
    image = image[rows_keep]
    image = image[:, cols_keep]
    
    image = image.unsqueeze(0)
    image = image.expand(3, -1, -1)
    image = image.to(device)
    return image


def process_metadata(filename:str, image_dir:str) -> Dict[str, pd.DataFrame]:
    """Load in the metadata as a Pandas DataFrame.
    
    Args:
        filename (str): The file location of the metadata
        image_dir (str): The directory containing images; e.g., 'train_images/'
    
    Returns:
        metadata (dict): The processed metadata. Each key represents a prediction
            ID (e.g., '10006_L') and each value provides a Pandas DataFrame
            containing necessary image information.
    """
    df_metadata = pd.read_csv(filename)
    
    patient_ids = df_metadata["patient_id"]
    lateralities = df_metadata["laterality"]
    image_ids = df_metadata["image_id"]
    cancers = df_metadata["cancer"]
    
    prediction_ids = [
        f"{patient_id}_{laterality}"
        for patient_id, laterality in zip(patient_ids, lateralities)
    ]
    
    fnames = [
        os.path.join(image_dir, str(patient_id), f"{image_id}.dcm")
        for patient_id, image_id in zip(patient_ids, image_ids)
    ]
    
    metadata = defaultdict(
        lambda: {
            "fname": [],
            "cancer": 0,
        }
    )
    
    for prediction_id, fname, cancer in zip(prediction_ids, fnames, cancers):
        metadata[prediction_id]["fname"].append(fname)
        metadata[prediction_id]["cancer"] = cancer
    
    return metadata