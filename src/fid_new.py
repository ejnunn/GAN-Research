import torch
import torchvision
import torchextractor as tx
import multiprocessing
import numpy as np
import cv2
from scipy import linalg
import os
import glob
from PIL import Image
import pprint
 

def get_activations(images, batch_size):
    """
    Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
    """
    os.system('pip install --upgrade torchvision')
    os.system('pip install --upgrade torch')
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                              ", but got {}".format(images.shape)

    num_images = images.shape[0]
    print('loading model from torch.__version__ ', torch.__version__)
    original_model = torchvision.models.inception_v3(pretrained=True)
    assert ['maxpool1', 'maxpool2', 'avgpool'] in tx.list_module_names(original_model), "Inception_v3 model does not contain necessary layers for FID calculations"
    inception_network = tx.Extractor(original_model, ['maxpool1', 'maxpool2', 'avgpool'])
    inception_network = to_cuda(inception_network)
    inception_network.eval()
    n_batches = int(np.ceil(num_images  / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = to_cuda(ims)
        model_output, features = inception_network(ims)
        act1, act2, act3 = features.values()

        act1 = act1.detach().cpu().numpy().flatten()
        act1 = np.expand_dims(act1, axis=0)
        act2 = act2.detach().cpu().numpy().flatten()
        act2 = np.expand_dims(act2, axis=0)
        act3 = act3.detach().cpu().numpy().squeeze()
        act3 = np.expand_dims(act3, axis=0)
        
        assert act1.shape == (ims.shape[0], 341056), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 341056), act1.shape)
        assert act2.shape == (ims.shape[0], 235200), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 235200), act2.shape)
        assert act3.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), act3.shape)
        
    return act1, act2, act3


def calculate_activation_statistics(images, batch_size):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer 
                of the inception model.
    """
    act1, act2, act3 = get_activations(images, batch_size)
    
    mu1 = np.mean(act1, axis=0)
    mu2 = np.mean(act2, axis=0)
    mu3 = np.mean(act3, axis=0)

    sigma1 = np.cov(act1, rowvar=False)
    sigma2 = np.cov(act2, rowvar=False)
    sigma3 = np.cov(act3, rowvar=False)
    return mu1, mu2, mu3, sigma1, sigma2, sigma3



# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(images1, images2, use_multiprocessing, batch_size):
    """ Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    images1 = preprocess_images(images1, use_multiprocessing)
    images2 = preprocess_images(images2, use_multiprocessing)
    mu11, mu12, mu13, sigma11, sigma12, sigma13 = calculate_activation_statistics(images1, batch_size)
    mu21, mu22, mu23, sigma21, sigma22, sigma23 = calculate_activation_statistics(images2, batch_size)
    fid1 = calculate_frechet_distance(mu11, sigma11, mu21, sigma21) / 341056 * 2048 # normalize with 2048-feature fid score
    fid2 = calculate_frechet_distance(mu12, sigma12, mu22, sigma22) / 235200 * 2048 # normalize with 2048-feature fid score
    fid3 = calculate_frechet_distance(mu13, sigma13, mu23, sigma23)

    # maxpool1 torch.Size([341056])
    # maxpool2 torch.Size([235200])
    # avgpool torch.Size([2048])

    return fid1, fid2, fid3



def preprocess_image(im):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
    Return:
        im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
    """
    assert im.shape[2] == 3
    assert len(im.shape) == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    if im.dtype == np.float64:
        im = im.astype(np.float32) / 255
    im = cv2.resize(im, (299, 299))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im)
    assert im.max() <= 1.0
    assert im.min() >= 0.0
    assert im.dtype == torch.float32
    assert im.shape == (3, 299, 299)

    return im


def preprocess_images(images, use_multiprocessing):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
    """
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for im in images:
                job = pool.apply_async(preprocess_image, (im,))
                jobs.append(job)
            final_images = torch.zeros(images.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                im = job.get()
                final_images[idx] = im#job.get()
    else:
        final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float32
    return final_images


def to_cuda(elements):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
        --
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.cuda()
    return elements


def load_images(path, verbose=False):
    """ Loads all .png or .jpg images from a given path
    Warnings: Expects all images to be of same dtype and shape.
    Args:
        path: relative path to directory
    Returns:
        final_images: np.array of image dtype and shape.
    """
    image_paths = []
    image_extensions = ["png", "jpg"]
    for ext in image_extensions:
        if verbose:
            print("Looking for images in", os.path.join(path, "*.{}".format(ext)))
        for impath in glob.glob(os.path.join(path, "*.{}".format(ext))):
            image_paths.append(impath)
    first_image = cv2.imread(image_paths[0])
    W, H = first_image.shape[:2]
    image_paths.sort()
    image_paths = image_paths
    final_images = np.zeros((len(image_paths), W, H, 3), dtype=first_image.dtype)
    for idx, impath in enumerate(image_paths):
        im = cv2.imread(impath)
        im = im[:, :, ::-1] # Convert from BGR to RGB
        assert im.dtype == final_images.dtype
        final_images[idx] = im
    return final_images


def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--p1", "--path1", dest="path1", 
                      help="Path to directory containing the real images")
    parser.add_option("--p2", "--path2", dest="path2", 
                      help="Path to directory containing the generated images")
    parser.add_option("--multiprocessing", dest="use_multiprocessing",
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Set batch size to use for InceptionV3 network",
                      type=int)
    
    options, _ = parser.parse_args()
    assert options.path1 is not None, "--path1 is an required option"
    assert options.path2 is not None, "--path2 is an required option"
    assert options.batch_size is not None, "--batch_size is an required option"
    images1 = load_images(options.path1)
    images2 = load_images(options.path2)
    fid1, fid2, fid3 = calculate_fid(images1, images2, options.use_multiprocessing, options.batch_size)
    print('FID1:', fid1)
    print('FID2:', fid2)
    print('FID3:', fid3)




if __name__ == '__main__':
    main()





