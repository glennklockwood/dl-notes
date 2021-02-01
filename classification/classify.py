#!/usr/bin/env python
"""Image classification with pytorch

Adapted from the NVIDIA Deep Learning Institute's "Getting Started with AI on
Jetson Nano" course notebook on classification.  Assumes you already have
captured a training dataset using the notebook provided by that course; this
reimplements the training and inference parts of that notebook in standalone
Python.

Implements the five-class example to infer how many fingers you are holding up.
"""

import os
import sys
import glob
import time

import cv2
import torch
import torch.utils.data
import torch.nn.functional
import torchvision
import torchvision.transforms
import PIL.Image

import jetcam.usb_camera

CATEGORIES = ['1', '2', '3', '4', '5']

PREPROCESS_MEAN = None
PREPROCESS_STD = None

USE_GPU = False

class ImageClassificationDataset(torch.utils.data.Dataset):
    """Dataset composed of images with classifications

    Loads images and classifications.  Assumes directory structure of

        directory/
        directory/categories[0]
        directory/categories[1]
        ...

    where categories[1] etc correspond to the elements passed as the
    ``categories`` argument.

    Args:
        directory (str): Path to base directory containing images
        categories (list of str): List of valid classifications

    Attributes:
        categories (list of str): List of valid classifications
        transform:
        annotations (list of dict): Metadata describing each member of the
            loaded dataset
    """
    def __init__(self, directory, categories, transform=None):
        self.categories = categories
        self.transform = transform
        self.annotations = []

        for category in categories:
            category_index = categories.index(category)
            for image_path in glob.glob(os.path.join(directory, category, '*.jpg')):
                self.annotations.append({
                    'image_path': image_path,
                    'category_index': category_index,
                    'category': category
                })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Returns an element of the dataset

        Returns:
            tuple: Elements (image, category) where image is a Tensor and
            category is an int corresponding to the index of self.categories
            to which image belongs.
        """
        annotation = self.annotations[idx]
        image = cv2.imread(annotation['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, annotation['category_index']

    def get_count(self, category):
        """Returns number of data with the given classification

        Args:
            category (str): Category which should be counted.  Must be an
                element of self.categories

        Returns:
            int: Number of elements that match given category
        """
        i = 0
        for annotation in self.annotations:
            if annotation['category'] == category:
                i += 1
        return i

def load_model(model, path):
    """Loads a trained pytorch model

    Args:
        model (Model): model to load trained state into
        path (str): Path to model trained state

    Returns:
        Model
    """
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, path):
    """Saves a trained pytorch model

    Args:
        model (Model): model whose trained state should be saved
        path (str): Path to file into which trained state should be saved
    """
    torch.save(model.state_dict(), path)

def preprocess(image):
    """Prepares the image for our model

    Performs the following:

    1. Transform the image in bgr8 format to a tensor
    2. Sends tensor to the GPU
    3. Normalizes the tensor
    4. Adds a new first dimension to the tensor full of Nones

    Args:
      image (numpy.array): Image from a camera represented as in bgr8
          format - an array with dimensions (224, 224, 3) on the host
          memory

    Returns:
        torch.Tensor: Transformed tensor on the GPU with dimensions
        [1, 3, 224, 224]
    """
    global PREPROCESS_MEAN, PREPROCESS_STD
    if USE_GPU:
        device = torch.device('cuda')

    # Convert pixel array into a tensor, then send it to GPU
    #
    #   The input is an array of [224, 224, 3]
    #   The output is a tensor of [3, 224, 224]
    #
    image = torchvision.transforms.functional.to_tensor(image)#.to(device)

    # Normalize the input image.
    #
    # Tensor.sub_ is the in-place version of torch.sub()
    # Tensor.div_ is the in-place version of torch.div()
    #
    # Thus we are:
    #
    # 1. subtracting the mean const from the image
    # 2. dividing this centeredby the standard deviation const
    #
    # The constants (mean and std) come from pytorch and what its models expect
    # input to look like.  See https://pytorch.org/docs/stable/torchvision/models.html
    #
    if PREPROCESS_MEAN is None:
        if USE_GPU:
            PREPROCESS_MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
            PREPROCESS_STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        else:
            PREPROCESS_MEAN = torch.Tensor([0.485, 0.456, 0.406])
            PREPROCESS_STD = torch.Tensor([0.229, 0.224, 0.225])

    image.sub_(PREPROCESS_MEAN[:, None, None]).div_(PREPROCESS_STD[:, None, None])
    return image[None, ...]

def infer_image(model, image):
    """Classifies a given image

    Args:
        model (Model): model to be used for inference
        image (numpy.array): bgr8 image to be classified

    Returns:
        numpy.array: Vector containing the softmax-normalized probabilities for
        each of the categories in the fully connected layer
    """
    preprocessed = preprocess(image)
    output = model(preprocessed) # output = torch.Tensor of size (1, 5)
    # softmax() returns a tensor of size (1, 5)
    # detach() returns the same tensor of size (1, 5)
    # cpu() returns the same tensor, but on the host instead of gpu
    # numpy() returns a numpy array of size (1, 5)
    # flatten() returns a numpy array of size (5,)
    output = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy().flatten()
    return output

def train(model, dataset, optimizer, epochs=10, batch_size=8):
    """Trains or evaluates a model given a dataset

    Args:
        model (Model): model to be used for inference
        dataset (ImageClassificationDataset):
        optimizer (torch.optim.Optimizer): optimizer to use
        epochs (int): number of epochs to use for training.  If zero, run the
            model in eval mode.
        batch_size (int): size of each batch

    Returns:
        Model: the mutated model object
    """

    if USE_GPU:
        device = torch.device('cuda')
        model = model.to(device)

    # train_loader is a "Map-style dataset" since it implements __getitem__()
    # and __len__().  These allow the DataLoader to randomly pick elements
    # between 0 and len(DataLoader) in batches of the given size.
    #
    # Our choice of __getitem__() in ImageClassificationDataset returns a tuple
    # with elements (image, category).
    #
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)

    # We normally keep the model in eval() mode so that new images fed into it
    # are used for inference, not training.  Here, we explicitly turn on
    # training mode since we are about to train.
    #
    # eval mode only matters if there is a dropout layer which would cause
    # the model to mutate on inference, but it's safe to be explicit either way.
    #
    if epochs:
        model = model.train()
    else:
        model = model.eval()

    eval_only = False
    epoch = epochs
    if epoch == 0:
        eval_only = True

    # Iterate through the epochs
    while epoch > 0 or eval_only:
        tstart = time.time()
        i = 0
        sum_loss = 0.0
        error_count = 0.0
        if not eval_only:
            print("beginning epoch %d" % (epochs - epoch + 1))

        # for each epoch, get a batch of samples
        for images, labels in iter(train_loader):
            # Send data to GPU - images and labels now refer to data on the GPU
            if USE_GPU:
                images = images.to(device) # of dimensions (epochs, 3, 224, 224)
                labels = labels.to(device) # of dimensions (epochs)

            if not eval_only:
                # Zero gradients of parameters, required because pytorch
                # otherwise accumulates gradients on backpropogation.
                optimizer.zero_grad()

            # Feed new image into model and get the output based on its training
            # so far.  outputs has dimensions (epochs, num_categories)
            outputs = model(images)

            # Compute the loss function based on the result our model gave and
            # the ground truth we know.  The cross entropy loss function
            # computes the softmax of the output (fully connected) layer, then
            # calculates the negative log likelihood of those layers.  It is
            # equivalent to
            #
            # torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(outputs, dim=1), labels)
            #
            # It returns a Tensor with grad_fn set to NllLossBackward so it can
            # be backpropagated using the .backward() method.  The return tensor
            # is a scalar whose sole value is the calculated loss.
            #
            # See https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
            #
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            if not eval_only:
                # run backpropogation to calculate and store gradients
                loss.backward()

                # run the optimizer to adjust parameters of the model
                optimizer.step()

                # Q: how are model, loss, and optimizer connected?  optimizer
                # was initialized with the model parameters, so are model
                # parameters pointers that optimizer.step() mutates?  how does
                # it know what the calculated loss is?

            # outputs.argmax(1) returns a Tensor containing the indices that
            # correspond to the maximum value along a dimension.  argmax(1)
            # tells us, for each epoch (dim=1), which category index had the
            # highest value.  For example,
            #
            # outputs[1] = tensor([-2.9911,  4.8263,  0.8418, -3.6851, -4.0162])
            # outputs.argmax(1)[1] = tensor(1)
            #
            # that is, for epoch 1, the category with the highest value
            # (predicted confidence) is index 1
            #
            # (outputs.argmax(1) - labels) compares the model with the highest
            # predicted classification to the ground-truth label and tells us
            # if the best-predicted class matched ground-truth (value is zero)
            # or not (value is nonzero).
            error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())

            # count is the number of images trained against during this epoch
            count = len(labels.flatten())
            # i is the total number of images trained against
            i += count
            # sum_loss is the sum of all losses
            sum_loss += float(loss)

        print("  complete: %.1f" % (i / len(dataset)))
        # we express loss as the average loss per image trained
        print("  loss:     %f" % (sum_loss / i))
        # accuracy is a function of the average error per image
        print("  accuracy: %f" % (1.0 - error_count / i))

        print("  duration: %.2f secs" % (time.time() - tstart))
        epoch = epoch - 1
        if eval_only:
            break

    # switch our model back to eval (inference) mode from train mode
    model = model.eval()
    return model

def main():
    """Train a model and begin inferencing
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_dir = 'thumbs_A'
    dataset = ImageClassificationDataset(dataset_dir, CATEGORIES, transforms)
    print("Loaded dataset from %s" % dataset_dir)
    for category in dataset.categories:
        print("%12s: %d images" % (category, dataset.get_count(category)))

    # model
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, len(dataset.categories))

    tstart = time.time()
    # trainining / transfer learning
    optimizer = torch.optim.Adam(model.parameters())
    model = train(model, dataset, optimizer, epochs=10, batch_size=8)
    print("Took %.2f seconds to train" % (time.time() - tstart))
    sys.exit(0)

    # inference
    camera = jetcam.usb_camera.USBCamera(width=224, height=224, capture_device=0)
    camera.running = True
    camera.unobserve_all()

    while True:
        output = infer_image(model, camera.value)
        category_index = output.argmax()
        print("Prediction: %s" % dataset.categories[category_index])
        for i, score in enumerate(list(output)):
            print("%s: %f" % (dataset.categories[i], score))

        time.sleep(1)

if __name__ == '__main__':
    main()
