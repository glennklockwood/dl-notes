#!/usr/bin/env python

import os
import glob
import time
import uuid
import subprocess

import cv2
import torch
import torch.utils.data
import torch.nn.functional
import torchvision
import torchvision.transforms
import PIL.Image

import jetcam.usb_camera

CATEGORIES = ['1', '2', '3', '4', '5']

PREPROCESS_MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
PREPROCESS_STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()

class ImageClassificationDataset(torch.utils.data.Dataset):
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
        annotation = self.annotations[idx]
        image = cv2.imread(annotation['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, annotation['category_index']

    def get_count(self, category):
        i = 0
        for annotation in self.annotations:
            if annotation['category'] == category:
                i += 1
        return i

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = torchvision.transforms.functional.to_tensor(image).to(device)
    image.sub_(PREPROCESS_MEAN[:, None, None]).div_(PREPROCESS_STD[:, None, None])
    return image[None, ...]

def infer_image(model, camera):
    image = camera.value
    preprocessed = preprocess(image)
    output = model(preprocessed)
    output = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy().flatten()
    return output

def train(model, dataset, optimizer, epochs=10, batch_size=8):

    device = torch.device('cuda')
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)

    if epochs:
        model = model.train()
    else:
        model = model.eval()

    eval_only = False
    epoch = epochs
    if epoch == 0:
        eval_only = True

    while epoch > 0 or eval_only:
        i = 0
        sum_loss = 0.0
        error_count = 0.0
        if not eval_only:
            print("beginning epoch %d" % (epochs - epoch + 1))

        for images, labels in iter(train_loader):
            # send data to device
            images = images.to(device)
            labels = labels.to(device)

            if not eval_only:
                # zero gradients of parameters
                optimizer.zero_grad()

            # execute model to get outputs
            outputs = model(images)

            # compute loss
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            if not eval_only:
                # run backpropogation to accumulate gradients
                loss.backward()

                # step optimizer to adjust parameters
                optimizer.step()

            # increment progress
            error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
            count = len(labels.flatten())
            i += count
            sum_loss += float(loss)

        print("  complete: %.1f" % (i / len(dataset)))
        print("  loss:     %f" % (sum_loss / i))
        print("  accuracy: %f" % (1.0 - error_count / i))

        epoch = epoch - 1
        if eval_only:
            break

    model = model.eval()
    return model

def main():
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

    # trainining / transfer learning
    optimizer = torch.optim.Adam(model.parameters())
    model = train(model, dataset, optimizer, epochs=10, batch_size=8)

    # inference
    camera = jetcam.usb_camera.USBCamera(width=224, height=224, capture_device=0)
    camera.running = True
    camera.unobserve_all()

    while True:
        output = infer_image(model, camera)
        category_index = output.argmax()
        print("Prediction: %s" % dataset.categories[category_index])
        for i, score in enumerate(list(output)):
            print("%s: %f" % (dataset.categories[i], score))

        time.sleep(1)

if __name__ == '__main__':
    main()
