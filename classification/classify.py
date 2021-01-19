#!/usr/bin/env python

import jetcam.usb_camera

import torch
import torch.utils.data
import torchvision.transforms
import torch
import torchvision
import threading
import time
import torch.nn.functional as F

import glob
import PIL.Image
import subprocess
import cv2
import os
import uuid
import subprocess
import ipywidgets
import traitlets
import jetcam.utils
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

class ImageClassificationDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory, categories, transform=None):
        self.categories = categories
        self.directory = directory
        self.transform = transform
        self._refresh()
    
    
    def __len__(self):
        return len(self.annotations)
    
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, ann['category_index']
    
    
    def _refresh(self):
        self.annotations = []
        for category in self.categories:
            category_index = self.categories.index(category)
            for image_path in glob.glob(os.path.join(self.directory, category, '*.jpg')):
                self.annotations += [{
                    'image_path': image_path,
                    'category_index': category_index,
                    'category': category
                }]
    
    def save_entry(self, image, category):
        """Saves an image in BGR8 format to dataset for category"""
        if category not in self.categories:
            raise KeyError('There is no category named %s in this dataset.' % category)
            
        filename = str(uuid.uuid1()) + '.jpg'
        category_directory = os.path.join(self.directory, category)
        
        if not os.path.exists(category_directory):
            subprocess.call(['mkdir', '-p', category_directory])
            
        image_path = os.path.join(category_directory, filename)
        cv2.imwrite(image_path, image)
        self._refresh()
        return image_path
    
    def get_count(self, category):
        i = 0
        for a in self.annotations:
            if a['category'] == category:
                i += 1
        return i

TASK = 'thumbs'
# TASK = 'emotions'
# TASK = 'fingers'
# TASK = 'diy'

CATEGORIES = ['thumbs_up', 'thumbs_down']
# CATEGORIES = ['none', 'happy', 'sad', 'angry']
# CATEGORIES = ['1', '2', '3', '4', '5']
# CATEGORIES = [ 'diy_1', 'diy_2', 'diy_3']

DATASETS = ['A', 'B']
# DATASETS = ['A', 'B', 'C']

TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set up the data directory location if not there already
DATA_DIR = '/nvdli-nano/data/classification/'

BATCH_SIZE = 8

print("{} task with {} categories defined".format(TASK, CATEGORIES))

dataset = ImageClassificationDataset('../data/classification/' + TASK + '_' + name, CATEGORIES, TRANSFORMS)

# ### Data Collection
# Execute the cell below to create the data collection tool widget. This cell should only take a few seconds to execute.

# unobserve all callbacks from camera in case we are running this cell for second time
camera.unobserve_all()

# create image preview
camera_widget = ipywidgets.Image()
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=jetcam.utils.bgr8_to_jpeg)

# create widgets
dataset_widget = ipywidgets.Dropdown(options=DATASETS, description='dataset')
category_widget = ipywidgets.Dropdown(options=dataset.categories, description='category')
count_widget = ipywidgets.IntText(description='count')
save_widget = ipywidgets.Button(description='add')

# manually update counts at initialization
count_widget.value = dataset.get_count(category_widget.value)

# sets the active dataset
def set_dataset(change):
    global dataset
    count_widget.value = dataset.get_count(category_widget.value)

dataset_widget.observe(set_dataset, names='value')

# update counts when we select a new category
def update_counts(change):
    count_widget.value = dataset.get_count(change['new'])

category_widget.observe(update_counts, names='value')

# save image for category and update counts
def save(c):
    dataset.save_entry(camera.value, category_widget.value)
    count_widget.value = dataset.get_count(category_widget.value)

save_widget.on_click(save)

data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget]), dataset_widget, category_widget, count_widget, save_widget
])

# display(data_collection_widget)
print("data_collection_widget created")


# ### Model
# Execute the following cell to define the neural network and adjust the fully connected layer (`fc`) to match the outputs required for the project.  This cell may take several seconds to execute.

device = torch.device('cuda')

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))

model = model.to(device)

preprocess_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
preprocess_std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    global preprocess_mean, preprocess_std
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(preprocess_mean[:, None, None]).div_(preprocess_std[:, None, None])
    return image[None, ...]

state_widget = ipywidgets.ToggleButtons(options=['stop', 'live'], description='state', value='stop')
prediction_widget = ipywidgets.Text(description='prediction')
score_widgets = []
for category in dataset.categories:
    score_widget = ipywidgets.FloatSlider(min=0.0, max=1.0, description=category, orientation='vertical')
    score_widgets.append(score_widget)

def live(state_widget, model, camera, prediction_widget, score_widget):
    global dataset
    while state_widget.value == 'live':
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        prediction_widget.value = dataset.categories[category_index]
        for i, score in enumerate(list(output)):
            score_widgets[i].value = score
            
def start_live(change):
    if change['new'] == 'live':
        execute_thread = threading.Thread(target=live, args=(state_widget, model, camera, prediction_widget, score_widget))
        execute_thread.start()

state_widget.observe(start_live, names='value')

live_execution_widget = ipywidgets.VBox([
    ipywidgets.HBox(score_widgets),
    prediction_widget,
    state_widget
])

# display(live_execution_widget)
print("live_execution_widget created")


# ### Training and Evaluation
# Execute the following cell to define the trainer, and the widget to control it. This cell may take several seconds to execute.


optimizer = torch.optim.Adam(model.parameters())

epochs_widget = ipywidgets.IntText(description='epochs', value=1)
eval_button = ipywidgets.Button(description='evaluate')
train_button = ipywidgets.Button(description='train')
loss_widget = ipywidgets.FloatText(description='loss')
accuracy_widget = ipywidgets.FloatText(description='accuracy')
progress_widget = ipywidgets.FloatProgress(min=0.0, max=1.0, description='progress')

def train_eval(is_training):
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer, eval_button, train_button, accuracy_widget, loss_widget, progress_widget, state_widget
    
    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        state_widget.value = 'stop'
        train_button.disabled = True
        eval_button.disabled = True
        time.sleep(1)

        if is_training:
            model = model.train()
        else:
            model = model.eval()
        while epochs_widget.value > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, labels in iter(train_loader):
                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute loss
                loss = F.cross_entropy(outputs, labels)

                if is_training:
                    # run backpropogation to accumulate gradients
                    loss.backward()

                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
                count = len(labels.flatten())
                i += count
                sum_loss += float(loss)
                progress_widget.value = i / len(dataset)
                loss_widget.value = sum_loss / i
                accuracy_widget.value = 1.0 - error_count / i
                
            if is_training:
                epochs_widget.value = epochs_widget.value - 1
            else:
                break
    except e:
        pass
    model = model.eval()

    train_button.disabled = False
    eval_button.disabled = False
    state_widget.value = 'live'
    
train_button.on_click(lambda c: train_eval(is_training=True))
eval_button.on_click(lambda c: train_eval(is_training=False))
    
train_eval_widget = ipywidgets.VBox([
    epochs_widget,
    progress_widget,
    loss_widget,
    accuracy_widget,
    ipywidgets.HBox([train_button, eval_button])
])

# display(train_eval_widget)
print("trainer configured and train_eval_widget created")


# ### Display the Interactive Tool!

# The interactive tool includes widgets for data collection, training, and testing.

# Execute the cell below to create and display the full interactive widget.  Follow the instructions in the online DLI course pages to build your project.

# Combine all the widgets into one display
all_widget = ipywidgets.VBox([
    ipywidgets.HBox([data_collection_widget, live_execution_widget]), 
    train_eval_widget,
    model_widget
])

IPython.display(all_widget)
