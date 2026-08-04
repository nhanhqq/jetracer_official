import nbformat as nbf

nb = nbf.v4.new_notebook()

text_cells = [
    "### Camera",
    "### Task (Apex + Obstacle Bounding Box)",
    "### Data Collection",
    "### Model",
    "### Live Execution",
    "### Training",
    "### All together!"
]

code_cells = [
    # Camera
    """from jetcam.csi_camera import CSICamera
# from jetcam.usb_camera import USBCamera

camera = CSICamera(width=224, height=224)
# camera = USBCamera(width=224, height=224)

camera.running = True""",

    # Task
    """import torchvision.transforms as transforms
from apex_obstacle_dataset import ApexObstacleDataset

TASK = 'road_following_with_obstacle'
CATEGORIES = ['apex_and_obstacle']
DATASETS = ['A', 'B']

# Using ColorJitter for obstacle color augmentation
TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets = {}
for name in DATASETS:
    datasets[name] = ApexObstacleDataset(TASK + '_' + name, CATEGORIES, TRANSFORMS, random_hflip=True)""",

    # Data Collection
    """import cv2
import ipywidgets
import traitlets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
from jupyter_clickable_image_widget import ClickableImageWidget

dataset = datasets[DATASETS[0]]
camera.unobserve_all()

camera_widget = ClickableImageWidget(width=camera.width, height=camera.height)
snapshot_widget = ipywidgets.Image(width=camera.width, height=camera.height)
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)

dataset_widget = ipywidgets.Dropdown(options=DATASETS, description='dataset')
category_widget = ipywidgets.Dropdown(options=dataset.categories, description='category')
count_widget = ipywidgets.IntText(description='count')
count_widget.value = dataset.get_count(category_widget.value)

state_msg = ipywidgets.HTML(value="<b>State:</b> Click 1/5 (Apex - Tim đường)")

# Track clicks per image: 5 clicks total
# 0: apex, 1-4: obstacle corners
current_clicks = []

def set_dataset(change):
    global dataset
    dataset = datasets[change['new']]
    count_widget.value = dataset.get_count(category_widget.value)
dataset_widget.observe(set_dataset, names='value')

def update_counts(change):
    count_widget.value = dataset.get_count(change['new'])
category_widget.observe(update_counts, names='value')

def save_snapshot(_, content, msg):
    global current_clicks
    if content['event'] == 'click':
        data = content['eventData']
        x, y = data['offsetX'], data['offsetY']
        current_clicks.append((x, y))
        
        snapshot = camera.value.copy()
        
        # Draw all clicks so far
        if len(current_clicks) >= 1:
            # Draw apex (red)
            cv2.circle(snapshot, current_clicks[0], 8, (0, 0, 255), 3)
            cv2.putText(snapshot, 'Apex', (current_clicks[0][0]+10, current_clicks[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
        if len(current_clicks) > 1:
            # Draw obstacle corners (green)
            for pt in current_clicks[1:]:
                cv2.circle(snapshot, pt, 5, (0, 255, 0), 2)
                
        # Draw lines if 5 clicks are done
        if len(current_clicks) == 5:
            pts = current_clicks[1:]
            cv2.line(snapshot, pts[0], pts[1], (255, 0, 0), 2)
            cv2.line(snapshot, pts[1], pts[2], (255, 0, 0), 2)
            cv2.line(snapshot, pts[2], pts[3], (255, 0, 0), 2)
            cv2.line(snapshot, pts[3], pts[0], (255, 0, 0), 2)
            
            # Save
            dataset.save_entry(category_widget.value, camera.value, current_clicks)
            count_widget.value = dataset.get_count(category_widget.value)
            current_clicks = [] # Reset for next image
            state_msg.value = "<b>State:</b> Saved! Click 1/5 (Apex - Tim đường)"
        else:
            state_msg.value = f"<b>State:</b> Click {len(current_clicks)+1}/5 " + ("(Góc vật cản)" if len(current_clicks) > 0 else "(Apex - Tim đường)")
            
        snapshot_widget.value = bgr8_to_jpeg(snapshot)

camera_widget.on_msg(save_snapshot)

data_collection_widget = ipywidgets.VBox([
    state_msg,
    ipywidgets.HBox([camera_widget, snapshot_widget]),
    dataset_widget,
    category_widget,
    count_widget
])
display(data_collection_widget)""",

    # Model
    """import torch
import torchvision

device = torch.device('cuda')
output_dim = 10  # (x, y) for apex + 4 * (x,y) for obstacle bounding box

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, output_dim)
model = model.to(device)

model_save_button = ipywidgets.Button(description='save model')
model_load_button = ipywidgets.Button(description='load model')
model_path_widget = ipywidgets.Text(description='model path', value='road_obstacle_model.pth')

def load_model(c):
    model.load_state_dict(torch.load(model_path_widget.value))
model_load_button.on_click(load_model)
    
def save_model(c):
    torch.save(model.state_dict(), model_path_widget.value)
model_save_button.on_click(save_model)

model_widget = ipywidgets.VBox([
    model_path_widget,
    ipywidgets.HBox([model_load_button, model_save_button])
])
display(model_widget)""",

    # Live Execution
    """import threading
import time
import torch.nn.functional as F

state_widget = ipywidgets.ToggleButtons(options=['stop', 'live'], description='state', value='stop')
prediction_widget = ipywidgets.Image(format='jpeg', width=camera.width, height=camera.height)

# Ensure preprocess is imported or defined
def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = TRANSFORMS(image).to(device)
    return image[None, ...]

import PIL.Image

def live(state_widget, model, camera, prediction_widget):
    while state_widget.value == 'live':
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed).detach().cpu().numpy().flatten()
        
        # output has 10 values in [-1, 1]
        def to_pixel(val, max_val):
            return int(max_val * (val / 2.0 + 0.5))
            
        ax = to_pixel(output[0], camera.width)
        ay = to_pixel(output[1], camera.height)
        
        pts = []
        for i in range(4):
            x = to_pixel(output[2 + i*2], camera.width)
            y = to_pixel(output[2 + i*2 + 1], camera.height)
            pts.append((x, y))
            
        prediction = image.copy()
        # draw apex
        cv2.circle(prediction, (ax, ay), 8, (0, 0, 255), 3)
        # draw bbox
        for i in range(4):
            cv2.line(prediction, pts[i], pts[(i+1)%4], (255, 0, 0), 2)
            
        prediction_widget.value = bgr8_to_jpeg(prediction)
            
def start_live(change):
    if change['new'] == 'live':
        execute_thread = threading.Thread(target=live, args=(state_widget, model, camera, prediction_widget))
        execute_thread.start()

state_widget.observe(start_live, names='value')

live_execution_widget = ipywidgets.VBox([
    prediction_widget,
    state_widget
])
display(live_execution_widget)""",

    # Training
    """BATCH_SIZE = 8

optimizer = torch.optim.Adam(model.parameters())

epochs_widget = ipywidgets.IntText(description='epochs', value=1)
eval_button = ipywidgets.Button(description='evaluate')
train_button = ipywidgets.Button(description='train')
loss_widget = ipywidgets.FloatText(description='loss')
progress_widget = ipywidgets.FloatProgress(min=0.0, max=1.0, description='progress')

def train_eval(is_training):
    global BATCH_SIZE, model, dataset, optimizer
    
    try:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
            for images, category_idx, xy in iter(train_loader):
                images = images.to(device)
                xy = xy.to(device) # xy is shape (B, 10)

                if is_training:
                    optimizer.zero_grad()

                outputs = model(images)
                
                # loss is MSE over all 10 outputs
                loss = torch.mean((outputs - xy)**2)

                if is_training:
                    loss.backward()
                    optimizer.step()

                i += len(images)
                sum_loss += float(loss) * len(images)
                progress_widget.value = i / len(dataset)
                loss_widget.value = sum_loss / i
                
            if is_training:
                epochs_widget.value = epochs_widget.value - 1
            else:
                break
    except Exception as e:
        print("Error:", e)
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
    ipywidgets.HBox([train_button, eval_button])
])
display(train_eval_widget)""",

    # All together
    """all_widget = ipywidgets.VBox([
    ipywidgets.HBox([data_collection_widget, live_execution_widget]), 
    train_eval_widget,
    model_widget
])
display(all_widget)"""
]

cells = []
for i in range(len(text_cells)):
    cells.append(nbf.v4.new_markdown_cell(text_cells[i]))
    cells.append(nbf.v4.new_code_cell(code_cells[i]))

nb['cells'] = cells

with open('interactive_regression_v2.ipynb', 'w') as f:
    nbf.write(nb, f)
