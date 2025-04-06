import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_video
from torchvision.transforms.functional import resize,normalize
import ultralytics
import depth_pro

ultralytics.checks()
model = ultralytics.YOLO("yolo11x-pose.pt")

# Load video to tensor
v_frames = read_video("Homebrew-video/Low-quality/IMG_5347.MP4",output_format="TCHW")[0].to("cuda").half()/255
v_frames = normalize(v_frames,[0.5,0.5,0.5],[0.5,0.5,0.5],inplace=True)

# Load depth model and preprocessing transform
depth_model, transform = depth_pro.create_model_and_transforms(device="cuda",precision=torch.half)
depth_model.eval()

for i,v_frame in enumerate(v_frames):
    depth_in = time.time()

    # Run inference.
    prediction = depth_model.infer(v_frame)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.
    
    depth_out = time.time()
    
    print(depth_out-depth_in)

    if i == 10: break