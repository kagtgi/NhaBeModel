import os
import numpy as np
import torch
import torchvision.transforms as transforms
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageOps
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import random
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import torch.nn.functional as F
import matplotlib.pyplot as plt

img_height = 128
img_width = 128
def add_background(frame, isLink = True):
        colors = [
            '#FFFFFF',                       # index 0: white
            '#D0E8FF', '#B3C8FF', '#99A8FF',  # indices 1-3: pastel shades of blue
            '#DFFFD9', '#A7D8A0', '#73B77C',  # indices 4-6: pastel shades of green
            '#FFFFB2', '#FFFF99', '#FFFF80',  # indices 7-9: pastel shades of yellow
            '#E6CCFF', '#D1A6FF',             # indices 13-14: pastel shades of purple
            '#FFD9D9', '#FFB2B2', '#FF8080'   # indices 10-12: pastel shades of red
            ]
        cmap = ListedColormap(colors)
        if isLink:
            image = Image.open(frame)
            image = ImageOps.invert(image).convert('L')
        else:
            image = frame.cpu().squeeze().numpy()
            #image = 1- image
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image).convert('L')
            #image.save("/kaggle/working/test.jpg")
        # Invert the colors
        #image = ImageOps.invert(image).convert('L')
        image = image.resize((128, 128))
        image_array = np.array(image)
        image_normalized = image_array / 255 * 70

        colored_image_array = cmap(image_normalized / 70)
        # Convert the colored array to uint8 format
        colored_image_array_uint8 = (colored_image_array[:, :, :3] * 255).astype(np.uint8)

        background = Image.fromarray(colored_image_array_uint8).convert("RGBA")
        img = Image.open('/kaggle/input/xoa-phong-bg/processed_background.png').convert("RGBA")  # Ensure this is an image file with transparency
        img = img.resize((128,128), Image.ANTIALIAS)
        background = background.resize((128, 128), Image.ANTIALIAS)
        # Calculate the offset for centering the image
        bg_w, bg_h = background.size
        img_w, img_h = img.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)

        # Create a new image for the result with an alpha channel (transparency)
        combined = Image.new("RGBA", img.size)

        # Paste the background onto the new image
        combined.paste(background, (0, 0))

        # Create an empty image for the overlay with the same size as the background
        overlay = Image.new("RGBA", img.size)

        # Paste the overlay image onto the empty overlay image at the calculated offset
        overlay.paste(img)

        # Blend the background and overlay images
        combined = Image.alpha_composite(combined, overlay)
        return combined
 
def visualize(file_paths, model):
    demo = []
    for i in range(6):
        img = load_img(file_paths[i],color_mode="grayscale", target_size=(img_height, img_width))
        img_array = img_to_array(img)
        demo.append(img_array)
        

    dataset2 = FramePredictionDataset(demo, transform=data_transform)
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=True, num_workers=2)


    # Print the shape of the first batch
    first_batch = next(iter(dataloader2))
    print(f'Input batch shape: {first_batch[0].shape}, Target batch shape: {first_batch[1].shape}')

    model.eval()  # Set model to evaluation mode

    # Predict Y_demo using the model
    with torch.no_grad():
        Y_demo = model(first_batch[0].to(device)) #torch.tensor [1,1,128,128]


    colors = [
        '#FFFFFF',                       # index 0: white
        '#D0E8FF', '#B3C8FF', '#99A8FF',  # indices 1-3: pastel shades of blue
        '#DFFFD9', '#A7D8A0', '#73B77C',  # indices 4-6: pastel shades of green
        '#FFFFB2', '#FFFF99', '#FFFF80',  # indices 7-9: pastel shades of yellow
        '#E6CCFF', '#D1A6FF',             # indices 13-14: pastel shades of purple
        '#FFD9D9', '#FFB2B2', '#FF8080'   # indices 10-12: pastel shades of red 
        ]
    cmap = ListedColormap(colors)
    imgs = []
    for i in range(len(file_paths)-1):
        #imgs.append((add_background(file_paths[i]), f"Original Frame {i + 1}"))
        imgs.append((add_background(file_paths[i]), f"Original Frame {i + 1}"))
        #imgs.append((add_background(frame_predicted, False), "Predicted"))
    imgs.append((add_background(Y_demo, False), "Predicted"))
    imgs.append((add_background(file_paths[-1]), "Ground truth"))

    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize = (40,40))
    for i in range(n):
        axs[i].imshow(imgs[i][0])
        axs[i].axis("off")
        axs[i].set_title(imgs[i][1], pad=10, fontsize = 20)

    norm = mcolors.Normalize(vmin=0, vmax=70)

    # Add a vertical colorbar to the figure with adjusted position
    cbar_ax = fig.add_axes([0.9, 0.436, 0.005, 0.12])  # Increase the first value to add space
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')

    # Customize the color bar
    cbar.set_label('Reflectivity (dBZ)', fontsize=25, labelpad=10)
    cbar.ax.tick_params(labelsize=20)  # Set the size of tick labels
    #cbar.set_ticks([0, 35, 70])  # Set the location of ticks
    #cbar.set_ticklabels(['Low', 'Medium', 'High'])  # Set the labels for ticks

    # Adjust layout to make space for the color bar
    plt.subplots_adjust(right=0.85)  # Adjust this value to fit the color bar
    path = '/kaggle/working/output' + str(random.randint(1, 10))+ '.png'
    print(path)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()

class FramePredictionDataset(Dataset):
    def __init__(self, images, transform=None):
        self.frames = images
        self.transform = transform
    def __len__(self):
        return len(self.frames) - 5

    def __getitem__(self, idx):
        frame_sequence = []
        for i in range(5):
            frame = self.frames[idx + i]
            frame = torch.tensor(frame, dtype=torch.float32).squeeze(-1)  # Remove singleton channel dimension
            if self.transform:
                frame = self.transform(frame)
            frame_sequence.append(frame)

        input_frames = torch.stack(frame_sequence, dim=0)  # Stack frames along a new channel dimension to get (5, H, W)

        target_frame = self.frames[idx + 5]
        target_frame = torch.tensor(target_frame, dtype=torch.float32).squeeze(-1).unsqueeze(0)  # Remove singleton channel dimension and add new channel dimension
        if self.transform:
            target_frame = self.transform(target_frame)

        return input_frames, target_frame
