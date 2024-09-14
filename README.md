
# Nha Be Weather Radar Model

This repository contains code for running a ConvLSTM model for radar weather prediction using historical data. The following steps will guide you through setting up and running a demo.

## Setup

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone <repository-url>
```

Replace `<repository-url>` with the actual URL of your Git repository.

### 2. Install Dependencies

Navigate to the project directory and install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Change Directory (For Kaggle Users)

If you're using Kaggle, navigate to the working directory of your model with the following command:

```bash
%cd /kaggle/working/NhaBeModel
```

## Running the Demo

To run a demo of the model, follow the steps below:

1. Import the necessary modules:

```python
import torch
import utils
from ConvLSTMModel import ConvLSTMModel
```

2. Set the device for running the model (CUDA or CPU):

```python
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
```

3. Load the pretrained model:

```python
model = ConvLSTMModel.from_pretrained("kag19/acompa_weather")
model.to(device)
```

4. Define the file paths for the input images and ground truth:

```python
image_1_path = '/kaggle/input/finaldataacompa/all_128/month_2/28/139.jpeg'
image_2_path = '/kaggle/input/finaldataacompa/all_128/month_2/28/140.jpeg'
image_3_path = '/kaggle/input/finaldataacompa/all_128/month_2/28/141.jpeg'
image_4_path = '/kaggle/input/finaldataacompa/all_128/month_2/28/142.jpeg'
image_5_path = '/kaggle/input/finaldataacompa/all_128/month_2/28/143.jpeg'
ground_truth_path = '/kaggle/input/finaldataacompa/all_128/month_2/28/144.jpeg'
file_paths = [image_1_path, image_2_path, image_3_path, image_4_path, image_5_path, ground_truth_path]
```

5. Visualize the input images and the model's prediction:

```python
visualize(file_paths, model)
```

This will visualize the sequence of input images along with the ground truth radar reflectivity image and the model’s predicted output.

## Credits

This project uses a ConvLSTM model pretrained on radar reflectivity data to predict weather patterns. If you use this model, please cite the original authors.
