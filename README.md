# A Fully Automatic Deep Learning Pipeline for WSI Quality Assessment
![WSI-QA](./WSI-QA.bmp)
# Abstract: 
In recent years, the use of deep learning (DL) methods, including convolutional neural networks (CNNs) and vision transformers (ViTs), has significantly advanced computational pathology, enhancing both diagnostic accuracy and efficiency. Whole Slide Imaging (WSI) plays a crucial role by providing detailed tissue samples for the analysis and training of DL models. However, WSIs often contain regions with artifacts such as tissue folds, blurring, as well as non-tissue regions (background), which can negatively impact DL model performance in diagnostic tasks. These artifacts are diagnostically irrelevant and can lead to inaccurate results. This paper proposes a fully automatic DL pipeline for WSI Quality Assessment (WSI-QA) that uses a fused model combining CNNs and ViTs to detect and exclude WSI regions with artifacts, ensuring that only qualified WSI regions are used to build DL-based computational pathology applications. The proposed pipeline employs a pixel-based segmentation model to classify WSI regions as either qualified or non-qualified based on the presence of artifacts. The proposed model was trained on a large and diverse dataset and validated with internal and external data from various human organs, scanners, and staining procedures. Quantitative and qualitative evaluations demonstrate the superiority of the proposed model, which outperforms state-of-the-art methods in WSI artifact detection. The proposed model consistently achieved over 95% accuracy, precision, recall, and F1 score across all artifact types. Furthermore, the WSI-QA pipeline shows strong generalization across different tissue types and scanning conditions.
# Setting Up the Pipeline
1. System requirements:
- Ubuntu 20.04 or 22.04
- CUDA version: 12.2
- Python version: 3.9 (using conda environments)
- Anaconda version 23.7.4

2. Steps to Set Up the Pipeline:
- Download the pipeline to your Desktop
- Navigate to the downloaded pipeline folder
- Right-click within the pipeline folder and select `Open Terminal`
- Create a conda environment:
```bash
  conda create -n WSI-QA python=3.9
```
- Activate the environment:
```bash
  conda activate WSI-QA
```
- Install required packages:
```bash
  pip install -r requirements.txt
```

# Running Inference

- Place your Whole Slide Image (WSI) into the `test_wsi` folder
- In the terminal execute:
```bash
  python test_wsi.py
```
- After running the inference, you will obtain the following outputs:
  - A thumbnail image of WSI
  - A thumbnail image of WSI with regions of interest identified
  - A segmentation mask highlighting segmented regions of the WSI
  - Excel file contains statistics on identified artifacts
  - A folder named `qualified` containing qualified tiles
  - A folder named `unqualified` containing unqualified tiles


# Training

- Visit https://drive.google.com/drive/folders/1mbnLH1JIztTMw7Cgv8pSzNxba-aGv1jT?usp=share_link and download the artifact datasets
- Extract and place the dataset into a folder named `train_dataset`
- Within `train_dataset`, refer to the example files provided to understand the structure
- Create two files, `train_images.txt` and `train_masks.txt`, with lists of the corresponding image and mask paths, that used for training. The datset can be split to training (e.g., 85%) and testing (e.g., 15%)

     Example content for `train_images.txt`:
     ```
     path/to/image1.png
     path/to/image2.png
     ...
     ```
     Example content for `train_masks.txt`:
     ```
     path/to/mask1.png
     path/to/mask2.png
     ...
     ```
   - Create an account on [Weights and Biases](https://docs.wandb.ai)
   - After signing up, go to your account settings and obtain your API key. It will look something like: `wandb.login(key='xxxxxxxxxxxxx')`
   - Open the file `trainer.py`
   - Find the line where the Weights and Biases login is required
   - Update it with your API key like this:
     ```python
     wandb.login(key='your_actual_key_here')
     ```

### 5. **Run the Training**:
   - Open a terminal in the directory where `train.py` is located.
   - Run the following command to start the training:
     ```bash
     python train.py
     ```

### 6. **Track and Visualize the Training Process**:
   - When training starts, a link to the Weights and Biases interface will appear in the terminal.
   - Click on the link to track and visualize the progress of your training.

### 7. **Find the Saved Weights**:
   - After the training is complete, the weights will be saved in the `logs` folder within your project directory.

Let me know if you need help with any specific step!









download the artifact datasets from: https://drive.google.com/drive/folders/1mbnLH1JIztTMw7Cgv8pSzNxba-aGv1jT?usp=share_link


put the dataset into folder "train_dataset"

generate trnaing and testing image lists  "train_images.txt" and "train_masks.txt". see  the examples are provided in folder "train_dataset" as a guidance

you need to create an acoount in https://docs.wandb.ai, then get login key "wandb.login(key='xxxxxxxxxxxxx')"

Open the file "trainer.py" and update the  the login key 

in terminal run "python train.py" to start trnaing 

you can track and visulize the trnaing process Weights and Biases interface  with bu cliking on the like generated in the trminal when the trnaing is started

when the trnaing is finishe the weights are saved in the folder "logs"








