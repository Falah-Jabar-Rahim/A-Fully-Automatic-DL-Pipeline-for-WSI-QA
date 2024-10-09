# A Fully Automatic Deep Learning Pipeline for WSI Quality Assessment
![WSI-QA](./WSI-QA.bmp)
# Abstract: 
In recent years, the use of deep learning (DL) methods, including convolutional neural networks (CNNs) and vision transformers (ViTs), has significantly advanced computational pathology, enhancing both diagnostic accuracy and efficiency. Whole Slide Imaging (WSI) plays a crucial role by providing detailed tissue samples for the analysis and training of DL models. However, WSIs often contain regions with artifacts such as tissue folds, blurring, as well as non-tissue regions (background), which can negatively impact DL model performance in diagnostic tasks. These artifacts are diagnostically irrelevant and can lead to inaccurate results. This paper proposes a fully automatic DL pipeline for WSI Quality Assessment (WSI-QA) that uses a fused model combining CNNs and ViTs to detect and exclude WSI regions with artifacts, ensuring that only qualified WSI regions are used to build DL-based computational pathology applications. The proposed pipeline employs a pixel-based segmentation model to classify WSI regions as either qualified or non-qualified based on the presence of artifacts. The proposed model was trained on a large and diverse dataset and validated with internal and external data from various human organs, scanners, and staining procedures. Quantitative and qualitative evaluations demonstrate the superiority of the proposed model, which outperforms state-of-the-art methods in WSI artifact detection. The proposed model consistently achieved over 95% accuracy, precision, recall, and F1 score across all artifact types. Furthermore, the WSI-QA pipeline shows strong generalization across different tissue types and scanning conditions.
# Setting Up the Pipeline
1. System Requirements: \
Operating System: Ubuntu 20.04 or 22.04, 
CUDA Version: 12.2, 
Python Version: 3.9 (using Conda environments)
2. Steps to Set Up the Pipeline
- Download the pipeline to your Desktop.
- Navigate to the downloaded pipeline folder.
2. Prepare Your Images:
Place your Whole Slide Images (WSI) into the folder named test_wsi.
3. Open Terminal:
Right-click within the pipeline folder and select “Open Terminal.”
Create a Conda Environment:
