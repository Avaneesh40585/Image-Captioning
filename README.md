# 🖼️ Image Captioning System

**Advanced Image Captioning System that generates natural language descriptions for any image using state-of-the-art Transformer architecture.**

The model architecture combines EfficientNetB0 CNN for visual feature extraction with a custom Transformer encoder-decoder for sequential text generation. Built entirely with TensorFlow/Keras, this system represents a modern approach to the image-to-text generation task.

---

## 📋 Table of Contents

1. 📊 [About Dataset](#about-dataset)  
2. 🏗️ [Model Architecture](#model-architecture)  
3. ✨ [Key Features](#key-features)  
4. 🔄 [Training Pipeline Overview](#training-pipeline-overview)  
   - Phase 1 – Data Setup  
   - Phase 2 – Model Architecture  
   - Phase 3 – Training Process  
   - Phase 4 – Testing & Evaluation  
   - ⏱️ Total Training Time  
5. 📋 [Requirements](#requirements)  
6. 📁 [Project Structure](#project-structure)  
7. 🎯 [Usage](#usage)  
   - 🔹 Pre-trained Model (Recommended)  
   - 🔸 From Scratch  
8. 🎨 [Results](#results)  
   - Generation Modes  
9. 📚 [References](#references)  
10. 📄 [License](#license)

---

## About Dataset

The model is trained on **Flickr8K Dataset** containing:
- 8,000 high-quality images
- 40,000 human-written captions (5 per image)
- Automatic download and preprocessing
- 80/20 train/validation split

*Can be easily adapted for Flickr30K, MS COCO, or custom datasets*

Download instructions and additional details are available here: [Flickr8k Datasets](https://github.com/Avaneesh40585/Image-Captioning/releases/tag/dataset) 

---

## Model Architecture
```
Input Image (299x299x3)
↓
EfficientNetB0 CNN
(Feature Extractor)
↓
Spatial Features → Sequential Embeddings (512D)
↓
Transformer Encoder
(Multi-Head Attention)
↓
Transformer Decoder ← Text Embeddings
(Masked + Cross Attention)
↓
Vocabulary Projection (12K tokens)
↓
Generated Caption
```

**Key Components:**
- **CNN Backbone**: EfficientNetB0 (ImageNet pre-trained, frozen)
- **Transformer Encoder**: 3 layers, 8 attention heads, 512D embeddings
- **Transformer Decoder**: 3 layers with causal masking and cross-attention
- **Text Processing**: 12K vocabulary with learned positional embeddings

**A comprehensive explanation of the code is provided in the notebook itself.**

---
## Key Features

- **Multi-Caption Training**: Processes 5 different captions per image simultaneously for robust learning  
- **Advanced Transformer Architecture**: Custom encoder-decoder blocks with multi-head attention and proper masking  
- **Intelligent Learning Rate Scheduling**: Warmup phase followed by cosine decay for optimal convergence  
- **Comprehensive Data Augmentation**: 5-layer image transformation pipeline (flip, rotation, contrast, brightness, zoom)  
- **Automatic Model Recovery**: Smart checkpointing system saves models during training interruptions  
- **Interactive Testing Suite**: Built-in functions for validation testing and custom image evaluation  

---

## Training Pipeline Overview

### Phase 1 – Data Setup (Cells 1–15)
- Downloads Flickr8K dataset (~1 GB) automatically via `wget`  
- Extracts 8,000 images and 40,000 caption annotations  
- Preprocesses text: lowercasing, adding special tokens, and vocabulary building  
- Creates 80/20 train/validation split: 6,400 training / 1,600 validation images  

### Phase 2 – Model Architecture (Cells 16–25)
- Builds **EfficientNetB0** CNN feature extractor with frozen ImageNet weights  
- Constructs custom Transformer encoder/decoder blocks  
- Sets up 12K vocabulary with learned positional embeddings  
- Configures a 5-layer image augmentation pipeline  

### Phase 3 – Training Process (Cells 26–35)
- Trains for 30 epochs using the **AdamW optimizer**  
- Applies warmup (first 10% of steps) + cosine decay for learning rate scheduling  
- Processes **5 captions per image** with individual gradient updates  
- Monitors training/validation loss and accuracy in real-time  
- Automatically saves the best model when validation loss improves  

### Phase 4 – Testing & Evaluation (Cells 36–40)
- Tests on 3 random validation images automatically  
- Includes `test_image()` and `test_with_sampling()` functions  
- Displays generated captions alongside input images  
- Saves final model as `final_caption_model.keras`  

### ⏱️ Total Training Time
- **GPU**: ~0.5-1 hour  
- **CPU**: ~2–3 hours

---

## Requirements
```
tensorflow>=2.13.0
keras>=2.13.0
numpy>=1.21.0
matplotlib>=3.5.0
```

These requirements can be easily installed by: `pip install -r requirements.txt`

---

## Project Structure
```
image-captioning-system/
├── Image Caption Generator.ipynb    # Complete implementation notebook
├── README.md                        # This file
├── requirements.txt                 # Dependencies
└── models/                          # Saved models (generated after training)
    ├── best_caption_model.keras
    └── final_caption_model.keras
```
---

## Usage

### 🔹 Pre-trained Model (Recommended)

1. **Download pre-trained weights** from [Releases](https://github.com/Avaneesh40585/Image-Captioning/releases/tag/v1.0-weights):
   - `best_caption_model.keras` - Best performing model
   - Follow the “How to Use” section from the [Releases](https://github.com/Avaneesh40585/Image-Captioning/releases/tag/v1.0-weights) page for detailed instructions.

2. **Open the notebook.**
```jupyter notebook “Image Caption Generator.ipynb”```

3. **Load and use the pre-trained model:**
   - Navigate to the “Model Loading & Inference” section (Cells 41–45)
   - Load the downloaded model:
     ```python
     caption_model = keras.models.load_model('best_caption_model.keras')
     ```
   - Use for standard captions:
     ```python
     test_image('your_image.jpg')
     ```
   - Use for creative captions:
     ```python
     test_with_sampling('your_image.jpg', temperature=0.8)
     ```

### 🔸 From Scratch

1. **Open the notebook.**
```jupyter notebook “Image Caption Generator.ipynb”```

2. **Execute the complete training pipeline:**  
   Run all cells sequentially to go through all 4 phases (see [Training Pipeline Overview](#training-pipeline-overview).)

3. **Monitor training progress:**
   - Real-time loss/accuracy metrics displayed during training  
   - Best model automatically saved as `best_caption_model.keras`  
   - Training completes in:
     - ~0.5–1 hour on **GPU**  
     - ~2–3 hours on **CPU**

4. **Test your trained model:**
   - Automatic validation testing on 3 random images  
   - Use provided functions to test with your own images  
   - Final model saved as `final_caption_model.keras`

---

## Results

| Image | Generated Caption |
|-------|------------------|
| ![Dog in grass](https://github.com/user-attachments/assets/07743043-bf9a-4738-af91-0725177f384b) | **Generated Caption:** *a dog runs through the grass* |
| ![Dog in water](https://github.com/user-attachments/assets/1c04991d-205e-42bb-befd-e839ab9bd3b4) | **Generated Caption:** *a black dog is jumping into the water* |
| ![Girl on a bike](https://github.com/user-attachments/assets/539a8058-4eb1-48b1-a892-852a97bc2652) | **Generated Caption:** *a young girl in a pink shirt is riding a bike* |

### Generation Modes

#### `test_image()` – Standard Caption Generation
- Always generates the same caption for the same image  
- Uses the most confident/probable word at each step  
- Produces reliable, consistent results  
- Best for when you need predictable captions  

#### `test_with_sampling()` – Creative Caption Generation
- Generates different captions each time you run it  
- Uses temperature parameter to control randomness  
- Lower temperature (e.g., `0.3`) = more conservative, similar to standard  
- Higher temperature (e.g., `0.8–1.0`) = more creative and varied captions  
- Best for when you want diverse caption options  

---

## References

**[1]** Vaswani, Ashish, et al. ["Attention is all you need."](https://arxiv.org/abs/1706.03762) Advances in neural information processing systems 30 (2017).

**[2]** Tan, Mingxing, and Quoc V. Le. ["EfficientNet: Rethinking model scaling for convolutional neural networks."](https://arxiv.org/abs/1905.11946) International conference on machine learning. PMLR, 2019.

**[3]** Vinyals, Oriol, et al. ["Show and tell: A neural image caption generator."](https://arxiv.org/abs/1411.4555) Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

**[4]** Xu, Kelvin, et al. ["Show, attend and tell: Neural image caption generation with visual attention."](https://arxiv.org/abs/1502.03044) International conference on machine learning. PMLR, 2015.

---

## License

MIT License. See [LICENSE](LICENSE) file for details.

---







