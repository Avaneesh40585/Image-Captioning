# 🖼️ Image Captioning System

**Advanced Image Captioning System that generates natural language descriptions for any image using state-of-the-art Transformer architecture.**

The model architecture combines EfficientNetB0 CNN for visual feature extraction with a custom Transformer encoder-decoder for sequential text generation. Built entirely with TensorFlow/Keras, this system represents a modern approach to the image-to-text generation task.

---

## 📋 Table of Contents

1. 📊 [About Dataset](#about-dataset)  
2. 🏗️ [Model Architecture](#model-architecture)  
3. ✨ [Key Features](#key-features)  
4. 🔄 [Training Pipeline Overview](#training-pipeline-overview)  
5. 📋 [Requirements](#requirements)  
6. 📁 [Project Structure](#project-structure)  
7. 🎯 [Usage](#usage)  
8. 🎨 [Results](#results)  
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
Input Image (299×299×3)
        ↓
EfficientNetB0 CNN (Feature Extractor)
        ↓
Spatial Features → Dense(768D) → Reshape
        ↓
Transformer Encoder (Multi-Head Attention)
        ↓
Transformer Decoder
   ↑                       ↓
Text Embeddings +      Masked + 
Positional Encoding    Cross Attention
        ↓
Vocabulary Projection (12K tokens)
        ↓
Generated Caption
```

**Architecture Specifications:**
- **CNN Backbone**: EfficientNetB0 (ImageNet pre-trained, fine-tuned last 30 layers)
- **Transformer Encoder**: 3 layers, 12 attention heads, 768D embeddings
- **Transformer Decoder**: 3 layers with causal masking and cross-attention
- **Text Processing**: 12K vocabulary with learned positional embeddings
- **Feed-Forward Dimension**: 2048D with GELU activation
- **Dropout Rate**: 15% for regularization

**A comprehensive explanation of the code is provided in the notebook itself.**

---

## Key Features

### Advanced Architecture
- **Custom Transformer Implementation**: Hand-built encoder-decoder blocks with proper masking
- **Multi-Caption Training**: Processes 5 captions per image with individual gradient updates
- **Fine-tuned CNN**: EfficientNetB0 with selective layer unfreezing for optimal feature extraction
- **Positional Embeddings**: Learned position encodings for sequence understanding

### Training Optimizations
- **Advanced Learning Rate Scheduling**: Warmup + cosine decay with AdamW optimizer
- **Comprehensive Data Augmentation**: 5-layer pipeline (flip, rotation, contrast, brightness, zoom)
- **Gradient Clipping**: Norm-based clipping for training stability
- **Smart Callbacks**: Early stopping and model checkpointing with best weight restoration

### Evaluation & Testing
- **Multiple Metrics**: BLEU-4 and CIDEr scoring for comprehensive evaluation
- **Flexible Generation**: Both greedy and temperature-based sampling
- **Interactive Testing**: Built-in functions for custom image evaluation
- **Robust Error Handling**: Graceful failure recovery and model saving

---

## Training Pipeline Overview

### Phase 1 – Data Setup & Preprocessing

- Downloads **Flickr8K dataset** automatically via `wget` commands (~1GB total)
- Extracts **8,000 images** and **caption files** from zip archives
- Loads and parses `Flickr8k.token.txt` containing **40,000 captions**
- Filters malformed captions and **builds vocabulary** (~12K tokens)
- Creates **train/validation split** (6,400 / 1,600 images)
- Implements **robust error handling** for missing or corrupted files

### Phase 2 – Model Architecture Construction

- Builds **EfficientNetB0 CNN** feature extractor with pretrained **ImageNet weights**
- Constructs **custom Transformer encoder blocks** with multi-head self-attention
- Creates **Transformer decoder** with **causal masking** and **cross-attention**
- Sets up **learned positional embeddings** and **text preprocessing pipeline**
- Configures **5-layer image augmentation** strategy
- Initializes complete `ImageCaptioningModel` with all components

### Phase 3 – Advanced Training Process

- Implements **learning rate scheduling** (warmup + cosine decay)
- Configures **AdamW optimizer** with **gradient clipping** and **weight decay**
- Processes **5 captions per image** with individual gradient updates
- Applies **comprehensive callbacks** (early stopping, model checkpointing)
- Monitors **training/validation metrics** in real-time
- Automatically saves **best performing model** based on validation loss

### Phase 4 – Evaluation & Testing

- Evaluates model performance using **BLEU-4** and **CIDEr** metrics
- Tests on **random validation samples** with **automatic caption generation**
- Provides **interactive testing functions** for custom image inputs
- Implements both **greedy decoding** and **temperature-based sampling**
- Saves **final trained model** and displays comprehensive results
- Generates **performance reports** with detailed metric analysis

---

## Requirements
```
tensorflow>=2.13.0
keras>=2.13.0
numpy>=1.21.0
matplotlib>=3.5.0
nltk>=3.8.0
```

These requirements can be easily installed by: `pip install -r requirements.txt`

---

## Project Structure
```
image-captioning-system/
├── Image Caption Generator.ipynb    # Complete implementation notebook
├── README.md                        # This file
├── requirements.txt                 # Dependencies
├── models/                          # Saved models (generated after training)
│   ├── best_caption_model.keras
│   └── final_caption_model.keras
├── Flicker8k_Dataset/              # Auto-downloaded image dataset
│   └── *.jpg                       # 8,000 images
└── Flickr8k_text/                  # Auto-downloaded text dataset
    ├── Flickr8k.token.txt         # Main caption annotations (used in training)
    ├── ExpertAnnotations.txt       # Expert-written captions
    ├── CrowdFlowerAnnotations.txt  # Crowd-sourced caption quality ratings
    ├── Flickr_8k.trainImages.txt  # Training set image list
    ├── Flickr_8k.devImages.txt    # Development/validation set image list
    └── Flickr_8k.testImages.txt   # Test set image list
```
---

## Usage

### 1. Open the Notebook

Open the training notebook in your preferred environment (e.g., Jupyter, Colab, VS Code).

### 2. Execute the Complete Training Pipeline

- Run all cells sequentially to go through all **4 phases** (see [Training Pipeline Overview](#training-pipeline-overview) )
- **Dataset (~1GB)** downloads automatically – no manual setup required
- **Vocabulary building** and **preprocessing** are handled internally

### 3. Monitor Training Progress

- Real-time **loss/accuracy metrics** displayed during training
- **BLEU-4** and **CIDEr** scores computed during validation
- **Best model** is automatically saved as: `models/best_caption_model.keras`
- **Early stopping** ensures optimal performance and prevents overfitting
- Estimated training time: **~1 hour on GPU** & **~2–3 hours on CPU**

### 4. Test Your Trained Model

- Automatic validation testing on **random images** after training
- **Interactive testing functions** available for custom images:
```
test_image("path/to/image.jpg")  # Standard caption generation
test_with_sampling("path/to/image.jpg", temperature=0.8)  # Creative captions
```
- **Final model** is saved as:`models/final_caption_model.keras`
- Comprehensive evaluation metrics displayed: `BLEU-4: ~0.19` , `CIDEr: ~0.14`

---

## Results

| Image | Generated Caption |
|-------|------------------|
| ![Dog in grass](https://github.com/user-attachments/assets/07743043-bf9a-4738-af91-0725177f384b) | **Generated Caption:** *a black and white dog is running through the grass* |
| ![Dog in water](https://github.com/user-attachments/assets/1c04991d-205e-42bb-befd-e839ab9bd3b4) | **Generated Caption:** * a black dog is jumping into the water* |
| ![Girl on a bike](https://github.com/user-attachments/assets/539a8058-4eb1-48b1-a892-852a97bc2652) | **Generated Caption:** *a young boy rides his bike down a hill* |


### Metric Interpretation:
- **BLEU-4 (0.1926)**: Competitive score for Flickr8K (typical range: 0.15-0.35)
- **CIDEr (0.1395)**: Room for improvement (typical range: 0.3-1.2)
- **Training Accuracy**: ~85-90% on final epochs
- **Validation Accuracy**: ~80-85% with good generalization

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

## Contributing

Contributions welcome! Areas for improvement:
- Beam search implementation
- Attention visualization
- Additional evaluation metrics
- Support for other datasets
- Add support for pre-trained models & inferencing
- Model architecture experiments

---

**⭐ If this implementation helps your research or project, please consider giving it a star!**






