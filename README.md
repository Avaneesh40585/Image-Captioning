# 🖼️ Image Captioning System

**Advanced Image Captioning System that generates natural language descriptions for any image using state-of-the-art Transformer architecture.**

The model architecture combines EfficientNetB0 CNN for visual feature extraction with a custom Transformer encoder-decoder for sequential text generation. Built entirely with TensorFlow/Keras, this system represents a modern approach to the image-to-text generation task.

The project features a complete end-to-end pipeline with automatic dataset handling, advanced training techniques, and multiple inference modes for flexible caption generation.

## 📊 Dataset

The model is trained on **Flickr8K Dataset** containing:
- 8,000 high-quality images
- 40,000 human-written captions (5 per image)
- Automatic download and preprocessing
- 80/20 train/validation split

*Can be easily adapted for Flickr30K, MS COCO, or custom datasets*

How to Download & More about it over here: ![Flickr8k Datasets](https://github.com/Avaneesh40585/Image-Captioning/releases/tag/dataset) 

## 🏗️ Model Architecture
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

## 🚀 Performance

The model has been trained for **30 epochs** on **6,400 training samples** of Flickr8K Dataset with advanced optimization techniques:

- **Training Samples**: 6,400 images (32,000 captions)
- **Validation Samples**: 1,600 images (8,000 captions)
- **Multi-Caption Training**: 5 captions processed per image
- **Advanced Scheduling**: Warmup + cosine decay learning rate
- **Data Augmentation**: 5-layer image transformation pipeline
- **Model Size**: ~250MB optimized architecture

*Achieves high-quality, grammatically correct captions with diverse vocabulary*

## 📋 Requirements
<code>
```
tensorflow>=2.13.0
keras>=2.13.0
numpy>=1.21.0
matplotlib>=3.5.0
```
</code>

These requirements can be easily installed by: `pip install -r requirements.txt`

## 📁 Project Structure
```
image-captioning-system/
├── Image Caption Generator.ipynb    # Complete implementation notebook
├── README.md                        # This file
├── requirements.txt                 # Dependencies
└── models/                          # Saved models (generated after training)
    ├── best_caption_model.keras
    └── final_caption_model.keras
```


## 🎯 Usage

### Pre-trained Model (Recommended)

1. **Download pre-trained weights** from [releases](https://github.com/Avaneesh40585/Image-Captioning/releases/tag/v1.0-weights):
   - `best_caption_model.keras` - Best performing model

2. **Open the notebook**:
<code>
```jupyter notebook “Image Caption Generator.ipynb”```
</code>

4. **Load pre-trained model** (in notebook):
   Follow the "How to Use" section from the above link to proceed.
   

### From Scratch

1. **Open the notebook**:
<code>
```jupyter notebook “Image Caption Generator.ipynb”```
</code>

3. **Run all cells** to:
   - Automatically download Flickr8K dataset
   - Build the complete model architecture
   - Train with optimal hyperparameters
   - Save the best model checkpoint

**The notebook handles:**
- Automatic dataset setup (no manual downloads needed)
- Multi-caption training (5 captions per image)
- Advanced learning rate scheduling
- Early stopping and model checkpointing
- Real-time training monitoring
- Automatic validation and testing


## 🎨 Results

| Image | Generated Caption |
|-------|------------------|
| ![1](https://github.com/user-attachments/assets/07743043-bf9a-4738-af91-0725177f384b) | **Generated Caption:** *a dog runs through the grass* |
| ![2](https://github.com/user-attachments/assets/1c04991d-205e-42bb-befd-e839ab9bd3b4) | **Generated Caption:** *a black dog is jumping into the water* |
| ![3](https://github.com/user-attachments/assets/539a8058-4eb1-48b1-a892-852a97bc2652) | **Generated Caption:** *a young girl in a pink shirt is riding a bike* |

### Generation Modes

**Greedy Decoding:**
- Most probable token selection
- Consistent, coherent captions
- Best for reliable results

**Temperature Sampling:**
- Controlled creativity (T=0.1 to 1.2)
- Diverse caption generation
- Adjustable creativity vs coherence


## 🚀 Key Innovations

- **🎯 Multi-Caption Training**: Simultaneous processing of 5 captions per image
- **🧠 Advanced Attention**: Custom Transformer blocks with proper masking
- **📊 Smart Scheduling**: Warmup + cosine decay optimization
- **🎨 Rich Augmentation**: 5-layer image transformation pipeline
- **💾 Robust Recovery**: Automatic model saving for all scenarios
- **🔄 Interactive Testing**: Built-in validation and custom image testing


## 📚 References

**[1]** Vaswani, Ashish, et al. *"Attention is all you need."* Advances in neural information processing systems 30 (2017).

**[2]** Tan, Mingxing, and Quoc V. Le. *"EfficientNet: Rethinking model scaling for convolutional neural networks."* International conference on machine learning. PMLR, 2019.

**[3]** Vinyals, Oriol, et al. *"Show and tell: A neural image caption generator."* Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

**[4]** Xu, Kelvin, et al. *"Show, attend and tell: Neural image caption generation with visual attention."* International conference on machine learning. PMLR, 2015.


## 📄 License

MIT License. See [LICENSE](LICENSE) file for details.

---

⭐ **Star this repository if you find it helpful!**

🔗 **Share your generated captions using #TransformerImageCaptioning**

💡 **Contribute to make it even better!**







