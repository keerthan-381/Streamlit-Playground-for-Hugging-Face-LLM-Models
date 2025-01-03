# Streamlit-Playground-for-Hugging-Face-LLM-Models

This project is a Streamlit-based application that allows users to interact with various Hugging Face models for tasks like text generation, image generation, image captioning, text classification, summarization, and text-to-video generation.

## Features
- **Text Generation**
- **Text-to-Image Generation**
- **Image-to-Text (Image Captioning)**
- **Text Classification**
- **Summarization**
- **Text-to-Video Generation**

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/keerthan-381/Streamlit-Playground-for-Hugging-Face-LLM-Models.git
   cd Streamlit-Playground-for-Hugging-Face-LLM-Models.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Application Workflow

### 1. **Text Generation**
   - Select between two models:
     - **[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)**
     - **[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)**
   - Input your query and receive a generated response.

### 2. **Text-to-Image Generation**
   - Choose from these models:
     - **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)**
     - **[Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)**
   - Enter a prompt to generate an image.

### 3. **Image-to-Text (Image Captioning)**
   - Select a model for image captioning:
     - **[ViT-GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)**
     - **[BLIP Image Captioning](https://huggingface.co/Salesforce/blip-image-captioning-base)**
   - Upload an image to generate captions.

### 4. **Text Classification**
   - Uses **[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)** for financial text classification.
   - Enter text to classify and receive predictions.

### 5. **Summarization**
   - Uses **[BART-Large CNN](https://huggingface.co/facebook/bart-large-cnn)** for text summarization.
   - Enter a long text to get a summarized output.

### 6. **Text-to-Video Generation**
   - Uses **[Dreamlike Video 1.0](https://huggingface.co/dreamlike-playground/dreamlike-video-1.0)** for generating videos.
   - Input a prompt to generate a video.

## Usage Guide

### API Key
1. Obtain your Hugging Face API token from [Hugging Face](https://huggingface.co/settings/tokens).
2. Input the token in the sidebar of the application.

### Modes and Outputs
- Select the desired mode (e.g., Text Generation, Summarization) from the sidebar.
- Provide necessary inputs (text, image, or prompt) in the main application area.
- Click the corresponding button (e.g., "Get Response", "Generate Image") to execute the task.



### Models Used
| Task                | Model Name                    | Description                                                                                     |
|---------------------|-------------------------------|-------------------------------------------------------------------------------------------------|
| Text Generation     | Mistral-7B-Instruct-v0.2      | Lightweight language model for generating coherent text.                                        |
|                     | Mistral-7B-Instruct-v0.3      | Improved version of Mistral-7B with enhanced instruction-following capabilities.               |
| Text-to-Image       | FLUX.1-dev                    | Advanced image generation model capable of rendering complex visuals.                          |
|                     | Stable Diffusion v1.5         | A widely-used model for generating high-quality images from text prompts.                      |
| Image-to-Text       | ViT-GPT2                      | Combines Vision Transformer (ViT) and GPT-2 for image captioning.                              |
|                     | BLIP Image Captioning         | Robust model for creating human-like captions for images.                                      |
| Text Classification | ProsusAI/finbert              | Specialized in analyzing financial texts and extracting sentiments or classifications.          |
| Summarization       | BART-Large CNN                | Powerful summarization model ideal for processing lengthy articles or texts.                   |
| Text-to-Video       | Dreamlike Video 1.0           | Experimental model for generating videos from descriptive prompts.                             |

### Error Handling
The application has robust error-handling mechanisms to catch exceptions and provide user-friendly error messages.

## Acknowledgments
- Hugging Face for providing the models and inference APIs.
- Streamlit for creating an easy-to-use interface for web applications.
