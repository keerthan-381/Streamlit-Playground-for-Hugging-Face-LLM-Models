import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
import requests
from PIL import Image
from io import BytesIO
import base64

def get_llm_response(repo_id, query, sec_key):
    """Get response from the specified model."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=4096,
            temperature=0.7,
            token=sec_key
        )
        return llm.invoke(query)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def generate_image(repo_id, prompt, sec_key):
    """Generate an image from the specified model."""
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {sec_key}"}
    payload = {"inputs": prompt}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        image_data = response.content
        return Image.open(BytesIO(image_data))
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"

def image_to_base64(image):
    """Convert image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_caption(repo_id, image, sec_key):
    """Generate a caption for the given image."""
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {sec_key}"}

    # Convert image to base64
    image_base64 = image_to_base64(image)
    payload = {"inputs": image_base64}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        # Check if the response is a list or dict
        if isinstance(response.json(), list):
            return response.json()[0] if response.json() else "No caption generated."
        else:
            return response.json().get("generated_text", "No caption generated.")

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"

def classify_text(repo_id, text, sec_key):
    """Classify the given text using the specified model."""
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {sec_key}"}
    payload = {"inputs": text}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"

def summarize_text(repo_id, text, sec_key):
    """Summarize the given text using the specified model."""
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {sec_key}"}
    payload = {"inputs": text}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["summary_text"]
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"


def generate_video(repo_id, prompt, sec_key):
    """Generate a video from the specified model."""
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {sec_key}"}
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        # Save video as binary content
        video_data = response.content
        video_path = "/tmp/generated_video.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(video_data)
        return video_path
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"





st.set_page_config(page_title="Hugging Face LLM Playground", page_icon="🤖", layout="wide")

# Streamlit App
st.title("Hugging Face LLM Playground")

# Sidebar setup
st.sidebar.title("Configuration")
hf_token = st.sidebar.text_input("Enter your Hugging Face API token:", type="password")

if not hf_token:
    st.sidebar.warning("Please enter your Hugging Face API token to proceed.")
else:
    mode = st.sidebar.radio(
        "**Choose Mode**",
        ["Text Generation", "Text-to-Image Generation", "Image-to-Text", "Text Classification", "Summarization", "Text to Video Generation"]
    )

    if mode == "Text Generation":
        model_version = st.sidebar.radio(
            "**Text Generation Model**",
            ["Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.3"]
        )
        query = st.text_input("Enter your query:")

        if st.button("Get Response"):
            if query.strip() == "":
                st.error("Query cannot be empty. Please enter a valid query.")
            else:
                with st.spinner(f"Fetching response from {model_version}..."):
                    repo_id = ("mistralai/Mistral-7B-Instruct-v0.2" if model_version == "Mistral-7B-Instruct-v0.2"
                               else "mistralai/Mistral-7B-Instruct-v0.3")
                    response = get_llm_response(repo_id, query, hf_token)
                if response.startswith("An error occurred"):
                    st.error(response)
                else:
                    st.subheader(f"Response from {model_version}:")
                    st.write(response)

    elif mode == "Text-to-Image Generation":
        image_model = st.sidebar.radio(
            "**Text-to-Image Model**",
            ["black-forest-labs/FLUX.1-dev", "stable-diffusion-v1-5/stable-diffusion-v1-5"]
        )
        query = st.text_input("Enter your prompt for image generation:")

        if st.button("Generate Image"):
            if query.strip() == "":
                st.error("Prompt cannot be empty. Please enter a valid prompt.")
            else:
                with st.spinner(f"Generating image using {image_model}..."):
                    image = generate_image(image_model, query, hf_token)
                if isinstance(image, str):
                    st.error(image)
                else:
                    st.image(image, caption=f"Generated Image using {image_model}", =True)

    elif mode == "Image-to-Text":
        image_to_text_model = st.sidebar.radio(
            "**Image-to-Text Model**",
            ["nlpconnect/vit-gpt2-image-captioning", "Salesforce/blip-image-captioning-base"]
        )
        uploaded_image = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

        if st.button("Generate Caption"):
            if uploaded_image is None:
                st.error("Please upload an image.")
            else:
                with st.spinner(f"Generating caption using {image_to_text_model}..."):
                    image = Image.open(uploaded_image)
                    caption = generate_caption(image_to_text_model, image, hf_token)

                if isinstance(caption, str) and caption.startswith("An error occurred"):
                    st.error(caption)
                else:
                    st.image(uploaded_image, caption="Uploaded Image", =True)
                    st.subheader("Generated Caption:")
                    st.write(caption)

    elif mode == "Text Classification":
        text = st.text_area("Enter your text for classification:")

        if st.button("Classify Text"):
            if text.strip() == "":
                st.error("Text cannot be empty. Please enter some text.")
            else:
                with st.spinner(f"Classifying text using ProsusAI/finbert..."):
                    repo_id = "ProsusAI/finbert"
                    classification_result = classify_text(repo_id, text, hf_token)
                if isinstance(classification_result, str) and classification_result.startswith("An error occurred"):
                    st.error(classification_result)
                else:
                    st.subheader("Classification Result:")
                    st.write(classification_result)

    elif mode == "Summarization":
        summarization_model = "facebook/bart-large-cnn"
        text = st.text_area("Enter the text to summarize:")

        if st.button("Summarize Text"):
            if text.strip() == "":
                st.error("Text cannot be empty. Please enter some text.")
            else:
                with st.spinner(f"Summarizing text using {summarization_model}..."):
                    summary = summarize_text(summarization_model, text, hf_token)
                if isinstance(summary, str) and summary.startswith("An error occurred"):
                    st.error(summary)
                else:
                    st.subheader("Summary:")
                    st.write(summary)

    
    elif mode == "Text to Video Generation":
        video_model = st.sidebar.radio(
            "**Text to Video Model**",
            ["dreamlike-playground/dreamlike-video-1.0"]
        )
        prompt = st.text_input("Enter your prompt for video generation:")

        if st.button("Generate Video"):
            if prompt.strip() == "":
                st.error("Prompt cannot be empty. Please enter a valid prompt.")
            else:
                with st.spinner(f"Generating video using {video_model}..."):
                    video_path = generate_video(video_model, prompt, hf_token)
                if isinstance(video_path, str) and video_path.startswith("An error occurred"):
                    st.error(video_path)
                else:
                    st.subheader("Generated Video:")
                    st.video(video_path)

# Additional details
st.markdown("---")
