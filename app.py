import streamlit as st
import base64
import requests
from dotenv import load_dotenv
import os
from pdf2image import convert_from_path
from io import BytesIO
from groq import Groq
import tempfile
import platform
import subprocess

# Load environment variables
load_dotenv()

# Fetch the GROQ API key from the .env file
API_KEY = os.getenv("GROQ_API_KEY")

# Set up the Groq client
client = Groq()

def check_poppler_installation():
    """
    Check if poppler is installed and accessible.
    Returns: bool indicating if poppler is properly installed
    """
    system = platform.system().lower()
    try:
        if system == "linux" or system == "darwin":
            subprocess.run(["pdftoppm", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif system == "windows":
            # Check if poppler is in PATH
            from shutil import which
            if which("pdftoppm.exe") is None:
                return False
        return True
    except FileNotFoundError:
        return False

def encode_image(file):
    """
    Encode an uploaded image file to a base64 string.
    :param file: File object from Streamlit uploader.
    :return: Base64-encoded string of the image.
    """
    encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"

def convert_pdf_to_images(pdf_file):
    """
    Convert each page of a PDF to an image.
    :param pdf_file: Streamlit UploadedFile object for a PDF.
    :return: List of base64-encoded images.
    """
    if not check_poppler_installation():
        st.error("""Poppler is not installed. Please install it:
        - MacOS: brew install poppler
        - Ubuntu/Debian: sudo apt-get install poppler-utils
        - Windows: Download from http://blog.alivate.com.au/poppler-windows/ and add to PATH""")
        return []

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            temp_file_path = tmp_file.name

        # Convert PDF to images
        images = convert_from_path(temp_file_path)
        encoded_images = []
        
        # Show progress bar for PDF conversion
        progress_bar = st.progress(0)
        for idx, image in enumerate(images):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            encoded_images.append(f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}")
            progress_bar.progress((idx + 1) / len(images))

        # Clean up the temporary file
        os.remove(temp_file_path)
        progress_bar.empty()

        return encoded_images

    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return []

def ocr_with_groq(image_content, model="llama-3.2-90b-vision-preview"):
    """
    Perform OCR on an image using the Groq API and convert the output to Markdown.
    :param image_content: Base64-encoded image or URL.
    :param model: Model to use, default is "llama-3.2-90b-vision-preview".
    :return: Extracted Markdown content.
    """
    try:
        # Define system prompt
        system_prompt = """Convert the provided image into Markdown format. Ensure that all content from the page is included, such as headers, footers, subtexts, images (with alt text if possible), tables, and any other elements.

        Requirements:
        - Output Only Markdown: Return solely the Markdown content without any additional explanations or comments.
        - No Delimiters: Do not use code fences or delimiters like \`\`\`markdown.
        - Complete Content: Do not omit any part of the page, including headers, footers, and subtext.
        """

        # Prepare message payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": image_content}},
                ],
            }
        ]

        # Send request to Groq API
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )

        return completion.choices[0].message.content

    except Exception as e:
        st.error(f"Error processing image with Groq: {str(e)}")
        return ""

# Streamlit app interface
st.title("OCR to Markdown with GROQ API")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG) or PDF", type=["jpg", "jpeg", "png", "pdf"])

model_choice = st.selectbox(
    "Select Model",
    options=["llama-3.2-90b-vision-preview"],
    index=0,
)

# Process the uploaded file
if uploaded_file is not None:
    with st.spinner("Processing file..."):
        # Check file type
        if uploaded_file.type == "application/pdf":
            st.info("Processing PDF file... This may take a moment.")
            # Convert PDF to images
            encoded_images = convert_pdf_to_images(uploaded_file)
        else:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            # Encode image directly
            encoded_images = [encode_image(uploaded_file)]
        
        if encoded_images:
            # Aggregate Markdown output
            markdown_output = ""
            for idx, img in enumerate(encoded_images):
                st.info(f"Processing page {idx + 1} of {len(encoded_images)}...")
                result = ocr_with_groq(img, model=model_choice)
                if result:
                    markdown_output += result + "\n\n"

            # Display Markdown output
            if markdown_output:
                st.markdown("### Extracted Markdown Content")
                st.text_area("Markdown Output", markdown_output, height=300)