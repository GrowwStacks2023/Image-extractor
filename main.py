from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import base64
import requests
import os
from io import BytesIO


app = Flask(__name__)

# Get from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
IMAGEKIT_PRIVATE_KEY = os.environ.get('IMAGEKIT_PRIVATE_KEY')
IMAGEKIT_PUBLIC_KEY = os.environ.get('IMAGEKIT_PUBLIC_KEY')
IMAGEKIT_URL_ENDPOINT = os.environ.get('IMAGEKIT_URL_ENDPOINT')

def upload_to_imagekit(image_bytes, filename):
    """Upload image to ImageKit"""
    url = "https://upload.imagekit.io/api/v1/files/upload"
    
    # Correct format for ImageKit API
    data = {
        'file': base64.b64encode(image_bytes).decode('utf-8'),
        'fileName': filename,
        'useUniqueFileName': 'true'
    }
    
    response = requests.post(
        url,
        data=data,  # Use 'data' not 'files'
        auth=(IMAGEKIT_PRIVATE_KEY, '')
    )
    
    if response.status_code == 200:
        return response.json()['url']
    else:
        print(f"ImageKit upload failed: {response.status_code}, {response.text}")
        return None

def describe_image(image_url):
    """Use GPT-4o-mini to describe image"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this Tesla manual diagram in 1-2 sentences. Focus on key components."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        }],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"OpenAI API failed: {response.status_code}")
    except Exception as e:
        print(f"OpenAI error: {e}")
    return "Image from manual"

@app.route('/extract-images', methods=['POST'])
def extract_images():
    """Main endpoint: Extract images from PDF"""
    try:
        # Get PDF file
        pdf_file = request.files.get('pdf')
        if not pdf_file:
            return jsonify({"error": "No PDF file provided"}), 400
        
        pdf_data = pdf_file.read()
        pdf = fitz.open(stream=pdf_data, filetype="pdf")
        
        results = []
        total_pages = len(pdf)  # Store before processing
        
        for page_num in range(total_pages):
            page = pdf[page_num]
            page_images = []
            
            # Extract images from this page (SAME AS YOUR LOCAL SCRIPT)
            for img_index, img in enumerate(page.get_images()):
                try:
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    print(f"Found image on page {page_num+1}: {len(image_bytes)} bytes, format: {base_image['ext']}")
                    
                    # Create filename
                    filename = f"page_{page_num+1}_img_{img_index}.{base_image['ext']}"
                    
                    # Upload to ImageKit
                    image_url = upload_to_imagekit(image_bytes, filename)
                    
                    if image_url:
                        # Describe the image
                        description = describe_image(image_url)
                        
                        page_images.append({
                            "filename": filename,
                            "url": image_url,
                            "description": description
                        })
                        print(f"Successfully processed: {filename}")
                    else:
                        print(f"Failed to upload: {filename}")
                        
                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num}: {e}")
                    continue
            
            results.append({
                "page": page_num + 1,
                "image_count": len(page_images),
                "images": page_images
            })
        
        pdf.close()
        
        return jsonify({
            "success": True,
            "total_pages": total_pages,
            "pages": results
        })
    
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/extract-from-url', methods=['POST'])
def extract_from_url():
    """Extract images from PDF URL"""
    try:
        data = request.get_json()
        pdf_url = data.get('pdf_url')
        
        if not pdf_url:
            return jsonify({"error": "No PDF URL provided"}), 400
        
        print(f"Downloading PDF from: {pdf_url}")
        
        # Download PDF from URL
        response = requests.get(pdf_url)
        pdf_data = response.content
        
        print(f"PDF downloaded: {len(pdf_data)} bytes")
        
        pdf = fitz.open(stream=pdf_data, filetype="pdf")
        
        results = []
        total_pages = len(pdf)  # Store before processing
        
        for page_num in range(total_pages):
            page = pdf[page_num]
            page_images = []
            
            # Extract images from this page
            image_list = page.get_images()
            print(f"Page {page_num+1}: Found {len(image_list)} images")
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    print(f"Processing page {page_num+1}, image {img_index}: {len(image_bytes)} bytes")
                    
                    filename = f"page_{page_num+1}_img_{img_index}.{base_image['ext']}"
                    
                    # Upload to ImageKit
                    image_url = upload_to_imagekit(image_bytes, filename)
                    
                    if image_url:
                        description = describe_image(image_url)
                        
                        page_images.append({
                            "filename": filename,
                            "url": image_url,
                            "description": description
                        })
                        print(f"✓ Uploaded: {filename}")
                    else:
                        print(f"✗ Upload failed: {filename}")
                        
                except Exception as e:
                    print(f"Error on page {page_num+1}, image {img_index}: {e}")
                    continue
            
            results.append({
                "page": page_num + 1,
                "image_count": len(page_images),
                "images": page_images
            })
        
        pdf.close()
        
        return jsonify({
            "success": True,
            "total_pages": total_pages,
            "pages": results
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Tesla PDF Image Extractor",
        "imagekit_configured": bool(IMAGEKIT_PRIVATE_KEY),
        "openai_configured": bool(OPENAI_API_KEY)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)