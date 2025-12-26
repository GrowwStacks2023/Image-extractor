from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import base64
import requests
import os
import time
import logging
from functools import wraps
from io import BytesIO

app = Flask(__name__)

# Setup proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
IMAGEKIT_PRIVATE_KEY = os.environ.get('IMAGEKIT_PRIVATE_KEY')
IMAGEKIT_PUBLIC_KEY = os.environ.get('IMAGEKIT_PUBLIC_KEY')
IMAGEKIT_URL_ENDPOINT = os.environ.get('IMAGEKIT_URL_ENDPOINT')

def retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            
            raise last_exception
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, initial_delay=2, backoff_factor=2)
def upload_to_imagekit(image_bytes, filename):
    """Upload image to ImageKit with retry logic"""
    url = "https://upload.imagekit.io/api/v1/files/upload"
    
    data = {
        'file': base64.b64encode(image_bytes).decode('utf-8'),
        'fileName': filename,
        'useUniqueFileName': 'true'
    }
    
    response = requests.post(
        url,
        data=data,
        auth=(IMAGEKIT_PRIVATE_KEY, ''),
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()['url']
    else:
        error_msg = f"ImageKit upload failed: {response.status_code}, {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)

@retry_with_backoff(max_retries=5, initial_delay=2, backoff_factor=2)
def describe_image(image_url):
    """Use GPT-4o-mini to describe image with retry and rate limit handling"""
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
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    # Handle rate limiting specifically
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
        time.sleep(retry_after)
        raise Exception("Rate limited - retrying")
    
    if response.status_code == 200:
        description = response.json()['choices'][0]['message']['content']
        return description, True
    else:
        error_msg = f"OpenAI API failed: {response.status_code}, {response.text[:200]}"
        logger.error(error_msg)
        raise Exception(error_msg)

def process_single_image(pdf, page_num, img_index, img):
    """Process a single image with complete error handling"""
    result = {
        "filename": None,
        "url": None,
        "description": None,
        "description_status": "not_attempted",
        "upload_status": "failed",
        "error": None
    }
    
    try:
        xref = img[0]
        base_image = pdf.extract_image(xref)
        image_bytes = base_image["image"]
        
        logger.info(f"Processing page {page_num+1}, image {img_index}: {len(image_bytes)} bytes, format: {base_image['ext']}")
        
        filename = f"page_{page_num+1}_img_{img_index}.{base_image['ext']}"
        result["filename"] = filename
        
        # Step 1: Upload to ImageKit
        try:
            image_url = upload_to_imagekit(image_bytes, filename)
            result["url"] = image_url
            result["upload_status"] = "success"
            logger.info(f"✓ Uploaded: {filename}")
        except Exception as e:
            result["error"] = f"Upload failed: {str(e)}"
            result["description"] = "Not able to describe - upload failed"
            result["description_status"] = "failed"
            logger.error(f"✗ Upload failed for {filename}: {e}")
            return result
        
        # Step 2: Describe the image (only if upload succeeded)
        try:
            description, description_success = describe_image(image_url)
            result["description"] = description
            result["description_status"] = "success"
            logger.info(f"✓ Described: {filename}")
        except Exception as e:
            result["description"] = "Not able to describe"
            result["description_status"] = "failed"
            result["error"] = f"Description failed: {str(e)}"
            logger.error(f"✗ Description failed for {filename}: {e}")
        
        return result
        
    except Exception as e:
        result["error"] = f"Processing failed: {str(e)}"
        result["description"] = "Not able to describe - processing failed"
        result["description_status"] = "failed"
        logger.error(f"✗ Failed to process image {img_index} on page {page_num+1}: {e}")
        return result

def process_pdf(pdf_data):
    """Core PDF processing logic with comprehensive tracking"""
    pdf = fitz.open(stream=pdf_data, filetype="pdf")
    
    results = []
    total_pages = len(pdf)
    
    # Tracking stats
    stats = {
        "total_images": 0,
        "successful_uploads": 0,
        "failed_uploads": 0,
        "successful_descriptions": 0,
        "failed_descriptions": 0,
        "processing_errors": 0
    }
    
    for page_num in range(total_pages):
        page = pdf[page_num]
        page_images = []
        
        image_list = page.get_images()
        logger.info(f"Page {page_num+1}/{total_pages}: Found {len(image_list)} images")
        
        for img_index, img in enumerate(image_list):
            stats["total_images"] += 1
            
            # Process each image independently
            image_result = process_single_image(pdf, page_num, img_index, img)
            
            # Update stats
            if image_result["upload_status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
            
            if image_result["description_status"] == "success":
                stats["successful_descriptions"] += 1
            elif image_result["description_status"] == "failed":
                stats["failed_descriptions"] += 1
            
            if image_result["error"]:
                stats["processing_errors"] += 1
            
            page_images.append(image_result)
            
            # Small delay between images to avoid rate limits
            time.sleep(0.5)
        
        results.append({
            "page": page_num + 1,
            "image_count": len(page_images),
            "images": page_images
        })
    
    pdf.close()
    
    return results, total_pages, stats

@app.route('/extract-images', methods=['POST'])
def extract_images():
    """Main endpoint: Extract images from PDF file upload"""
    try:
        # Get PDF file
        pdf_file = request.files.get('pdf')
        if not pdf_file:
            return jsonify({"error": "No PDF file provided"}), 400
        
        logger.info(f"Processing uploaded PDF: {pdf_file.filename}")
        pdf_data = pdf_file.read()
        
        # Process PDF
        results, total_pages, stats = process_pdf(pdf_data)
        
        logger.info(f"Processing complete. Stats: {stats}")
        
        return jsonify({
            "success": True,
            "total_pages": total_pages,
            "statistics": stats,
            "pages": results
        })
    
    except Exception as e:
        logger.error(f"Main error in extract_images: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/extract-from-url', methods=['POST'])
def extract_from_url():
    """Extract images from PDF URL"""
    try:
        data = request.get_json()
        pdf_url = data.get('pdf_url')
        
        if not pdf_url:
            return jsonify({"error": "No PDF URL provided"}), 400
        
        logger.info(f"Downloading PDF from: {pdf_url}")
        
        # Download PDF from URL with timeout
        response = requests.get(pdf_url, timeout=120)
        response.raise_for_status()
        pdf_data = response.content
        
        logger.info(f"PDF downloaded: {len(pdf_data)} bytes")
        
        # Process PDF
        results, total_pages, stats = process_pdf(pdf_data)
        
        logger.info(f"Processing complete. Stats: {stats}")
        
        return jsonify({
            "success": True,
            "total_pages": total_pages,
            "statistics": stats,
            "pages": results
        })
    
    except Exception as e:
        logger.error(f"Main error in extract_from_url: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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