# Standard library imports
import os
import sys
import tempfile
import shutil
import datetime
import time
import subprocess
import platform
import uuid
import random
import string
import json
import signal
import atexit
import re
import io
import threading
from math import pi, atan2, asin
from glob import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
import xml.dom.minidom as minidom

# Third-party imports
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to file storing active process states
ACTIVE_PROCESSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'active_processes.json')

# Dictionary to store active processes
active_processes = {}

# Define important paths
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
WEBTOOLS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHANMENGOC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phanmengoc')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configuration for image processing
CUBE_SIZE = 1920
MAX_WORKERS = min(cpu_count(), 8)  # Limit workers to avoid RAM overload

# Face transformation parameters
face_params = {
    'pano_f': lambda x, y, z: ( x,  y,  z),    # front
    'pano_r': lambda x, y, z: ( z,  y, -x),    # right
    'pano_b': lambda x, y, z: (-x,  y, -z),    # back
    'pano_l': lambda x, y, z: (-z,  y,  x),    # left
    'pano_d': lambda x, y, z: ( x,  z, -y),    # up
    'pano_u': lambda x, y, z: ( x, -z,  y),    # down
}

# Image processing functions from toolfunny.py
def resize_panorama_fast(image, target_width=6144, target_height=3072):
    """
    Resize panorama with higher ratio to improve quality
    """
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def vector_to_spherical(x, y, z):
    theta = atan2(x, z)  # longitude
    phi = asin(y)        # latitude
    return theta, phi

def create_cube_face_optimized(pano_img, face, size):
    """
    Optimize cube face creation using numpy vectorization
    """
    h, w = pano_img.shape[:2]
    
    # Create grid coordinates once
    y_coords, x_coords = np.mgrid[0:size, 0:size]
    
    # Normalize coordinates
    nx = (2 * x_coords / size) - 1
    ny = (2 * y_coords / size) - 1
    
    # Calculate vectors
    length = np.sqrt(nx*nx + ny*ny + 1)
    
    # Apply face transformation
    face_func = face_params[face]
    vx, vy, vz = face_func(nx, ny, np.ones_like(nx))
    
    # Normalize vectors
    vx = vx / length
    vy = vy / length
    vz = vz / length
    
    # Convert to spherical coordinates
    theta = np.arctan2(vx, vz)
    phi = np.arcsin(vy)
    
    # Convert to image coordinates
    uf = 0.5 * (theta / pi + 1)
    vf = 0.5 * (phi / (pi/2) + 1)
    
    px = (uf * (w - 1)).astype(np.int32)
    py = ((1 - vf) * (h - 1)).astype(np.int32)
    
    # Clamp coordinates
    px = np.clip(px, 0, w-1)
    py = np.clip(py, 0, h-1)
    
    # Sample from panorama
    face_img = pano_img[py, px]
    
    return face_img

def create_cube_face_batch(args):
    """
    Wrapper function for multiprocessing
    """
    pano_img, face, size = args
    return face, create_cube_face_optimized(pano_img, face, size)

def rotate_image(img, angle):
    if angle == 0:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def correct_rotation(face, img):
    rotation_angles = {
        'pano_f': 0,
        'pano_l': 0,
        'pano_r': 0,
        'pano_u': 0,
        'pano_d': 0,
        'pano_b': 0
    }
    angle = rotation_angles.get(face, 0)
    return rotate_image(img, angle)

def create_preview_image_fast(faces_dict, output_folder, preview_size=(256, 1536)):
    """
    Create faster preview with threading and higher quality
    """
    width, height = preview_size
    face_height = height // 6
    face_width = width

    preview_img = np.zeros((height, width, 3), dtype=np.uint8)
    order = ['pano_r', 'pano_f', 'pano_l', 'pano_b', 'pano_u', 'pano_d']

    def resize_face(i, face):
        img = faces_dict.get(face)
        if img is not None:
            resized = cv2.resize(img, (face_width, face_height), interpolation=cv2.INTER_LANCZOS4)
            preview_img[i*face_height:(i+1)*face_height, 0:face_width] = resized

    # Use threading for parallel resizing
    threads = []
    for i, face in enumerate(order):
        thread = threading.Thread(target=resize_face, args=(i, face))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    preview_path = os.path.join(output_folder, "preview.jpg")
    cv2.imwrite(preview_path, preview_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return preview_path

def create_thumbnail_fast(face_img, output_folder, size=(360, 360)):
    thumb_img = cv2.resize(face_img, size, interpolation=cv2.INTER_LANCZOS4)
    thumb_path = os.path.join(output_folder, "thumb.jpg")
    cv2.imwrite(thumb_path, thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return thumb_path

def convert_spherical_to_cube_optimized(input_path, output_folder, size=CUBE_SIZE):
    """
    Optimized version with multiprocessing for cube faces
    """
    pano_img = cv2.imread(input_path)
    if pano_img is None:
        print(f"❌ Could not read image {input_path}")
        return False

    # Keep original resolution for best quality
    # Only resize if image is too large and may cause memory errors
    h, w = pano_img.shape[:2]
    if w > 8192 or h > 4096:
        pano_img = resize_panorama_fast(pano_img, 6144, 3072)
    
    # Rotate pano 180 degrees
    pano_img = cv2.rotate(pano_img, cv2.ROTATE_180)

    os.makedirs(output_folder, exist_ok=True)

    # Use multiprocessing to create cube faces
    with ProcessPoolExecutor(max_workers=min(6, MAX_WORKERS)) as executor:
        face_args = [(pano_img, face, size) for face in face_params.keys()]
        results = list(executor.map(create_cube_face_batch, face_args))

    faces_images = {}
    
    # Save face images
    for face, face_img in results:
        face_img = correct_rotation(face, face_img)
        out_file = os.path.join(output_folder, f"{face}.jpg")
        
        # Use threading to save file
        def save_image(path, img):
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        threading.Thread(target=save_image, args=(out_file, face_img)).start()
        faces_images[face] = face_img

    # Create preview and thumbnail
    create_preview_image_fast(faces_images, output_folder)
    create_thumbnail_fast(faces_images['pano_f'], output_folder)

    return True

def process_single_image(args):
    """
    Wrapper function to process a single image in multiprocessing
    """
    input_path, panosuser_folder, size = args
    
    image_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(panosuser_folder, image_name)
    
    print(f"🛠 Processing {input_path}")
    success = convert_spherical_to_cube_optimized(input_path, output_dir, size)
    
    if success:
        return {
            'name': image_name,
            'input_path': input_path,
            'output_path': output_dir
        }
    return None

def create_krpano_xml(processed_images, output_folder, title="Virtual Tour"):
    """
    Create krpano XML file
    """
    # Create directory for panosuser directly in output_folder
    panosuser_folder = os.path.join(output_folder, "panosuser")
    os.makedirs(panosuser_folder, exist_ok=True)
    
    doc = minidom.Document()
    root = doc.createElement('krpano')
    root.setAttribute('title', title)
    doc.appendChild(root)

    version = doc.createElement('version')
    version_text = doc.createTextNode(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    version.appendChild(version_text)
    root.appendChild(version)
    
    # Absolute path to skin files
    include = doc.createElement('include')
    include.setAttribute('url', '/api/phanmengoc/skin/vtourskin.xml')
    include2 = doc.createElement('include')
    include2.setAttribute('url', '/api/phanmengoc/skin/vtourskin_design_ultra_light.xml')
    root.appendChild(include)
    root.appendChild(include2)
    
    output_folder_name = os.path.basename(output_folder)
    
    for img_info in processed_images:
        image_name = img_info['name']
        
        scene = doc.createElement('scene')
        scene.setAttribute('name',f'funny_{image_name}')
        scene.setAttribute('title', image_name)
        scene.setAttribute('onstart', '')
        # Use absolute path for thumbnail image
        scene.setAttribute('thumburl', f'/api/output/{output_folder_name}/panosuser/{image_name}/thumb.jpg')
        scene.setAttribute('lat', '')
        scene.setAttribute('lng', '')
        scene.setAttribute('alt', '')
        scene.setAttribute('heading', '')
        
        control = doc.createElement('control')
        control.setAttribute('bouncinglimits', 'calc:image.cube ? true : false')
        scene.appendChild(control)
        
        view = doc.createElement('view')
        view.setAttribute('hlookat', '0.0')
        view.setAttribute('vlookat', '0.0')
        view.setAttribute('fovtype', 'MFOV')
        view.setAttribute('fov', '120')
        view.setAttribute('maxpixelzoom', '2.0')
        view.setAttribute('fovmin', '70')
        view.setAttribute('fovmax', '140')
        view.setAttribute('limitview', 'auto')
        scene.appendChild(view)
        
        preview = doc.createElement('preview')
        # Absolute path for preview image
        preview.setAttribute('url', f'/api/output/{output_folder_name}/panosuser/{image_name}/preview.jpg')
        scene.appendChild(preview)
        
        image = doc.createElement('image')
        cube = doc.createElement('cube')
        # Absolute path for cube images
        cube.setAttribute('url', f'/api/output/{output_folder_name}/panosuser/{image_name}/pano_%s.jpg')
        image.appendChild(cube)
        scene.appendChild(image)
        
        root.appendChild(scene)
    
    root.appendChild(doc.createComment('next scene'))
    
    xml_path = os.path.join(output_folder, "user1.xml")
    with open(xml_path, 'w', encoding='utf-8') as f:
        xml_str = doc.toprettyxml(indent='\t')
        f.write(xml_str)
    
    print(f"✅ Created XML file for krpano at {xml_path}")
    return xml_path

def create_krpano_html(output_folder, title="Virtual Tour 360"):
    """
    Create krpano HTML file
    """
    # Check which capitalization of funny.js exists
    funny_js_path = '/api/phanmengoc/funny.js'  # Default lowercase
    
    # Try to find which capitalization exists
    funny_js_lower = os.path.join(PHANMENGOC_FOLDER, 'funny.js')
    funny_js_upper = os.path.join(PHANMENGOC_FOLDER, 'Funny.js')
    if os.path.exists(funny_js_upper) and not os.path.exists(funny_js_lower):
        funny_js_path = '/api/phanmengoc/Funny.js'
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
	<title>{title}</title>
	<meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
	<meta name="apple-mobile-web-app-capable" content="yes" />
	<meta name="apple-mobile-web-app-status-bar-style" content="black" />
	<meta name="mobile-web-app-capable" content="yes" />
	<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
	<meta http-equiv="x-ua-compatible" content="IE=edge" />
	<style>
		html {{ height:100%; }}
		body {{ height:100%; overflow:hidden; margin:0; padding:0; font-family:Arial, Helvetica, sans-serif; font-size:16px; color:#FFFFFF; background-color:#000000; }}
	</style>
    <style>
        .date-time {{
            position: fixed;
                z-index: 1;
                top: 30px;
                left: 30px;
                color: #FFFFFF;
                font-size: 16px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                display: flex;
                flex-direction: column;
        }}
    </style>
</head>
<body>

<script src="{funny_js_path}"></script>

<div class="date-time">
    <span id="time"> Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    <span id="scene"> Project: {title} </span>
</div>

<div id="pano" style="width:100%;height:100%;">
	<noscript><table style="width:100%;height:100%;"><tr style="vertical-align:middle;"><td><div style="text-align:center;">ERROR:<br/><br/>Javascript not activated<br/><br/></div></td></tr></table></noscript>
	<script>
		embedpano({{xml:"user1.xml", target:"pano", passQueryParameters:"startscene,startlookat"}});
	</script>
</div>

</body>
</html>
"""
    html_path = os.path.join(output_folder, "Toolstour.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Created HTML file for krpano at {html_path}")
    return html_path

def batch_convert_optimized(input_folder, output_folder, size=CUBE_SIZE, title="Virtual Tour"):
    """
    Optimized version with multiprocessing for batch processing
    """
    supported = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files = []
    for ext in supported:
        files += glob(os.path.join(input_folder, ext))
    
    # Remove duplicate files
    files = list(set(files))

    if not files:
        print("❌ No panorama images found in directory.")
        return

    print(f"📁 Found {len(files)} images. Processing with {MAX_WORKERS} workers...")

    # Create panosuser directory in output_folder
    panosuser_folder = os.path.join(output_folder, "panosuser")
    os.makedirs(panosuser_folder, exist_ok=True)

    # Use multiprocessing to process multiple images at once
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_args = [(f, panosuser_folder, size) for f in files]
        results = list(executor.map(process_single_image, process_args))

    # Filter successful results
    processed_images = [r for r in results if r is not None]
    
    # Create XML and HTML
    if processed_images:
        xml_path = create_krpano_xml(processed_images, output_folder, title)
        html_path = create_krpano_html(output_folder, f"VR360 - {title}")
        
        # Remove usertools directory if automatically created
        usertools_path = os.path.join(output_folder, "usertools")
        if os.path.exists(usertools_path):
            try:
                shutil.rmtree(usertools_path)
                print(f"✅ Removed unnecessary directory: {usertools_path}")
            except Exception as e:
                print(f"⚠️ Cannot remove directory {usertools_path}: {str(e)}")
                
        print(f"🎉 Successfully processed {len(processed_images)}/{len(files)} panorama images")
    else:
        print("⚠️ No images were processed successfully")

def load_active_processes():
    """
    Load active process information from file
    """
    global active_processes
    if os.path.exists(ACTIVE_PROCESSES_FILE):
        try:
            with open(ACTIVE_PROCESSES_FILE, 'r') as f:
                active_processes = json.load(f)
                print(f"Loaded {len(active_processes)} active processes from file")
        except Exception as e:
            print(f"Error loading processes from file: {str(e)}")
            active_processes = {}
    else:
        active_processes = {}

def save_active_processes():
    """
    Save active process information to file
    """
    try:
        with open(ACTIVE_PROCESSES_FILE, 'w') as f:
            json.dump(active_processes, f, indent=2)
            print(f"Saved {len(active_processes)} processes to file")
    except Exception as e:
        print(f"Error saving processes to file: {str(e)}")

def register_process(process_id, info):
    """
    Register a new process in the active processes list
    
    Args:
        process_id: Unique identifier for the process
        info: Dictionary containing process information
    """
    active_processes[process_id] = info
    save_active_processes()

def unregister_process(process_id):
    """
    Remove a process from the active processes list
    
    Args:
        process_id: Unique identifier for the process to remove
    """
    if process_id in active_processes:
        del active_processes[process_id]
        save_active_processes()

def cleanup_before_exit(signum=None, frame=None):
    """
    Cleanup function called before application exit
    """
    print("Saving process state before exit...")
    save_active_processes()

# Register cleanup function for exit
atexit.register(cleanup_before_exit)
signal.signal(signal.SIGTERM, cleanup_before_exit)
signal.signal(signal.SIGINT, cleanup_before_exit)

# Load active processes on startup
load_active_processes()

# Simple test route
@app.route('/')
def home():
    return "Server is running! Go to /api/check-resources to test resources."

@app.route('/test')
def test():
    return jsonify({
        "status": "success",
        "message": "API test route is working"
    })

# Route to check active processes
@app.route('/api/active-processes')
def get_active_processes():
    return jsonify({
        "active_processes": active_processes
    })

@app.route('/check-resources')
def check_resources():
    """
    Check if necessary resource files are available
    """
    try:
        # Check for funny.js with both capitalizations
        funny_js_lower = os.path.exists(os.path.join(PHANMENGOC_FOLDER, 'funny.js'))
        funny_js_upper = os.path.exists(os.path.join(PHANMENGOC_FOLDER, 'Funny.js'))
        
        resources_status = {
            'phanmengoc_folder': os.path.exists(PHANMENGOC_FOLDER),
            'funny_js': funny_js_lower or funny_js_upper,
            'skin_folder': os.path.exists(os.path.join(PHANMENGOC_FOLDER, 'skin')),
            'plugins_folder': os.path.exists(os.path.join(PHANMENGOC_FOLDER, 'plugins')),
            'vtourskin_xml': os.path.exists(os.path.join(PHANMENGOC_FOLDER, 'skin', 'vtourskin.xml')) if os.path.exists(os.path.join(PHANMENGOC_FOLDER, 'skin')) else False,
            'output_folder': os.path.exists(OUTPUT_FOLDER)
        }
        
        # Check all directories used in routes
        route_paths = {
            '/api/phanmengoc': PHANMENGOC_FOLDER,
            '/api/output': OUTPUT_FOLDER
        }
        
        missing_resources = []
        for key, exists in resources_status.items():
            if not exists:
                missing_resources.append(key)
                
        return jsonify({
            'success': len(missing_resources) == 0,
            'resources_status': resources_status,
            'missing_resources': missing_resources,
            'route_paths': {k: os.path.abspath(v) for k, v in route_paths.items()},
            'application_root': os.path.abspath(os.path.dirname(__file__))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_gdrive_folder_id(url):
    """
    Extract folder ID from Google Drive URL
    
    Args:
        url (str): Google Drive URL
        
    Returns:
        str: Folder ID if found, None otherwise
    """
    # Regex pattern for folder ID
    pattern = r'folders/([a-zA-Z0-9_-]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def get_files_from_gdrive_folder(folder_id):
    """
    Get list of files from a public Google Drive folder
    Uses a simple API that doesn't require authentication
    
    Args:
        folder_id (str): Google Drive folder ID
        
    Returns:
        list: List of file information dictionaries
    """
    try:
        # API endpoint for folder information
        api_url = f"https://www.googleapis.com/drive/v3/files"
        params = {
            'q': f"'{folder_id}' in parents and mimeType contains 'image/' and trashed=false",
            'fields': "files(id,name,mimeType,webContentLink)",
            'key': "AIzaSyCtUn8dqqbMVeW3xo21fA-YPwJ6E-kCFcE"  
        }
        
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            return response.json().get('files', [])
        return []
    except Exception as e:
        print(f"Error fetching files from Google Drive: {str(e)}")
        return []

def is_gdown_installed():
    """
    Check if gdown is installed
    
    Returns:
        bool: True if gdown is installed, False otherwise
    """
    try:
        # Try to check gdown version
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        return 'gdown' in result.stdout
    except:
        return False

def install_gdown():
    """
    Install gdown library if needed
    
    Returns:
        bool: True if installation succeeds or already installed, False if it fails
    """
    if is_gdown_installed():
        return True
        
    print("Installing gdown to download large files from Google Drive...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        return True
    except:
        print("Cannot install gdown. Please install manually: pip install gdown")
        return False

def download_with_gdown(file_id, target_path):
    """
    Download a file from Google Drive using gdown library
    
    Args:
        file_id (str): Google Drive file ID
        target_path (str): Full path to save the file to
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if not is_gdown_installed() and not install_gdown():
        print("Cannot use gdown because it's not installed.")
        return False
        
    print(f"Using gdown to download large file (ID: {file_id})...")
    try:
        import gdown
        # Create directory for the file if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Use gdown to download the file
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, target_path, quiet=False)
        
        # Check if the file exists and has size > 0
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"Successfully downloaded file using gdown. Size: {os.path.getsize(target_path)/(1024*1024):.2f} MB")
            return True
        else:
            print("Failed to download file using gdown (file empty or doesn't exist)")
            return False
    except Exception as e:
        print(f"Error using gdown: {str(e)}")
        return False

def download_file_from_gdrive(file_id, filename, target_path):
    """
    Download file from Google Drive by file ID, supporting files larger than 25MB
    Uses block-based downloading to handle large files and save memory
    Includes retry mechanism for connection failures
    Uses gdown as a fallback for large files
    
    Args:
        file_id (str): Google Drive file ID
        filename (str): Filename to save as
        target_path (str): Full path to save the file to
        
    Returns:
        bool: True if download successful, False otherwise
    """
    max_retries = 3  # Maximum retry attempts
    timeout = 300  # Response timeout (5 minutes)
    chunk_size = 1024 * 1024  # Download block size: 1MB
    
    # Quick check of file size - use gdown immediately if too large
    if _should_use_gdown_directly(file_id, filename):
        return download_with_gdown(file_id, target_path)
    
    for retry in range(max_retries):
        try:
            # Use session to maintain cookies
            session = requests.Session()
            
            # Direct download URL
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            
            # Initial request with timeout
            response = session.get(download_url, stream=True, timeout=timeout)
            
            # Check for confirmation page (large file)
            confirm_code = _get_confirmation_code(response)
            if confirm_code:
                # Make download request with confirmation code
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_code}"
                print(f"Large file detected, using confirmation code for {filename}")
                response = session.get(download_url, stream=True, timeout=timeout)
            
            # Print file size information
            _print_file_size_info(response, filename)
            
            # Check response
            if response.status_code == 200:
                # If response is HTML and no Content-Disposition header, might be an error page
                if _is_html_error_page(response):
                    print(f"Warning: Received HTML response for {filename}. File may be too large or inaccessible.")
                    
                    # If this is the last regular attempt, switch to gdown
                    if retry == max_retries - 1:
                        print(f"Trying alternative method (gdown) for {filename}")
                        return download_with_gdown(file_id, target_path)
                    continue
                
                # Create directory for file if it doesn't exist
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Download in small blocks to save memory
                if _download_in_chunks(response, target_path, chunk_size):
                    print(f"Successfully downloaded {filename} to {target_path}")
                    return True
            else:
                print(f"Cannot download file with status code: {response.status_code}")
                # Retry if not last attempt
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # If maximum attempts with default method reached, switch to gdown
                    print(f"Trying alternative method (gdown) for {filename}")
                    return download_with_gdown(file_id, target_path)
                
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            # Handle connection and timeout errors
            print(f"Connection error when downloading {filename}: {str(e)}")
            
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Cannot download after {max_retries} attempts, switching to gdown")
                return download_with_gdown(file_id, target_path)
                
        except Exception as e:
            print(f"Unidentified error when downloading from Google Drive: {str(e)}")
            # If this is the last regular attempt, switch to gdown
            if retry == max_retries - 1:
                print(f"Trying alternative method (gdown) for {filename}")
                return download_with_gdown(file_id, target_path)
    
    # If all methods failed
    return False

def _should_use_gdown_directly(file_id, filename):
    """
    Check if we should use gdown immediately based on file size or name
    
    Args:
        file_id (str): Google Drive file ID
        filename (str): Filename to download
        
    Returns:
        bool: True if gdown should be used directly
    """
    # Special handling for known large file pattern
    if "TONG THE NIGHT" in filename:
        print(f"Detected special large file: {filename}, using gdown")
        return True
        
    # Check file size
    try:
        session = requests.Session()
        response = session.head(f"https://drive.google.com/uc?id={file_id}&export=download", timeout=30)
        
        # If Content-Length header exists and file > 100MB, use gdown
        if 'Content-Length' in response.headers:
            file_size = int(response.headers['Content-Length'])
            file_size_mb = file_size / (1024 * 1024)
            if file_size_mb > 100:
                print(f"Large file detected ({file_size_mb:.1f} MB), using gdown")
                return True
    except:
        # If can't check size, continue with normal method
        pass
        
    return False

def _get_confirmation_code(response):
    """
    Extract confirmation code for large files from response cookies
    
    Args:
        response: Response object from initial request
        
    Returns:
        str: Confirmation code if found, None otherwise
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _is_html_error_page(response):
    """
    Check if the response is an HTML error page
    
    Args:
        response: Response object to check
        
    Returns:
        bool: True if response is an HTML error page
    """
    content_type = response.headers.get('Content-Type', '')
    return 'text/html' in content_type and 'Content-Disposition' not in response.headers

def _print_file_size_info(response, filename):
    """
    Print information about file size
    
    Args:
        response: Response object containing file headers
        filename: Name of the file being downloaded
    """
    if 'Content-Length' in response.headers:
        file_size = int(response.headers['Content-Length'])
        file_size_mb = file_size / (1024 * 1024)
        print(f"Downloading {filename} from Google Drive. Size: {file_size_mb:.2f} MB")
    else:
        print(f"Downloading {filename} from Google Drive. Size: unknown")

def _download_in_chunks(response, target_path, chunk_size):
    """
    Download file in chunks and show progress
    
    Args:
        response: Response object with the file data
        target_path: Path to save the file to
        chunk_size: Size of each chunk in bytes
        
    Returns:
        bool: True if download successful
    """
    try:
        total_downloaded = 0
        start_time = time.time()
        last_update = start_time
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Ensure not empty chunk
                    f.write(chunk)
                    total_downloaded += len(chunk)
                    
                    # Show progress every 3 seconds
                    current_time = time.time()
                    if current_time - last_update > 3:
                        _show_download_progress(response, total_downloaded, current_time, start_time)
                        last_update = current_time
        
        return True
    except Exception as e:
        print(f"Error downloading in chunks: {str(e)}")
        return False

def _show_download_progress(response, total_downloaded, current_time, start_time):
    """
    Show download progress
    
    Args:
        response: Response object with the file data
        total_downloaded: Number of bytes downloaded so far
        current_time: Current timestamp
        start_time: Download start timestamp
    """
    elapsed = current_time - start_time
    if 'Content-Length' in response.headers:
        file_size = int(response.headers['Content-Length'])
        progress = total_downloaded / file_size * 100
        speed = total_downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
        print(f"Downloaded {total_downloaded/(1024*1024):.1f}MB / {file_size/(1024*1024):.1f}MB ({progress:.1f}%) - {speed:.2f} MB/s")
    else:
        print(f"Downloaded {total_downloaded/(1024*1024):.1f}MB")

def generate_unique_project_folder(base_name):
    """
    Generate a unique project folder name by appending a random identifier
    
    Args:
        base_name (str): Base project name provided by user
        
    Returns:
        tuple: (unique_folder_name, display_name)
    """
    # Sanitize the base name to be safe for file systems
    safe_base_name = secure_filename(base_name) if base_name else "project"
    
    # Generate a random string (6 characters)
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    # Generate timestamp (YYMMDDHHmmss format)
    timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    
    # Create unique folder name: sanitized_name_timestamp_random
    unique_folder_name = f"{safe_base_name}_{timestamp}_{random_id}"
    
    # Check if the folder already exists (although highly unlikely with timestamp)
    counter = 1
    test_folder = unique_folder_name
    while os.path.exists(os.path.join(OUTPUT_FOLDER, test_folder)):
        test_folder = f"{unique_folder_name}_{counter}"
        counter += 1
    
    # Return both the unique folder name and the display name
    return test_folder, base_name

@app.route('/api/fetch-from-gdrive', methods=['POST'])
def fetch_from_gdrive():
    """
    API endpoint to fetch files from Google Drive and process them
    
    Expected JSON body:
    {
        "drive_url": "https://drive.google.com/drive/folders/YOUR_FOLDER_ID",
        "project_name": "Your Project Name"
    }
    
    Returns JSON with processing results
    """
    try:
        data = request.get_json()
        drive_url = data.get('drive_url')
        project_name = data.get('project_name')
        
        # Validate input parameters
        if not drive_url:
            return jsonify({'error': 'Missing Google Drive URL'}), 400
            
        if not project_name:
            return jsonify({'error': 'Missing project name'}), 400
            
        # Extract folder ID from URL
        folder_id = extract_gdrive_folder_id(drive_url)
        if not folder_id:
            return jsonify({'error': 'Invalid Google Drive folder URL'}), 400
            
        # Create process ID for tracking
        process_id = str(uuid.uuid4())
        
        # Generate unique folder name for this project
        unique_folder_name, display_name = generate_unique_project_folder(project_name)
        
        # Register new process
        register_process(process_id, {
            'type': 'gdrive_fetch',
            'status': 'starting',
            'started_at': datetime.datetime.now().isoformat(),
            'project_name': project_name,
            'unique_folder': unique_folder_name,
            'folder_id': folder_id,
            'drive_url': drive_url
        })
            
        # Get file list from Google Drive
        files_list = get_files_from_gdrive_folder(folder_id)
        if not files_list:
            unregister_process(process_id)
            return jsonify({'error': 'No image files found in the folder or folder is not public'}), 404
            
        # Create project directories
        project_upload_dir = os.path.join(UPLOAD_FOLDER, unique_folder_name)
        project_output_dir = os.path.join(OUTPUT_FOLDER, unique_folder_name)
        
        os.makedirs(project_upload_dir, exist_ok=True)
        os.makedirs(project_output_dir, exist_ok=True)
        
        # Create panosuser directory directly in project output dir
        panosuser_folder = os.path.join(project_output_dir, "panosuser")
        os.makedirs(panosuser_folder, exist_ok=True)
        
        # Update process status
        active_processes[process_id]['status'] = 'downloading'
        active_processes[process_id]['total_files'] = len(files_list)
        active_processes[process_id]['downloaded_files'] = 0
        save_active_processes()
        
        # Download and save files
        downloaded_files = []
        failed_files = []
        
        print(f"Starting download of {len(files_list)} files from Google Drive")
        
        for file_info in files_list:
            file_id = file_info['id']
            filename = secure_filename(file_info['name'])
            
            # Full path to save file
            file_path = os.path.join(project_upload_dir, filename)
            
            # Update process info
            active_processes[process_id]['current_file'] = filename
            save_active_processes()
            
            # Download file directly to destination path
            print(f"Downloading {filename} (ID: {file_id})")
            success = download_file_from_gdrive(file_id, filename, file_path)
            
            if success:
                downloaded_files.append(file_path)
                active_processes[process_id]['downloaded_files'] += 1
                save_active_processes()
                print(f"Successfully downloaded {filename}")
            else:
                failed_files.append(filename)
                print(f"Failed to download {filename}")
        
        if not downloaded_files:
            unregister_process(process_id)
            return jsonify({
                'error': 'Failed to download any files from Google Drive',
                'failed_files': failed_files
            }), 500
        
        # Download statistics
        print(f"Downloaded {len(downloaded_files)} files. Failed: {len(failed_files)}")
        
        # Update process status
        active_processes[process_id]['status'] = 'processing'
        active_processes[process_id]['total_processing'] = len(downloaded_files)
        active_processes[process_id]['processed_images'] = 0
        save_active_processes()
            
        # Process files using internal functions
        return process_downloaded_images(process_id, downloaded_files, project_upload_dir, 
                                         project_output_dir, panosuser_folder, 
                                         unique_folder_name, display_name, 
                                         len(files_list), failed_files)
                
    except Exception as e:
        return jsonify({'error': f'Error processing Google Drive folder: {str(e)}'}), 500
        
def process_downloaded_images(process_id, downloaded_files, project_upload_dir, 
                             project_output_dir, panosuser_folder, 
                             unique_folder_name, display_name, 
                             total_files_count, failed_downloads=None):
    """
    Process downloaded images using internal image processing functions
    
    Args:
        process_id: ID of the current process
        downloaded_files: List of downloaded file paths
        project_upload_dir: Upload directory for the project
        project_output_dir: Output directory for the project
        panosuser_folder: Panosuser directory for processing results
        unique_folder_name: Unique folder name for the project
        display_name: Display name for the project
        total_files_count: Total number of files from source
        failed_downloads: List of files that failed to download
        
    Returns:
        Flask response with processing results
    """
    processed_images = []
    processing_failed = []
    failed_downloads = failed_downloads or []
    
    for input_path in downloaded_files:
        image_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Update process info
        active_processes[process_id]['current_processing'] = image_name
        save_active_processes()
        
        # Final output path - directly in panosuser folder
        output_path = os.path.join(panosuser_folder, image_name)
        
        # Process image
        try:
            success = convert_spherical_to_cube_optimized(input_path, output_path, CUBE_SIZE)
            
            if success:
                processed_images.append({
                    'name': image_name,
                    'input_path': input_path,
                    'output_path': output_path
                })
                active_processes[process_id]['processed_images'] += 1
                save_active_processes()
                print(f"Successfully processed {image_name}")
            else:
                processing_failed.append(image_name)
                print(f"Failed to process {image_name}")
        except Exception as e:
            processing_failed.append(image_name)
            print(f"Error processing {image_name}: {str(e)}")
            
    # Create XML and HTML if any images were processed successfully
    if processed_images:
        # Update process status
        active_processes[process_id]['status'] = 'finalizing'
        save_active_processes()
        
        xml_path = create_krpano_xml(processed_images, project_output_dir, display_name)
        html_path = create_krpano_html(project_output_dir, f"{display_name}")
        
        # Clean up unnecessary directories
        cleanup_project_directories(project_upload_dir, project_output_dir)
            
        # Unregister process after completion
        unregister_process(process_id)
            
        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(processed_images)} images from Google Drive',
            'processed_count': len(processed_images),
            'total_count': total_files_count,
            'html_path': html_path,
            'project_name': unique_folder_name,  # Return the unique folder name
            'display_name': display_name,  # Return the original display name
            'failed_downloads': failed_downloads,
            'failed_processing': processing_failed
        })
    else:
        # Clean up upload directory if no successful processing
        try:
            shutil.rmtree(project_upload_dir)
            print(f"Removed upload directory: {project_upload_dir}")
        except Exception as e:
            print(f"Cannot remove upload directory {project_upload_dir}: {str(e)}")
        
        # Unregister process after completion with error
        unregister_process(process_id)
            
        return jsonify({
            'error': 'No images were processed successfully from Google Drive', 
            'failed_downloads': failed_downloads,
            'failed_processing': processing_failed
        }), 500
        
def cleanup_project_directories(upload_dir, output_dir):
    """
    Clean up temporary directories after processing
    
    Args:
        upload_dir: Upload directory to remove
        output_dir: Output directory to clean up
    """
    # Remove usertools directory if it was automatically created
    usertools_path = os.path.join(output_dir, "usertools")
    if os.path.exists(usertools_path):
        try:
            shutil.rmtree(usertools_path)
            print(f"Removed unnecessary directory: {usertools_path}")
        except Exception as e:
            print(f"Cannot remove directory {usertools_path}: {str(e)}")
    
    # Remove uploads directory after processing to save space
    try:
        shutil.rmtree(upload_dir)
        print(f"Removed upload directory: {upload_dir}")
    except Exception as e:
        print(f"Cannot remove upload directory {upload_dir}: {str(e)}")

@app.route('/api/process', methods=['POST'])
def process_images():
    """
    Endpoint to process uploaded panorama images
    
    Expected form data:
    - files[]: List of image files to process
    - projectName: Name of the project to create
    
    Returns JSON with processing results
    """
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        project_name = request.form.get('projectName', 'default_project')
        files = request.files.getlist('files[]')

        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400

        # Create process ID for tracking
        process_id = str(uuid.uuid4())

        # Generate unique folder name for this project
        unique_folder_name, display_name = generate_unique_project_folder(project_name)

        # Register new process
        register_process(process_id, {
            'type': 'direct_upload',
            'status': 'starting',
            'started_at': datetime.datetime.now().isoformat(),
            'project_name': project_name,
            'unique_folder': unique_folder_name,
            'total_files': len(files)
        })

        # Create project directories
        project_upload_dir = os.path.join(UPLOAD_FOLDER, unique_folder_name)
        project_output_dir = os.path.join(OUTPUT_FOLDER, unique_folder_name)
        
        os.makedirs(project_upload_dir, exist_ok=True)
        os.makedirs(project_output_dir, exist_ok=True)

        # Create panosuser directory directly in project output dir
        panosuser_folder = os.path.join(project_output_dir, "panosuser")
        os.makedirs(panosuser_folder, exist_ok=True)

        # Update process status
        active_processes[process_id]['status'] = 'saving_uploads'
        save_active_processes()

        # Save uploaded files
        saved_files = []
        for i, file in enumerate(files):
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(project_upload_dir, filename)
                file.save(file_path)
                saved_files.append(file_path)
                
                # Update progress
                active_processes[process_id]['current_file'] = filename
                active_processes[process_id]['uploaded_count'] = i + 1
                save_active_processes()

        if not saved_files:
            unregister_process(process_id)
            return jsonify({'error': 'Failed to save files'}), 500

        # Update process status for processing phase
        active_processes[process_id]['status'] = 'processing'
        active_processes[process_id]['total_processing'] = len(saved_files)
        active_processes[process_id]['processed_count'] = 0
        save_active_processes()

        # Process the uploaded files using the same helper function used for Google Drive
        return process_downloaded_images(
            process_id, saved_files, project_upload_dir, 
            project_output_dir, panosuser_folder,
            unique_folder_name, display_name, len(files)
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recover-process/<process_id>', methods=['POST'])
def recover_process(process_id):
    """
    Recover an interrupted process
    
    Args:
        process_id: ID of the process to recover
        
    Returns:
        JSON with recovery information
    """
    try:
        if process_id not in active_processes:
            return jsonify({'error': 'Process not found'}), 404
            
        process_info = active_processes[process_id]
        process_type = process_info.get('type')
        
        # Check if process is already completed or has errored
        if process_info.get('status') in ['completed', 'error']:
            return jsonify({'error': 'Process already completed or failed'}), 400
            
        # Try to recover process based on type
        if process_type == 'gdrive_fetch':
            # Redirect to corresponding endpoint
            return jsonify({
                'message': 'Recovery not implemented yet. Please start a new process.',
                'process_info': process_info
            })
        elif process_type == 'direct_upload':
            return jsonify({
                'message': 'Cannot recover direct upload processes. Please start a new upload.',
                'process_info': process_info
            })
        else:
            return jsonify({'error': 'Unknown process type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resume-unfinished', methods=['GET'])
def get_unfinished_processes():
    """
    Get list of unfinished processes
    
    Returns:
        JSON with unfinished processes
    """
    try:
        unfinished = {pid: info for pid, info in active_processes.items() 
                     if info.get('status') not in ['completed', 'error']}
        
        return jsonify({
            'count': len(unfinished),
            'processes': unfinished
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-processes', methods=['POST'])
def clear_processes():
    """
    Clear all tracked processes
    
    Returns:
        JSON with success status
    """
    try:
        global active_processes
        active_processes = {}
        save_active_processes()
        
        return jsonify({
            'success': True,
            'message': 'All process records cleared'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<path:project_name>', methods=['GET'])
def get_results(project_name):
    """
    Endpoint to get information about processed project
    
    Args:
        project_name: Name of the project to retrieve
        
    Returns:
        JSON with project information
    """
    project_dir = os.path.join(OUTPUT_FOLDER, project_name)
    
    if not os.path.exists(project_dir):
        return jsonify({'error': 'Project not found'}), 404
    
    html_path = os.path.join(project_dir, "Toolstour.html")
    
    if os.path.exists(html_path):
        return jsonify({
            'success': True,
            'project_name': project_name,
            'html_path': html_path
        })
    else:
        return jsonify({'error': 'Project results not found'}), 404

@app.route('/api/output/<path:filename>')
def serve_output(filename):
    """
    Serve output files
    
    Args:
        filename: Path to the file to serve
    """
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/phanmengoc/<path:filename>')
def serve_phanmengoc(filename):
    """
    Serve phanmengoc resources
    
    This function tries to find the requested file in multiple possible locations:
    1. Local phanmengoc folder in the application directory
    2. Parent directory's phanmengoc folder
    3. WEBTOOLS_ROOT/phanmengoc folder
    
    Args:
        filename: Path to the resource to serve
    """
    try:
        # Handle both capitalizations of funny.js
        if filename.lower() == 'funny.js':
            # Check both capitalizations
            for funny_name in ['funny.js', 'Funny.js']:
                # First look in the local phanmengoc folder
                local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phanmengoc', funny_name)
                if os.path.exists(local_path):
                    print(f"Serving {funny_name} from local phanmengoc")
                    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phanmengoc'), funny_name)
                
                # Then check in the project repository directory
                repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phanmengoc", funny_name)
                if os.path.exists(repo_path):
                    print(f"Serving {funny_name} from repository phanmengoc")
                    return send_from_directory(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phanmengoc"), funny_name)
                    
                # If not found in the above locations, check WEBTOOLS_ROOT
                if os.path.exists(WEBTOOLS_ROOT):
                    webtools_path = os.path.join(WEBTOOLS_ROOT, "phanmengoc", funny_name)
                    if os.path.exists(webtools_path):
                        print(f"Serving {funny_name} from WEBTOOLS_ROOT phanmengoc")
                        return send_from_directory(os.path.join(WEBTOOLS_ROOT, "phanmengoc"), funny_name)

        # Normal handling for other files
        # First look in the local phanmengoc folder
        local_phanmengoc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phanmengoc')
        if os.path.exists(os.path.join(local_phanmengoc, filename)):
            print(f"Serving {filename} from {local_phanmengoc}")
            return send_from_directory(local_phanmengoc, filename)

        # Then check in the project repository directory
        phanmengoc_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phanmengoc")
        if os.path.exists(os.path.join(phanmengoc_folder, filename)):
            print(f"Serving {filename} from {phanmengoc_folder}")
            return send_from_directory(phanmengoc_folder, filename)
            
        # If not found in the above locations, check WEBTOOLS_ROOT
        if os.path.exists(WEBTOOLS_ROOT):
            webtools_phanmengoc = os.path.join(WEBTOOLS_ROOT, "phanmengoc")
            if os.path.exists(os.path.join(webtools_phanmengoc, filename)):
                print(f"Serving {filename} from {webtools_phanmengoc}")
                return send_from_directory(webtools_phanmengoc, filename)
                
        # File not found
        print(f"File not found: {filename}")
        return f"File not found: {filename}", 404
    except Exception as e:
        print(f"Error serving {filename}: {str(e)}")
        return str(e), 500

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """
    Get list of all processed projects
    
    Returns:
        JSON with projects list
    """
    try:
        projects = []
        for project_name in os.listdir(OUTPUT_FOLDER):
            project_dir = os.path.join(OUTPUT_FOLDER, project_name)
            
            # Only process directories
            if not os.path.isdir(project_dir):
                continue
                
            # Check if project has HTML and XML files
            html_path = os.path.join(project_dir, "Toolstour.html")
            xml_path = os.path.join(project_dir, "user1.xml")
            
            if not (os.path.exists(html_path) and os.path.exists(xml_path)):
                continue
            
            # Extract display name from XML if possible
            display_name = project_name
            try:
                if os.path.exists(xml_path):
                    dom = minidom.parse(xml_path)
                    krpano = dom.getElementsByTagName('krpano')
                    if krpano and krpano[0].getAttribute('title'):
                        display_name = krpano[0].getAttribute('title')
            except Exception as e:
                print(f"Error extracting display name from XML: {str(e)}")
                
            # Find first thumbnail from panosuser
            panosuser_dir = os.path.join(project_dir, "panosuser")
            thumbnail_url = None
            scene_count = 0
            
            if os.path.exists(panosuser_dir):
                # Count number of scenes (subdirectories) in panosuser
                scene_dirs = [d for d in os.listdir(panosuser_dir) if os.path.isdir(os.path.join(panosuser_dir, d))]
                scene_count = len(scene_dirs)
                
                if scene_dirs:
                    first_scene = scene_dirs[0]
                    thumb_path = os.path.join(panosuser_dir, first_scene, "thumb.jpg")
                    
                    if os.path.exists(thumb_path):
                        # Get path relative to output folder
                        rel_path = os.path.relpath(thumb_path, OUTPUT_FOLDER)
                        thumbnail_url = f'/api/output/{rel_path}'
            
            # Add project information
            projects.append({
                'name': project_name,
                'display_name': display_name,
                'html_url': f'/api/output/{project_name}/Toolstour.html',
                'thumbnail_url': thumbnail_url,
                'scene_count': scene_count,
                'created_time': os.path.getctime(project_dir)
            })
        
        # Sort by creation time, newest first
        projects.sort(key=lambda x: x['created_time'], reverse=True)
        
        # Convert timestamp to string
        for project in projects:
            project['created_time'] = datetime.datetime.fromtimestamp(project['created_time']).strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'projects': projects
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>', methods=['DELETE'])
def delete_project(project_name):
    """
    Delete a project
    
    Args:
        project_name: Name of the project to delete
        
    Returns:
        JSON with deletion status
    """
    try:
        project_dir = os.path.join(OUTPUT_FOLDER, project_name)
        
        if not os.path.exists(project_dir):
            return jsonify({'error': 'Project does not exist'}), 404
            
        # Delete entire project directory
        shutil.rmtree(project_dir)
        
        return jsonify({
            'success': True,
            'message': f'Project {project_name} successfully deleted'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/rename', methods=['POST'])
def rename_project(project_name):
    """
    Rename a project (only changes display name in XML)
    
    Args:
        project_name: Name of the project to rename
        
    Expected JSON body:
    {
        "new_name": "New Display Name"
    }
    
    Returns:
        JSON with rename status
    """
    try:
        data = request.get_json()
        new_name = data.get('new_name')
        
        if not new_name:
            return jsonify({'error': 'New name cannot be empty'}), 400
            
        # Path to project directory
        project_path = os.path.join(OUTPUT_FOLDER, project_name)
        
        # Check if project exists
        if not os.path.exists(project_path):
            return jsonify({'error': 'Project does not exist'}), 404
            
        # Path to XML file
        xml_path = os.path.join(project_path, "user1.xml")
        html_path = os.path.join(project_path, "Toolstour.html")
        
        if not os.path.exists(xml_path):
            return jsonify({'error': 'XML file does not exist'}), 404
            
        # Update project name in XML file
        try:
            dom = minidom.parse(xml_path)
            krpano = dom.getElementsByTagName('krpano')
            if krpano:
                krpano[0].setAttribute('title', new_name)
                with open(xml_path, 'w', encoding='utf-8') as f:
                    dom.writexml(f)
                print(f"Updated display name in XML to {new_name}")
        except Exception as e:
            return jsonify({'error': f'Error updating XML: {str(e)}'}), 500
            
        # Update project name in HTML file
        if os.path.exists(html_path):
            try:
                # Read HTML file
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Find and replace title
                import re
                # Replace in title tag
                html_content = re.sub(r'<title>.*?</title>', f'<title>{new_name}</title>', html_content)
                # Replace in span id="scene"
                html_content = re.sub(r'<span id="scene">.*?</span>', f'<span id="scene"> Project: {new_name} </span>', html_content)
                
                # Write back HTML file
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
                print(f"Updated display name in HTML to {new_name}")
            except Exception as e:
                print(f"Error updating HTML: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully renamed project to {new_name}',
            'old_name': project_name,
            'new_name': new_name,
            'folder_name': project_name  # Folder name doesn't change
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-resources')
def test_resources():
    """
    Check and display important resource files
    
    Returns:
        JSON with resource information
    """
    try:
        # Check for Funny.js (try both capitalizations)
        funny_js_path_lower = os.path.join(PHANMENGOC_FOLDER, 'funny.js')
        funny_js_path_upper = os.path.join(PHANMENGOC_FOLDER, 'Funny.js')
        funny_js_exists = os.path.exists(funny_js_path_lower) or os.path.exists(funny_js_path_upper)
        funny_js_content = None
        
        if os.path.exists(funny_js_path_lower):
            funny_js_path = funny_js_path_lower
        elif os.path.exists(funny_js_path_upper):
            funny_js_path = funny_js_path_upper
        else:
            funny_js_path = funny_js_path_lower  # Default to lowercase if neither exists
        
        if funny_js_exists:
            with open(funny_js_path, 'r', encoding='utf-8', errors='ignore') as f:
                funny_js_content = f.read(500)  # Only read first 500 characters
        
        # Check projects
        projects = []
        if os.path.exists(OUTPUT_FOLDER):
            for project_name in os.listdir(OUTPUT_FOLDER):
                project_dir = os.path.join(OUTPUT_FOLDER, project_name)
                if os.path.isdir(project_dir):
                    xml_path = os.path.join(project_dir, "user1.xml")
                    html_path = os.path.join(project_dir, "Toolstour.html")
                    xml_content = None
                    if os.path.exists(xml_path):
                        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                            xml_content = f.read(500)  # Only read first 500 characters
                    
                    projects.append({
                        'name': project_name,
                        'has_xml': os.path.exists(xml_path),
                        'has_html': os.path.exists(html_path),
                        'xml_sample': xml_content,
                        'html_url': f'/api/output/{project_name}/Toolstour.html',
                        'panosuser_exists': os.path.exists(os.path.join(project_dir, "panosuser"))
                    })
        
        return jsonify({
            'funny_js': {
                'exists': funny_js_exists,
                'path': funny_js_path,
                'sample': funny_js_content
            },
            'projects': projects,
            'routes': {
                'serve_phanmengoc': '/api/phanmengoc/<path:filename>',
                'serve_output': '/api/output/<path:filename>'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Check if we're running in a production environment
    is_production = os.environ.get('PRODUCTION', 'False').lower() == 'true'
    
    if is_production:
        # Production settings - use host/port from environment or defaults
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 80))
        debug = False
        print(f"Running in production mode on {host}:{port}")
        app.run(debug=debug, host=host, port=port)
    else:
        # Development settings
        print("Running in development mode on 0.0.0.0:51333")
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 51333))) 