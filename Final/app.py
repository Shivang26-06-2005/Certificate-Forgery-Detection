from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
import os
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import tempfile
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# Model Configuration
# ----------------------------
DETECTION_MODEL_PATH = r"D:\Certificate-Detection\Second Dataset\outputs\certificate_retrain2\weights\best.pt"
FORGERY_MODEL_PATH = r"D:\Certificate-Detection\First Dataset\certificate_forgery\weights\best.pt"

# Thresholds
IOU_THRESHOLD = 0.3
FORG_CONF_THRESHOLD = 0.5

# Global variables for models
detection_model = None
forgery_model = None

def load_models():
    """Load YOLO models at startup"""
    global detection_model, forgery_model
    try:
        logger.info("Loading detection model...")
        detection_model = YOLO(DETECTION_MODEL_PATH)
        logger.info("Detection model loaded successfully")
        
        logger.info("Loading forgery model...")
        forgery_model = YOLO(FORGERY_MODEL_PATH)
        logger.info("Forgery model loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) of two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def process_certificate(image_path):
    """Process certificate using the same logic as Combined Model.py"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Cannot read image file"}
        
        annotated = img.copy()
        certificate_fake = False
        detected_elements = []
        fake_regions = []
        
        # Step 1: Run forgery model on whole image
        logger.info("Running forgery detection...")
        forg_results = forgery_model(img)
        forg_boxes = forg_results[0].boxes
        forg_labels = forg_results[0].names
        
        if forg_boxes is not None and len(forg_boxes) > 0:
            for i, f_box in enumerate(forg_boxes):
                f_cls = int(f_box.cls)
                f_label = forg_labels.get(f_cls, "")
                conf = float(f_box.conf) if hasattr(f_box, "conf") else 1.0
                
                if f_label == "fake" and conf >= FORG_CONF_THRESHOLD:
                    x1, y1, x2, y2 = f_box.xyxy[0].tolist()
                    fake_regions.append([x1, y1, x2, y2])
                    logger.info(f"Found fake region with confidence: {conf}")
        
        # Step 2: Run detection model on whole image
        logger.info("Running element detection...")
        det_results = detection_model(img)
        det_boxes = det_results[0].boxes
        det_labels = det_results[0].names
        
        if det_boxes is not None and len(det_boxes) > 0:
            for i, det_box in enumerate(det_boxes):
                x1, y1, x2, y2 = det_box.xyxy[0].tolist()
                label_id = int(det_box.cls)
                label_name = det_labels.get(label_id, "unknown")
                confidence = float(det_box.conf) if hasattr(det_box, "conf") else 1.0
                
                element_fake = False
                
                # Step 3: Check overlap with any fake region
                for f_box in fake_regions:
                    iou = compute_iou([x1, y1, x2, y2], f_box)
                    if iou >= IOU_THRESHOLD:
                        element_fake = True
                        certificate_fake = True
                        logger.info(f"Element {label_name} marked as fake due to IoU: {iou}")
                
                # Store detected element info
                detected_elements.append({
                    "element": label_name,
                    "confidence": confidence,
                    "status": "FAKE" if element_fake else "GENUINE",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                
                # Draw detection box
                color = (0, 0, 255) if element_fake else (0, 255, 0)  # Red for fake, Green for genuine
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, f"{label_name}: {'FAKE' if element_fake else 'GENUINE'}",
                           (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Step 4: Overlay fake regions for visualization
        for f_box in fake_regions:
            fx1, fy1, fx2, fy2 = map(int, f_box)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
        
        # Convert annotated image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Overall result
        overall_result = "FAKE" if certificate_fake else "GENUINE"
        
        return {
            "overall_result": overall_result,
            "detected_elements": detected_elements,
            "fake_regions_count": len(fake_regions),
            "annotated_image": img_base64,
            "confidence_summary": {
                "genuine_elements": len([e for e in detected_elements if e["status"] == "GENUINE"]),
                "fake_elements": len([e for e in detected_elements if e["status"] == "FAKE"]),
                "total_elements": len(detected_elements)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing certificate: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_certificate():
    """Handle certificate upload and processing"""
    try:
        if 'certificate' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['certificate']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            try:
                # Process the certificate
                result = process_certificate(temp_path)
                
                # Clean up temp file
                os.remove(temp_path)
                
                if "error" in result:
                    return jsonify(result), 500
                
                return jsonify(result)
                
            except Exception as e:
                # Clean up temp file in case of error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        
        return jsonify({"error": "Invalid file format"}), 400
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    models_loaded = detection_model is not None and forgery_model is not None
    return jsonify({
        "status": "healthy" if models_loaded else "models_not_loaded",
        "models_loaded": models_loaded
    })

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    logger.info("Starting Certificate Detection Service...")
    
    # Load models before starting the server
    if load_models():
        logger.info("Models loaded successfully. Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("Failed to load models. Please check model paths.")
        print("❌ Failed to load models. Please check the model paths in the configuration.")