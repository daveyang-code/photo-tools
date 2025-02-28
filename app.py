from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)
import os
import uuid
import zipfile
import numpy as np
from werkzeug.utils import secure_filename
import cv2


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULT_FOLDER"] = "results"
app.config["FINAL_FOLDER"] = "final_selections"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Create directories if they don't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
os.makedirs(app.config["FINAL_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def get_saliency_map(image):
    """Compute saliency map using OpenCV's Static Saliency Spectral Residual"""
    # Convert image to BGR if it's not already
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)

    # Ensure saliency map has same dimensions as original image
    if success:
        saliency_map = (saliency_map * 255).astype(np.uint8)
        if saliency_map.shape[:2] != image.shape[:2]:
            saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
        return saliency_map
    return None


def get_intersection_points(height, width, method):
    """Get intersection points based on the method (thirds, golden, or all)."""
    x_points = []
    y_points = []

    # Rule of thirds points
    if method == "thirds" or method == "all":
        x_points.extend([width // 3, 2 * width // 3])
        y_points.extend([height // 3, 2 * height // 3])

    # Golden ratio points
    if method == "golden" or method == "all":
        x_points.extend([int(width * 0.382), int(width * 0.618)])
        y_points.extend([int(height * 0.382), int(height * 0.618)])

    # Bisecting points (always included)
    x_points.append(width // 2)
    y_points.append(height // 2)

    # Remove duplicates (if any)
    x_points = list(set(x_points))
    y_points = list(set(y_points))

    return x_points, y_points


def evaluate_crop(saliency_map, x1, y1, x2, y2, x_points, y_points):
    """Evaluate the saliency of a crop region around the intersection points and along the lines."""
    crop = saliency_map[y1:y2, x1:x2]
    crop_height, crop_width = crop.shape
    total_saliency = 0

    # Evaluate saliency at intersection points
    for x in x_points:
        for y in y_points:
            x_crop = int(x * (crop_width / saliency_map.shape[1]))
            y_crop = int(y * (crop_height / saliency_map.shape[0]))
            if 0 <= x_crop < crop_width and 0 <= y_crop < crop_height:
                total_saliency += crop[y_crop, x_crop]

    # Evaluate saliency along vertical lines
    for x in x_points:
        x_crop = int(x * (crop_width / saliency_map.shape[1]))
        if 0 <= x_crop < crop_width:
            total_saliency += np.mean(crop[:, x_crop])

    # Evaluate saliency along horizontal lines
    for y in y_points:
        y_crop = int(y * (crop_height / saliency_map.shape[0]))
        if 0 <= y_crop < crop_height:
            total_saliency += np.mean(crop[y_crop, :])

    return total_saliency


def crop_image(image, saliency_map, method="all"):
    if saliency_map is None:
        return image

    height, width = image.shape[:2]
    aspect_ratio = width / height

    # Apply Gaussian blur to saliency map to reduce noise
    saliency_blur = cv2.GaussianBlur(saliency_map, (5, 5), 0)

    # Get intersection points
    x_points, y_points = get_intersection_points(height, width, method)

    # Initialize variables to track the best crop
    best_crop = None
    max_saliency = -1

    # Iterate over possible starting points (intersections)
    for x_start in x_points:
        for y_start in y_points:
            # Initialize crop dimensions
            crop_w = min(2 * x_start, 2 * (width - x_start))
            crop_h = int(crop_w / aspect_ratio)

            # Ensure the crop fits within the image
            if crop_h > height:
                crop_h = height
                crop_w = int(crop_h * aspect_ratio)

            # Grow the crop while maintaining aspect ratio
            step_size = 10  # Pixels to grow in each step
            while True:
                # Calculate crop boundaries
                x1 = max(0, x_start - crop_w // 2)
                y1 = max(0, y_start - crop_h // 2)
                x2 = min(width, x_start + crop_w // 2)
                y2 = min(height, y_start + crop_h // 2)

                # Evaluate the crop
                saliency_score = evaluate_crop(
                    saliency_blur, x1, y1, x2, y2, x_points, y_points
                )

                # Update best crop if this one is better
                if saliency_score > max_saliency:
                    max_saliency = saliency_score
                    best_crop = (x1, y1, x2, y2)

                # Stop growing if the crop reaches the image boundaries
                if x1 == 0 and x2 == width and y1 == 0 and y2 == height:
                    break

                # Grow the crop
                crop_w += step_size
                crop_h = int(crop_w / aspect_ratio)

    # Apply the best crop
    if best_crop:
        x1, y1, x2, y2 = best_crop
        return image[y1:y2, x1:x2]
    return image


def process_image(file_path):
    # Read the image
    image = cv2.imread(file_path)

    # Get saliency map
    saliency_map = get_saliency_map(image)

    # Apply different cropping techniques
    thirds_crop = crop_image(image, saliency_map, "thirds")
    golden_crop = crop_image(image, saliency_map, "golden")

    # Generate unique filenames for results
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    thirds_filename = f"{name}_thirds{ext}"
    golden_filename = f"{name}_golden{ext}"

    # Save cropped images
    thirds_path = os.path.join(app.config["RESULT_FOLDER"], thirds_filename)
    golden_path = os.path.join(app.config["RESULT_FOLDER"], golden_filename)

    cv2.imwrite(thirds_path, thirds_crop)
    cv2.imwrite(golden_path, golden_crop)

    return {
        "original": filename,
        "rule_of_thirds": thirds_filename,
        "golden_ratio": golden_filename,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files[]")
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(file_path)

            # Process the image and get results
            result = process_image(file_path)
            result["id"] = str(uuid.uuid4())  # Add a unique ID for selection tracking
            results.append(result)

    return jsonify({"results": results})


@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


@app.route("/uploads/<filename>")
def get_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/save-selections", methods=["POST"])
def save_selections():
    selections = request.json.get("selections", [])

    if not selections:
        return jsonify({"success": False, "message": "No images selected"}), 400

    # Create a zip file to hold all selected images
    zip_filename = f"{uuid.uuid4()}.zip"
    zip_path = os.path.join(app.config["FINAL_FOLDER"], zip_filename)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for selection in selections:
            filename = selection.get("filename")

            # Get paths
            crop_path = os.path.join(app.config["RESULT_FOLDER"], filename)

            # Add individual files to the zip
            zipf.write(crop_path, arcname=f"crop_{filename}")

    # After creating the zip, return the path to download
    return jsonify(
        {
            "success": True,
            "message": f"Packaged {len(selections)} selections",
            "download_url": f"/download/{zip_filename}",
        }
    )


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["FINAL_FOLDER"], filename, as_attachment=True)


@app.route("/cleanup", methods=["POST"])
def cleanup():
    # Clean up uploaded and result files after download
    try:
        # Remove files from upload folder
        for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Remove files from results folder
        for filename in os.listdir(app.config["RESULT_FOLDER"]):
            file_path = os.path.join(app.config["RESULT_FOLDER"], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Remove files from final selections folder
        # download_filename = request.json.get("filename")
        for filename in os.listdir(app.config["FINAL_FOLDER"]):
            # if filename != download_filename:
            file_path = os.path.join(app.config["FINAL_FOLDER"], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        return jsonify({"success": True, "message": "Cleanup completed successfully"})
    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error during cleanup: {str(e)}"}),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
