from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import torch
import torchvision
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load the YOLOv5 model for object detection
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
if torch.cuda.is_available():
    midas.to("cuda")

# Define transform for depth estimation
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "image_url": url_for("static", filename=f"uploads/{filename}"),
            }
        )

    return jsonify({"error": "File type not allowed"})


@app.route("/apply_blur", methods=["POST"])
def apply_blur():
    data = request.json
    filename = data.get("filename")
    blur_value = int(data.get("blur_value", 5))
    kernel_type = data.get("kernel_type", "gaussian")
    kernel_shape = data.get("kernel_shape", "square")
    blur_mode = data.get("blur_mode", "simple")

    if not filename:
        return jsonify({"error": "No filename provided"})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"})

    # Read image
    image = cv2.imread(file_path)
    original_image = image.copy()

    try:
        # Different blur modes
        if blur_mode == "simple":
            # Apply selected kernel type and shape
            blurred_image = apply_kernel_blur(
                image, kernel_type, blur_value, kernel_shape
            )
        elif blur_mode == "object_detection":
            # Apply object detection and blur background
            blurred_image = apply_object_detection_blur(
                image, kernel_type, blur_value, kernel_shape
            )
        elif blur_mode == "depth":
            # Apply depth-based blur
            blurred_image = apply_depth_based_blur(
                image, kernel_type, blur_value, kernel_shape
            )
        elif blur_mode == "combined":
            # Apply combined object detection and depth-based blur
            blurred_image = apply_combined_blur(
                image, kernel_type, blur_value, kernel_shape
            )
        else:
            blurred_image = image.copy()

        # Convert the image to base64 string
        _, buffer = cv2.imencode(".jpg", blurred_image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {"success": True, "blurred_image": f"data:image/jpeg;base64,{img_base64}"}
        )
    except Exception as e:
        return jsonify({"error": str(e)})


def create_kernel(size, shape="square"):
    """Create a custom shaped kernel for convolution."""
    # Ensure size is odd
    if size % 2 == 0:
        size += 1

    # Create empty kernel
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = center

    if shape == "square":
        # Square kernel (already handled by OpenCV's built-in functions)
        kernel.fill(1)
    elif shape == "circle":
        # Circular kernel
        y, x = np.ogrid[-center : center + 1, -center : center + 1]
        mask = x * x + y * y <= radius * radius
        kernel[mask] = 1
    elif shape == "hexagon":
        # Hexagonal kernel
        # Draw a hexagon using angles
        for i in range(size):
            for j in range(size):
                # Calculate distance from center
                y_dist = i - center
                x_dist = j - center
                dist = np.sqrt(y_dist**2 + x_dist**2)

                # Calculate angle
                angle = np.arctan2(y_dist, x_dist) % (2 * np.pi)

                # Hexagon has 6 sides, so we check if we're within radius at 6 different angle ranges
                k = np.floor(angle / (np.pi / 3))
                a1 = k * (np.pi / 3)
                a2 = a1 + (np.pi / 3)

                # r is distance to edge at this angle
                r = radius / np.cos(min(angle - a1, a2 - angle))

                if dist <= r:
                    kernel[i, j] = 1
    elif shape == "star":
        # Star-shaped kernel
        for i in range(size):
            for j in range(size):
                y_dist = i - center
                x_dist = j - center
                dist = np.sqrt(y_dist**2 + x_dist**2)

                # Calculate angle
                angle = np.arctan2(y_dist, x_dist) % (2 * np.pi)

                # Star shape: radius varies with angle
                points = 5  # 5-pointed star
                inner_radius = radius * 0.4  # Inner radius of star

                # Determine the current angle segment
                segment = (angle % (2 * np.pi / points)) / (2 * np.pi / points)

                # Calculate radius at this angle (interpolate between inner and outer)
                if segment < 0.5:
                    # Going from outer to inner
                    r = radius * (1.0 - segment * 2) + inner_radius * (segment * 2)
                else:
                    # Going from inner to outer
                    segment = segment - 0.5
                    r = inner_radius * (1.0 - segment * 2) + radius * (segment * 2)

                if dist <= r:
                    kernel[i, j] = 1
    elif shape == "diamond":
        # Diamond-shaped kernel
        for i in range(size):
            for j in range(size):
                # Manhattan distance for diamond shape
                if abs(i - center) + abs(j - center) <= radius:
                    kernel[i, j] = 1
    else:
        # Default to square
        kernel.fill(1)

    # Normalize the kernel
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()

    return kernel


def apply_custom_blur(image, kernel):
    """Apply blur using custom kernel."""
    # Apply the kernel to each channel
    b_channel = cv2.filter2D(image[:, :, 0], -1, kernel)
    g_channel = cv2.filter2D(image[:, :, 1], -1, kernel)
    r_channel = cv2.filter2D(image[:, :, 2], -1, kernel)

    # Merge channels
    return cv2.merge([b_channel, g_channel, r_channel])


def apply_kernel_blur(image, kernel_type, blur_value, kernel_shape="square"):
    """Apply blur with specified kernel type and shape."""
    # Ensure blur value is odd for some kernels
    if blur_value % 2 == 0:
        blur_value += 1

    # Size of the kernel
    kernel_size = blur_value * 2 + 1

    # For non-square shapes, we need custom kernels
    if kernel_shape != "square" and kernel_type in ["box", "gaussian"]:
        # Create custom shaped kernel
        kernel = create_kernel(kernel_size, kernel_shape)

        if kernel_type == "gaussian":
            # Apply Gaussian distribution to kernel values
            y, x = np.ogrid[-blur_value : blur_value + 1, -blur_value : blur_value + 1]
            g = np.exp(-(x * x + y * y) / (2 * (blur_value / 2) ** 2))
            kernel = kernel * g
            kernel = kernel / kernel.sum()  # Normalize

        # Apply the custom kernel
        return apply_custom_blur(image, kernel)

    # Use OpenCV's built-in functions for square kernels or when shape doesn't apply
    if kernel_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif kernel_type == "box":
        return cv2.blur(image, (kernel_size, kernel_size))
    elif kernel_type == "median":
        # Median blur doesn't work with custom shapes directly
        return cv2.medianBlur(image, kernel_size)
    elif kernel_type == "bilateral":
        # Bilateral filter doesn't work with custom shapes directly
        return cv2.bilateralFilter(image, kernel_size, blur_value * 15, blur_value * 15)
    else:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_object_detection_blur(image, kernel_type, blur_value, kernel_shape="square"):
    """Apply blur to background while keeping detected objects in focus."""

    # Debug: Ensure apply_kernel_blur is a function
    if not callable(apply_kernel_blur):
        raise TypeError(
            f"Error: apply_kernel_blur is not callable, found type {type(apply_kernel_blur)}"
        )

    # Convert BGR to RGB for YOLO
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    results = yolo_model(rgb_image)

    bounding_boxes = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        bounding_boxes.append([x1, y1, x2, y2])

    sam_predictor.set_image(image)
    input_boxes = torch.tensor(bounding_boxes, device=sam_predictor.device)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2]
    )
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Create a mask for the entire image
    full_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    # Subtract the object masks from the full mask to get the background mask
    for mask in masks:
        full_mask[mask[0].cpu().numpy()] = 0

    # Apply blur to the entire image
    blurred_image = apply_kernel_blur(image, kernel_type, blur_value, kernel_shape)

    # Combine the original image and the blurred image using the mask
    result = np.where(full_mask[:, :, None] == 0, image, blurred_image)

    return result


def apply_depth_based_blur(image, kernel_type, blur_value, kernel_shape="square"):
    """Apply blur based on estimated depth, with custom kernel shapes."""
    # Prepare image for depth estimation
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to meet model input requirements
    img_height, img_width = image.shape[:2]

    # Use the model's transform which handles resizing correctly
    input_batch = transform(rgb_image).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run depth estimation
    with torch.no_grad():
        depth_map = midas(input_batch)

        # Interpolate the prediction to match the original image dimensions
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(img_height, img_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Move to CPU and convert to numpy
    depth_map = depth_map.cpu().numpy()

    # Normalize depth map to 0-1 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # Invert the depth map so distant objects have higher blur values
    depth_map = 1.0 - depth_map

    # Apply blur based on depth
    result = np.zeros_like(image)
    blurred_images = []

    # Create a series of differently blurred images
    max_blur = blur_value
    for i in range(max_blur + 1):
        # Skip blur level 0 to avoid redundant computation
        if i == 0:
            blurred_images.append(image)
            continue

        blur_size = i * 2 + 1
        blurred_images.append(apply_kernel_blur(image, kernel_type, i, kernel_shape))

    # Normalize and scale depth to use as indices
    depth_indices = np.clip((depth_map * max_blur).astype(np.int32), 0, max_blur)

    # For each pixel, choose from the appropriate blur level
    for y in range(img_height):
        for x in range(img_width):
            blur_idx = depth_indices[y, x]
            result[y, x] = blurred_images[blur_idx][y, x]

    return result


def apply_combined_blur(image, kernel_type, blur_value, kernel_shape="square"):
    """Apply blur combining both object detection and depth information."""
    # Convert BGR to RGB for processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Step 1: Object Detection with YOLO
    results = yolo_model(rgb_image)

    # Create mask for detected objects
    bounding_boxes = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        bounding_boxes.append([x1, y1, x2, y2])

    sam_predictor.set_image(image)
    input_boxes = torch.tensor(bounding_boxes, device=sam_predictor.device)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2]
    )
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Create a mask for the entire image
    object_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Add the object masks to the full mask to get the object mask
    for mask in masks:
        object_mask[mask[0].cpu().numpy()] = 255

    # Step 2: Depth Estimation
    # Use the model's transform which handles resizing correctly
    input_batch = transform(rgb_image).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run depth estimation
    with torch.no_grad():
        depth_map = midas(input_batch)

        # Interpolate to match original dimensions
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(img_height, img_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Move to CPU and convert to numpy
    depth_map = depth_map.cpu().numpy()

    # Normalize depth map to 0-1 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # Invert depth map so distant objects get more blur
    depth_map = 1.0 - depth_map

    # Step 3: Create multiple blur levels
    blurred_images = []
    max_blur = blur_value

    for i in range(max_blur + 1):
        if i == 0:
            blurred_images.append(image)
            continue

        blurred_images.append(apply_kernel_blur(image, kernel_type, i, kernel_shape))

    # Step 4: Combine object detection and depth information
    # Map depth values to blur indices
    depth_indices = np.clip((depth_map * max_blur).astype(np.int32), 0, max_blur)

    # Initialize result with most blurred version
    result = blurred_images[max_blur].copy()

    # For each pixel, apply appropriate blur level based on depth
    for y in range(img_height):
        for x in range(img_width):
            # Skip detected objects (they'll be handled separately)
            if object_mask[y, x] > 0:
                continue

            blur_idx = depth_indices[y, x]
            result[y, x] = blurred_images[blur_idx][y, x]

    # Step 5: Copy original image content for detected objects (keep them in focus)
    result[object_mask > 0] = image[object_mask > 0]

    return result


if __name__ == "__main__":
    app.run(debug=True)
