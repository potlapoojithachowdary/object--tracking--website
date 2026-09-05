from flask import Flask, render_template, request, send_from_directory
import cv2
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/track", methods=["POST"])
def track():
    if "video" not in request.files:
        return "No video selected"

    video = request.files["video"]

    if video.filename == "":
        return "Please select a video"

    # Save uploaded video
    input_name = str(uuid.uuid4()) + ".mp4"
    input_path = os.path.join(UPLOAD_FOLDER, input_name)
    video.save(input_path)

    # Open video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return "Unable to open video"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    output_name = str(uuid.uuid4()) + ".mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # Background subtractor
    background = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect moving objects
        mask = background.apply(frame)

        # Remove shadows and noise
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find objects
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        largest = None
        largest_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 500 and area > largest_area:
                largest_area = area
                largest = contour

        # Draw tracking box
        if largest is not None:
            x, y, w, h = cv2.boundingRect(largest)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                "Object Tracked",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        out.write(frame)

    cap.release()
    out.release()

    return render_template(
        "index.html",
        result=output_name
    )


@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
