from typing import Any, Generator

import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)


def generate_frames() -> Generator[bytes, Any, None]:
    cap = cv2.VideoCapture(
        0
    )  # Make sure to provide the correct path to your video file

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

        cv2.waitKey(
            int(1000 / cap.get(cv2.CAP_PROP_FPS))
        )  # Introduce delay based on fps

    cap.release()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/video_feed")
def video_feed() -> Response:
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port="5000")
