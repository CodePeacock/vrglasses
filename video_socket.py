import os
import subprocess
import threading
import time  # Import the time module

import cv2
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)


class VideoStream:
    def __init__(self, file_path):
        self.file_path = file_path
        self.temp_file_path = "temp_output.mp4"
        self.thread = threading.Thread(target=self._stream_video)
        self.thread.daemon = True
        self.thread.start()

    def frames(self):
        return self._stream_video()

    def _stream_video(self):
        while True:
            try:
                # Use ffmpeg to extract frames and create a new temporary file
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        self.file_path,
                        "-c:v",
                        "libx264",
                        "-vf",
                        "fps=15",
                        self.temp_file_path,
                    ]
                )

                # Wait for the file to be fully created before proceeding
                while not os.path.exists(self.temp_file_path):
                    time.sleep(0.1)

                # Read from the temporary file
                cap = cv2.VideoCapture(self.temp_file_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    _, jpeg = cv2.imencode(".jpg", frame)
                    socketio.emit(
                        "video_frame", {"image": jpeg.tobytes()}, namespace="/video"
                    )

                socketio.emit("video_complete", namespace="/video")
            except Exception as e:
                print(f"Error processing video: {e}")
            finally:
                # Remove the temporary file after processing
                try:
                    os.remove(self.temp_file_path)
                except FileNotFoundError:
                    pass  # Ignore if the file is already removed

    def release(self):
        pass


# Replace 'your_video.mp4' with the actual path to your .mp4 file
if not os.path.exists("output.mp4"):
    exit(1)
else:
    video_stream = VideoStream("output.mp4")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect", namespace="/video")
def video_connect():
    print("Client connected to video stream")


@socketio.on("disconnect", namespace="/video")
def video_disconnect():
    print("Client disconnected from video stream")


@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream.frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    socketio.run(app, debug=True)
