import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(BASE_DIR, "data/projectdata.mp4")
output_video_path = os.path.join(BASE_DIR, "outputs/output.avi")
graph_path = os.path.join(BASE_DIR, "report_images/graph.png")
frame_dir = os.path.join(BASE_DIR, "report_images/frames")

os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
os.makedirs(os.path.dirname(graph_path), exist_ok=True)
os.makedirs(frame_dir, exist_ok=True)

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Video açılamadı")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1:
    fps = 25

fgbg = cv2.createBackgroundSubtractorMOG2(500, 50, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

ret, frame = cap.read()
frame = cv2.transpose(frame)
frame = cv2.flip(frame, 1)

h, w = frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------- WRITER ----------------
writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (w, h)
)

counts = []
frames = []
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.transpose(frame)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))

    mask = fgbg.apply(frame)
    _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_count = 0

    for c in contours:
        if cv2.contourArea(c) > 150:
            x, y, ww, hh = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            person_count += 1

    # ---------------- YOĞUNLUK ----------------
    if person_count < 3:
        text = "DUSUK YOGUNLUK"
        color = (0, 255, 0)
    elif person_count < 8:
        text = "ORTA YOGUNLUK"
        color = (0, 255, 255)
    else:
        text = "YUKSEK YOGUNLUK"
        color = (0, 0, 255)

    cv2.putText(frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # ---------------- SAVE VIDEO ----------------
    writer.write(frame)

    # ---------------- SAVE FRAME (HER 10 FRAME) ----------------
    if i % 10 == 0:
        cv2.imwrite(
            os.path.join(frame_dir, f"frame_{i}.jpg"),
            frame
        )

    counts.append(person_count)
    frames.append(i)

    # ---------------- SHOW ----------------
    cv2.imshow("Video", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    # Q = çık
    if key == ord('q'):
        break

    if key == 27:
        break

    i += 1

cap.release()
writer.release()
cv2.destroyAllWindows()

# ---------------- GRAPH ----------------
plt.figure(figsize=(10, 5))
plt.plot(frames, counts)
plt.title("Insan Yogunlugu Analizi")
plt.xlabel("Frame")
plt.ylabel("Kisi Sayisi")
plt.grid(True)
plt.tight_layout()

plt.savefig(graph_path, dpi=300)
plt.show()
plt.close()

# ---------------- RESULT ----------------
avg = np.mean(counts)

if avg < 3:
    result = "DUSUK YOGUNLUK"
elif avg < 8:
    result = "ORTA YOGUNLUK"
else:
    result = "YUKSEK YOGUNLUK"

print("\nAVG:", avg)
print("RESULT:", result)
print("GRAPH:", graph_path)
print("FRAMES SAVED TO:", frame_dir)