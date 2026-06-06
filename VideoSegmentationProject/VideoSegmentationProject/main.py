import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. PROJE KLASÖRÜ
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(BASE_DIR, "data")
output_dir = os.path.join(BASE_DIR, "outputs")
report_dir = os.path.join(BASE_DIR, "report_images")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

video_path = os.path.join(data_dir, "projectdata.mp4")

output_video_path = os.path.join(output_dir, "insan_yogunlugu.avi")
graph_path = os.path.join(report_dir, "insan_grafik.png")

print("VIDEO:", video_path)

# =========================
# 2. VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Video açılamadı!")

# =========================
# 3. FPS
# =========================
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = 25

# =========================
# 4. BACKGROUND MODEL
# =========================
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# =========================
# 5. İLK FRAME
# =========================
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Frame okunamadı!")

# 🔥 ROTATE FIX
frame = cv2.transpose(frame)
frame = cv2.flip(frame, 1)

height, width = frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# =========================
# 6. WRITER
# =========================
writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)

# =========================
# 7. VERİ
# =========================
counts = []
frames = []
i = 0

# =========================
# 8. LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 ROTATE FIX
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (width, height))

    # =========================
    # FRAME KAYDETME (REPORT_IMAGES DOLACAK)
    # =========================
    frame_path = os.path.join(report_dir, f"frame_{i}.jpg")
    cv2.imwrite(frame_path, frame)

    # =========================
    # SEGMENTATION
    # =========================
    mask = fgbg.apply(frame)

    _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_count = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area > 150:
            x, y, w, h = cv2.boundingRect(c)
            person_count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # =========================
    # YOĞUNLUK
    # =========================
    if person_count < 3:
        text = "DUSUK YOGUNLUK"
        color = (0,255,0)
    elif person_count < 8:
        text = "ORTA YOGUNLUK"
        color = (0,255,255)
    else:
        text = "YUKSEK YOGUNLUK"
        color = (0,0,255)

    cv2.putText(frame, f"People: {person_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, text, (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    writer.write(frame)

    counts.append(person_count)
    frames.append(i)
    i += 1

    cv2.imshow("Insan Yogunlugu", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
writer.release()
cv2.destroyAllWindows()

# =========================
# GRAFİK
# =========================
plt.figure(figsize=(10,5))
plt.plot(frames, counts)
plt.title("Insan Yogunlugu Analizi")
plt.xlabel("Frame")
plt.ylabel("Kisi Sayisi")
plt.grid(True)

plt.tight_layout()
plt.savefig(graph_path, dpi=300)
plt.close()

# =========================
# SONUÇ
# =========================
avg = np.mean(counts)

if avg < 3:
    result = "DUSUK YOGUNLUK"
elif avg < 8:
    result = "ORTA YOGUNLUK"
else:
    result = "YUKSEK YOGUNLUK"

print("\nAVG:", avg)
print("RESULT:", result)
print("VIDEO:", output_video_path)
print("GRAPH:", graph_path)
print("FRAME COUNT:", i)