from flask import Flask, render_template, request, send_file, Response
from ultralytics import YOLO
import numpy as np
import cv2
import os
import threading
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

detector = YOLO("yolov8n.pt")
classifier = YOLO("material_classifier.pt")

UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

COLOR_MAP = {
    "plastic": (0, 255, 140),
    "metal": (0, 170, 255),
    "glass": (170, 0, 255),
    "paper": (255, 200, 0),
    "cloth": (220, 0, 0)
}

THEME = {
    "plastic": "#00ff95",
    "metal": "#00b7ff",
    "glass": "#b44bff",
    "paper": "#ffdd29",
    "cloth": "#ff5050"
}

MATERIAL_INFO = {
    "plastic": {
        "recyclable": True,
        "bin": "Dry / Recyclable (Blue bin)",
        "note": "PET bottles, containers, packaging material.",
        "action": "Rinse and drop in dry recyclable bin or sell as scrap.",
        "weight_factor": 0.00000030,
        "scrap_price_per_kg": 18.0
    },
    "metal": {
        "recyclable": True,
        "bin": "Scrap (Metal)",
        "note": "Utensils, cans, foil, mixed metal parts.",
        "action": "Segregate clean metal and sell to scrap dealer.",
        "weight_factor": 0.00000080,
        "scrap_price_per_kg": 45.0
    },
    "glass": {
        "recyclable": True,
        "bin": "Glass recycling bin",
        "note": "Bottles, jars, glass containers.",
        "action": "Avoid breakage, store safely and give to glass recycler.",
        "weight_factor": 0.00000095,
        "scrap_price_per_kg": 12.0
    },
    "paper": {
        "recyclable": True,
        "bin": "Paper recycling",
        "note": "Newspapers, office paper, cartons.",
        "action": "Keep dry and flat, bundle and give to raddiwala.",
        "weight_factor": 0.00000020,
        "scrap_price_per_kg": 10.0
    },
    "cloth": {
        "recyclable": True,
        "bin": "Textile recycling / Donation",
        "note": "Old clothes, sheets, fabric.",
        "action": "If usable, donate; else send to textile recycler.",
        "weight_factor": 0.00000025,
        "scrap_price_per_kg": 8.0
    },
    "other": {
        "recyclable": False,
        "bin": "Check guidelines",
        "note": "Unknown or mixed material.",
        "action": "Consult local waste guidelines, avoid mixing with recyclables.",
        "weight_factor": 0.00000025,
        "scrap_price_per_kg": 0.0
    }
}

captured_frame = None


def speak(text):
    try:
        import pyttsx3
        e = pyttsx3.init()
        e.say(text)
        e.runAndWait()
        e.stop()
    except:
        pass


def generate_charts(class_count, material_details):
    pie_path = os.path.join(UPLOAD_FOLDER, "pie_chart.png")
    bar_path = os.path.join(UPLOAD_FOLDER, "bar_chart.png")

    labels = list(class_count.keys())
    values = list(class_count.values())

    if len(labels) > 0:
        plt.figure(figsize=(4, 4))
        plt.pie(values, labels=labels, autopct="%1.0f%%")
        plt.title("Material Distribution (Count)")
        plt.tight_layout()
        plt.savefig(pie_path)
        plt.close()
    else:
        pie_path = ""

    scrap_labels = []
    scrap_values = []
    for m in material_details:
        scrap_labels.append(m["class"])
        scrap_values.append(max(0.0, float(m["est_scrap"])))

    if len(scrap_labels) > 0:
        plt.figure(figsize=(4, 3))
        plt.bar(scrap_labels, scrap_values)
        plt.ylabel("Scrap Value (₹)")
        plt.title("Estimated Scrap Value per Material")
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()
    else:
        bar_path = ""

    pie_web = pie_path if pie_path and os.path.exists(pie_path) else ""
    bar_web = bar_path if bar_path and os.path.exists(bar_path) else ""
    return pie_web, bar_web


def generate_heatmap(original_img, det_result):
    if original_img is None or len(det_result[0].boxes) == 0:
        return ""
    h, w = original_img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for b in det_result[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0]) if hasattr(b, "conf") else 1.0
        conf = max(0.1, min(conf, 1.0))
        heatmap[y1:y2, x1:x2] += conf
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.jpg")
    cv2.imwrite(heatmap_path, overlay)
    return heatmap_path


def build_stats(class_count, class_area):
    material_details = []
    totals = {
        "total_items": 0,
        "recyclable_items": 0,
        "non_recyclable_items": 0,
        "total_weight_kg": 0.0,
        "total_scrap_value": 0.0,
        "recyclable_percent": 0.0,
        "sustainability_score": 0.0,
        "co2_saved_kg": 0.0
    }
    summary_lines = []
    detail_lines = []

    for label, count in class_count.items():
        info = MATERIAL_INFO.get(label.lower(), MATERIAL_INFO["other"])
        area = class_area.get(label, 0)
        weight = area * info["weight_factor"]
        scrap = weight * info["scrap_price_per_kg"]

        totals["total_items"] += count
        totals["total_weight_kg"] += weight
        totals["total_scrap_value"] += scrap
        if info["recyclable"]:
            totals["recyclable_items"] += count
        else:
            totals["non_recyclable_items"] += count

        material_details.append({
            "class": label,
            "count": count,
            "recyclable": "Yes" if info["recyclable"] else "No",
            "bin": info["bin"],
            "note": info["note"],
            "action": info["action"],
            "est_weight": weight,
            "est_scrap": scrap
        })

        summary_lines.append(f"{label}: {count} item(s)")
        detail_lines.append(
            f"{label}: count={count}, weight={weight:.2f}kg, scrap≈₹{scrap:.1f}, bin={info['bin']}, action={info['action']}"
        )

    if totals["total_items"] > 0:
        totals["recyclable_percent"] = totals["recyclable_items"] / totals["total_items"] * 100.0

    recyclable_ratio = totals["recyclable_items"] / totals["total_items"] if totals["total_items"] > 0 else 0.0
    recyclable_weight = totals["total_weight_kg"] * recyclable_ratio
    co2_saved = recyclable_weight * 1.5
    totals["co2_saved_kg"] = co2_saved

    score = 0.0
    score += min(60.0, totals["recyclable_percent"] * 0.6)
    score += min(25.0, totals["total_weight_kg"] * 5.0)
    score += min(15.0, totals["total_scrap_value"] / 10.0)
    totals["sustainability_score"] = min(100.0, score)

    summary_lines.insert(0, f"Sustainability Score: {totals['sustainability_score']:.1f}/100")
    summary_lines.insert(1, f"Estimated CO2 Saved: {totals['co2_saved_kg']:.2f} kg")

    return material_details, totals, "|".join(summary_lines), "|".join(detail_lines)


@app.route("/")
def index():
    return render_template(
        "index.html",
        result_text=None,
        output_image=None,
        heatmap_image=None,
        pie_chart=None,
        bar_chart=None,
        table=None,
        theme_color="#00d4ff",
        material_details=None,
        totals=None,
        pdf_summary="",
        pdf_details="",
        pdf_image="",
        pdf_pie="",
        pdf_bar=""
    )


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return index()
    img_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    file.save(img_path)
    return run_detection(img_path)


def run_detection(img_path):
    img = cv2.imread(img_path)
    det = detector(img, conf=0.30)
    num_boxes = len(det[0].boxes)

    class_count, class_area = {}, {}

    if num_boxes <= 1:
        cls = classifier(img, verbose=False)
        label = cls[0].names[int(cls[0].probs.top1)]
        conf = float(cls[0].probs.top1conf)
        h, w = img.shape[:2]
        class_count[label] = 1
        class_area[label] = h * w
        mat, totals, sumry, details = build_stats(class_count, class_area)
        pie_path, bar_path = generate_charts(class_count, mat)
        heatmap_path = generate_heatmap(img, det)
        speak(label)
        return render_template(
            "index.html",
            result_text=f"{label} ({conf:.2f})",
            output_image=img_path,
            heatmap_image=heatmap_path,
            pie_chart=pie_path,
            bar_chart=bar_path,
            table=[{"class": label, "count": 1}],
            theme_color=THEME.get(label.lower(), "#00d4ff"),
            material_details=mat,
            totals=totals,
            pdf_summary=sumry,
            pdf_details=details,
            pdf_image=img_path,
            pdf_pie=pie_path,
            pdf_bar=bar_path
        )

    final_img = img.copy()
    speak_labels = []

    for b in det[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = img[y1:y2, x1:x2]
        cls = classifier(crop, verbose=False)
        label = cls[0].names[int(cls[0].probs.top1)]
        speak_labels.append(label)
        class_count[label] = class_count.get(label, 0) + 1
        area = max(1, (x2 - x1) * (y2 - y1))
        class_area[label] = class_area.get(label, 0) + area
        cv2.rectangle(final_img, (x1, y1), (x2, y2), COLOR_MAP.get(label.lower(), (0, 255, 0)), 3)
        cv2.putText(final_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(out, final_img)
    mat, totals, sumry, details = build_stats(class_count, class_area)
    pie_path, bar_path = generate_charts(class_count, mat)
    heatmap_path = generate_heatmap(img, det)
    threading.Thread(target=speak, args=(", ".join(speak_labels),), daemon=True).start()

    return render_template(
        "index.html",
        result_text=" | ".join([f"{c}: {n}" for c, n in class_count.items()]),
        output_image=out,
        heatmap_image=heatmap_path,
        pie_chart=pie_path,
        bar_chart=bar_path,
        table=[{"class": c, "count": n} for c, n in class_count.items()],
        theme_color=THEME.get(max(class_count, key=class_count.get).lower(), "#00d4ff"),
        material_details=mat,
        totals=totals,
        pdf_summary=sumry,
        pdf_details=details,
        pdf_image=out,
        pdf_pie=pie_path,
        pdf_bar=bar_path
    )


@app.route("/download", methods=["POST"])
def download_pdf():
    pdf_path = os.path.join(UPLOAD_FOLDER, "report.pdf")
    summary = request.form.get("summary", "")
    details = request.form.get("details", "")
    image = request.form.get("image_path", "")
    pie_path = request.form.get("pie_path", "")
    bar_path = request.form.get("bar_path", "")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, "AI Material Classification Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, h - 70, f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    y = h - 110
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Summary")
    y -= 20
    c.setFont("Helvetica", 11)
    for r in summary.split("|"):
        if y < 80:
            c.showPage()
            y = h - 50
            c.setFont("Helvetica", 11)
        c.drawString(60, y, "- " + r)
        y -= 16

    y -= 15
    if y < 80:
        c.showPage()
        y = h - 50

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Detailed Breakdown")
    y -= 20
    c.setFont("Helvetica", 11)
    for r in details.split("|"):
        if y < 80:
            c.showPage()
            y = h - 50
            c.setFont("Helvetica", 11)
        c.drawString(60, y, "- " + r)
        y -= 16

    if image and os.path.exists(image):
        try:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, h - 60, "Detected Objects View")
            c.drawImage(ImageReader(image), 50, h - 380, width=420, height=300)
        except:
            pass

    if pie_path and os.path.exists(pie_path):
        try:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, h - 60, "Material Distribution (Count)")
            c.drawImage(ImageReader(pie_path), 80, h - 420, width=380, height=320)
        except:
            pass

    if bar_path and os.path.exists(bar_path):
        try:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, h - 60, "Scrap Value by Material")
            c.drawImage(ImageReader(bar_path), 80, h - 380, width=380, height=280)
        except:
            pass

    c.save()
    return send_file(pdf_path, as_attachment=True, download_name="material_report.pdf")


camera_running = False
cap = None


@app.route("/camera")
def camera_page():
    return render_template("camera.html")




def gen_frames():
    global camera_running, cap, captured_frame
    cap = cv2.VideoCapture(0)
    camera_running = True
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break
        captured_frame = frame.copy()
        det = detector(frame, conf=0.40)
        for b in det[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls = classifier(frame[y1:y2, x1:x2], verbose=False)
            label = cls[0].names[int(cls[0].probs.top1)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_MAP.get(label.lower(), (0, 255, 0)), 3)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buf = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture")
def capture_image():
    global captured_frame
    if captured_frame is None:
        return "Camera not ready"
    img_path = os.path.join(UPLOAD_FOLDER, "captured.jpg")
    cv2.imwrite(img_path, captured_frame)
    return run_detection(img_path)


@app.route("/stop_camera")
def stop_camera():
    global camera_running
    camera_running = False
    return "Camera Stopped"

import qrcode
import socket

def show_qr(port=5000):
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    url = f"http://{ip}:{port}"
    print("\n🔗 Website Link:", url)

    img = qrcode.make(url)
    qr_path = os.path.join(UPLOAD_FOLDER, "qr.png")
    img.save(qr_path)
    print("📱 QR generated → static/qr.png (Scan to open on phone)")


if __name__ == "__main__":
    port = 5000
    show_qr(port)
    app.run(host="0.0.0.0", port=port)


