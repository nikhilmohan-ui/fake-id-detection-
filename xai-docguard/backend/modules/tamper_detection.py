import cv2
import numpy as np

def error_level_analysis(img, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    ela = cv2.absdiff(img, compressed)
    ela = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela = cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX)
    ela = cv2.convertScaleAbs(ela, alpha=3.0)
    return ela

def detect_noise_inconsistency(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blur)

    kernel = np.ones((15, 15), np.float32) / 225
    local_mean = cv2.filter2D(noise.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((noise.astype(np.float32)**2), -1, kernel)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    return local_std

def analyze_tampering(img):
    reasons = []
    score = 0

    # 1. Error Level Analysis
    ela = error_level_analysis(img)
    ela_mean = np.mean(ela)
    ela_std = np.std(ela)

    if ela_mean > 25 or ela_std > 30:
        score += 35
        reasons.append(f"High ELA residual (mean={ela_mean:.1f}) → possible digital edit / photo replacement")

    # 2. Noise inconsistency
    noise_map = detect_noise_inconsistency(img)
    noise_std = np.std(noise_map)
    if noise_std > 12:
        score += 25
        reasons.append("Abnormal noise pattern detected → possible region splicing")

    # 3. Edge density
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.02 or edge_density > 0.18:
        score += 15
        reasons.append("Unusual edge density → possible text manipulation")

    # 4. Color channel correlation
    b, g, r = cv2.split(img)
    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0, 1]
    if corr_rg < 0.85:
        score += 15
        reasons.append("Low color channel correlation → possible color manipulation")

    score = min(score, 100)

    # Heatmap
    heatmap = cv2.addWeighted(
        cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 0.6,
        cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 0.4, 0
    )
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return {
        "tamper_score": int(score),
        "reasons": reasons,
        "heatmap": heatmap_color,
        "ela_image": ela
    }