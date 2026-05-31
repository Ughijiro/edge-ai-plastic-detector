# VERSIUNE SOFTWARE ONLY - fara hardware

LARGE_THRESHOLD = 0.15

def decide_action(detection):
    label = detection["label"]
    area_ratio = detection.get("area_ratio", 0.0)

    if area_ratio > LARGE_THRESHOLD:
        print(f"⚠️  WARNING: Gunoi MARE detectat ({label}) - size {area_ratio:.1%} - nu poate fi colectat!")
        return "WARNING"
    else:
        print(f"✅ COLLECT: Gunoi mic detectat ({label}) - size {area_ratio:.1%} - colector se activează")
        print("   (astept 2 secunde ca sa se deschida mecanismul...)")
        return "COLLECT"