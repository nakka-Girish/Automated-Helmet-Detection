def draw_boxes(img, results):
    for r in results:
        if len(r.boxes.xyxy) == 0:  # Check if any boxes exist
            print("⚠️ No detections found in the image.")
            return img  # Return the original image without modifications
        
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img
