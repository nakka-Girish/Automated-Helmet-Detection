import cv2
from ultralytics import YOLO
import os
from extraction import extract_text_from_plate
from data_fetch import process_number_plate

import torch
from ultralytics import YOLO
model = YOLO(r"C:\GIRISH_5V8\projects\Automated Helmet Detection\runs\weights\best.pt")

def is_point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def resize_to_fit(image, max_width=1200, max_height=700):
    h, w = image.shape[:2]
    scale_factor = min(max_width / w, max_height / h, 1.0)
    if scale_factor < 1.0:
        return cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    return image

def draw_label(img, text, pos, label_color, font_scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = pos
    cv2.rectangle(img, (x, y - text_h - baseline - 4), (x + text_w, y), label_color, -1)
    cv2.putText(img, text, (x, y - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def process_frame(frame, f=False):
    results = model(frame)
    predictions = results[0]
    riders = []
    without_helmet = []
    helmets = []
    number_plates = []

    for i in range(len(predictions.boxes)):
        box = predictions.boxes[i]
        coords = box.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, coords)
        cls = int(box.cls.cpu().numpy()[0])
        label = predictions.names[cls].lower()

        if label == "rider":
            riders.append([x1, y1, x2, y2])
        elif label == "without helmet":
            without_helmet.append([x1, y1, x2, y2])
        elif label == "helmet":
            helmets.append([x1, y1, x2, y2])
        elif label == "number plate":
            number_plates.append([x1, y1, x2, y2])

    nonHelmeted_riders = []
    if not f:
        for rider in riders:
            for wh in without_helmet:
                center_x = (wh[0] + wh[2]) / 2
                center_y = (wh[1] + wh[3]) / 2
                if is_point_in_box(center_x, center_y, rider):
                    nonHelmeted_riders.append(rider)
                    break
    else:
        nonHelmeted_riders = riders.copy()

    for idx, rider in enumerate(riders, start=1):
        x1, y1, x2, y2 = rider
        if rider in nonHelmeted_riders:
            rider_label = f"Rider {idx}: No Helmet"
            color = (0, 0, 255)
        else:
            rider_label = f"Rider {idx}: Helmet"
            color = (0, 255, 0)
            print(f"Rider {idx} found with Helmet")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, rider_label, (x1, y1), color)

    for helmet in helmets:
        hx1, hy1, hx2, hy2 = helmet
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        draw_label(frame, "Helmet", (hx1, hy1), (0, 255, 0))

    for wh in without_helmet:
        whx1, why1, whx2, why2 = wh
        cv2.rectangle(frame, (whx1, why1), (whx2, why2), (0, 0, 255), 2)
        draw_label(frame, "Without Helmet", (whx1, why1), (0, 0, 255))

    for plate in number_plates:
        px1, py1, px2, py2 = plate
        center_plate_x = (px1 + px2) / 2
        center_plate_y = (py1 + py2) / 2
        associated_rider = None
        for idx, rider in enumerate(riders, start=1):
            if rider in nonHelmeted_riders and is_point_in_box(center_plate_x, center_plate_y, rider):
                associated_rider = idx
                break
        if associated_rider:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
            draw_label(frame, "Number Plate", (px1, py1), (255, 255, 0))
            plate_img = frame[py1:py2, px1:px2]
            temp_plate_path = "temp_plate.jpg"
            cv2.imwrite(temp_plate_path, plate_img)
            plate_text = extract_text_from_plate(temp_plate_path).strip()
            process_number_plate(plate_text)

    return resize_to_fit(frame)

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    output = process_frame(img)
    cv2.imshow("Final Detections", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    import time
    from collections import defaultdict
    import os
    import datetime

    try:
        tracker = cv2.TrackerKCF_create
    except:
        try:
            tracker = cv2.legacy.TrackerKCF_create
        except:
            print("Warning: OpenCV tracking modules not available. Using basic tracking.")
            tracker = None
    
    CONFIDENCE_THRESHOLD = 0.6
    MAX_CONFIDENCE = 0.9
    MIN_CONFIDENCE = 0.1
    CONFIDENCE_INCREMENT = 0.2
    CONFIDENCE_DECREMENT = 0.1
    IOU_THRESHOLD = 0.5
    
    # Create directory for violation images and plates
    violation_dir = os.path.join(os.path.dirname(video_path), "violations")
    os.makedirs(violation_dir, exist_ok=True)
    
    # Prepare violation log file
    log_file = os.path.join(os.path.dirname(video_path), "helmet_violations.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Timestamp,Rider ID,License Plate,Confidence\n")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = video_path.replace('.', '_processed.')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    riders = {}
    next_rider_id = 1
    processed_plates = set()  # To avoid processing the same plate multiple times
    
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def check_rider_helmet(rider_box, helmets, without_helmets):
        rider_center_x = (rider_box[0] + rider_box[2]) / 2
        rider_center_y = (rider_box[1] + rider_box[3]) / 2
        
        for wh in without_helmets:
            wh_center_x = (wh[0] + wh[2]) / 2
            wh_center_y = (wh[1] + wh[3]) / 2
            if is_point_in_box(wh_center_x, wh_center_y, rider_box):
                return False
                
        for h in helmets:
            h_center_x = (h[0] + h[2]) / 2
            h_center_y = (h[1] + h[3]) / 2
            if is_point_in_box(h_center_x, h_center_y, rider_box):
                return True
        
        return None
    
    def process_number_plate(plate_text, rider_id, frame, plate_box):
        """Process the detected license plate and log violation"""
        if not plate_text:
            return

        # Clean the plate text (remove non-alphanumeric)
        plate_text = ''.join(c for c in plate_text if c.isalnum())
        if len(plate_text) < 4:  # Too short to be valid
            return
            
        # Generate a unique identifier for this violation
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        img_name = f"violation_{rider_id}_{timestamp}.jpg"
        plate_img_name = f"plate_{rider_id}_{timestamp}.jpg"
        
        # Save the violation image
        violation_img_path = os.path.join(violation_dir, img_name)
        cv2.imwrite(violation_img_path, frame)
        
        # Save just the plate image for reference
        px1, py1, px2, py2 = plate_box
        plate_img = frame[py1:py2, px1:px2]
        plate_img_path = os.path.join(violation_dir, plate_img_name)
        cv2.imwrite(plate_img_path, plate_img)
        
        # Log the violation
        with open(log_file, 'a') as f:
            confidence = riders[rider_id]['confidence'] if rider_id in riders else 0.0
            f.write(f"{timestamp},{rider_id},{plate_text},{confidence:.2f}\n")
            
        print(f"🚨 VIOLATION: Rider {rider_id} without helmet, license plate: {plate_text}")
        
        # Add to processed plates to avoid duplicates
        key = f"{rider_id}_{plate_text}"
        processed_plates.add(key)
        
        return plate_text
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        output_frame = frame.copy()
        riders_to_remove = []
        for rider_id, rider_data in list(riders.items()):
            if 'tracker' in rider_data and rider_data['tracker'] is not None:
                success, box = rider_data['tracker'].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    riders[rider_id]['box'] = [x, y, x+w, y+h]
                else:
                    riders_to_remove.append(rider_id)
    
        for rider_id in riders_to_remove:
            del riders[rider_id]
        
        results = model(frame)
        predictions = results[0]
        
        current_riders = []
        current_helmets = []
        current_without_helmets = []
        current_plates = []
        
        for i in range(len(predictions.boxes)):
            box = predictions.boxes[i]
            coords = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, coords)
            cls = int(box.cls.cpu().numpy()[0])
            label = predictions.names[cls].lower()
            conf = float(box.conf.cpu().numpy()[0])
            
            # Only consider high confidence detections
            if conf < 0.5:  # You can adjust this threshold
                continue
                
            if label == "rider":
                current_riders.append([x1, y1, x2, y2])
            elif label == "without helmet":
                current_without_helmets.append([x1, y1, x2, y2])
            elif label == "helmet":
                current_helmets.append([x1, y1, x2, y2])
            elif label == "number plate":
                current_plates.append([x1, y1, x2, y2])
        
        # Match current detections with existing tracked riders
        matched_rider_ids = []
        
        for rider_box in current_riders:
            best_match_id = None
            best_match_iou = 0
            
            # Find the best matching existing rider
            for rider_id, rider_data in riders.items():
                iou = calculate_iou(rider_box, rider_data['box'])
                if iou > IOU_THRESHOLD and iou > best_match_iou:
                    best_match_id = rider_id
                    best_match_iou = iou
            
            if best_match_id is not None:
                riders[best_match_id]['box'] = rider_box
                
                helmet_status = check_rider_helmet(rider_box, current_helmets, current_without_helmets)
                if helmet_status is not None:
                    current_status = riders[best_match_id]['has_helmet']
                    current_confidence = riders[best_match_id]['confidence']
                    
                    if helmet_status == current_status:
                        new_confidence = min(current_confidence + CONFIDENCE_INCREMENT, MAX_CONFIDENCE)
                    else:
                        new_confidence = max(current_confidence - CONFIDENCE_DECREMENT, MIN_CONFIDENCE)
                        
                        # If confidence drops below threshold, change status
                        if new_confidence < 0.5 and current_confidence >= 0.5:
                            riders[best_match_id]['has_helmet'] = helmet_status
                    
                    riders[best_match_id]['confidence'] = new_confidence
                
                if tracker is not None:
                    x1, y1, x2, y2 = rider_box
                    riders[best_match_id]['tracker'] = tracker()
                    riders[best_match_id]['tracker'].init(frame, (x1, y1, x2-x1, y2-y1))
                
                matched_rider_ids.append(best_match_id)
            else:
                helmet_status = check_rider_helmet(rider_box, current_helmets, current_without_helmets)
                if helmet_status is None:
                    # If uncertain, assume they have a helmet (be lenient)
                    helmet_status = True
                    initial_confidence = 0.5
                else:
                    initial_confidence = CONFIDENCE_THRESHOLD
                
                rider_id = next_rider_id
                next_rider_id += 1
                
                x1, y1, x2, y2 = rider_box
                new_tracker = None
                if tracker is not None:
                    new_tracker = tracker()
                    new_tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                
                riders[rider_id] = {
                    'box': rider_box,
                    'has_helmet': helmet_status,
                    'confidence': initial_confidence,
                    'tracker': new_tracker,
                    'frames_tracked': 0,
                    'plate_processed': False
                }
                
                matched_rider_ids.append(rider_id)
        
        for rider_id in list(riders.keys()):
            if rider_id not in matched_rider_ids:
                riders[rider_id]['frames_tracked'] += 1
                # Remove after 30 frames (about 1 second) of not being detected
                if riders[rider_id]['frames_tracked'] > 30:
                    del riders[rider_id]
        
        # Process plates for non-helmeted riders
        for plate in current_plates:
            px1, py1, px2, py2 = plate
            center_plate_x = (px1 + px2) / 2
            center_plate_y = (py1 + py2) / 2
            
            # Find associated rider
            associated_rider_id = None
            for rider_id, rider_data in riders.items():
                rider_box = rider_data['box']
                if (is_point_in_box(center_plate_x, center_plate_y, rider_box) and 
                    not rider_data['has_helmet'] and
                    rider_data['confidence'] > CONFIDENCE_THRESHOLD and
                    not rider_data.get('plate_processed', False)):
                    associated_rider_id = rider_id
                    break
            
            # Draw plate box and process if associated with non-helmeted rider
            if associated_rider_id:
                cv2.rectangle(output_frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
                draw_label(output_frame, f"Number Plate (Rider {associated_rider_id})", (px1, py1), (255, 255, 0))
                
                # Extract plate text with OCR
                plate_img = frame[py1:py2, px1:px2]
                temp_plate_path = os.path.join(violation_dir, f"temp_plate_{frame_count}.jpg")
                cv2.imwrite(temp_plate_path, plate_img)
                
                # Extract text from plate
                plate_text = extract_text_from_plate(temp_plate_path).strip()
                
                # Process the plate only if we got some text and haven't processed it before
                key = f"{associated_rider_id}_{plate_text}"
                if plate_text and key not in processed_plates:
                    process_number_plate(plate_text, associated_rider_id, frame, plate)
                    riders[associated_rider_id]['plate_processed'] = True
                
                try:
                    os.remove(temp_plate_path)  # Clean up temp file
                except:
                    pass
            else:
                cv2.rectangle(output_frame, (px1, py1), (px2, py2), (255, 255, 0), 1)
                draw_label(output_frame, "Number Plate", (px1, py1), (255, 255, 0))
        
        for helmet in current_helmets:
            hx1, hy1, hx2, hy2 = helmet
            cv2.rectangle(output_frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 1)
            
        for wh in current_without_helmets:
            whx1, why1, whx2, why2 = wh
            cv2.rectangle(output_frame, (whx1, why1), (whx2, why2), (0, 0, 255), 1)
        
        for rider_id, rider_data in riders.items():
            x1, y1, x2, y2 = rider_data['box']
            has_helmet = rider_data['has_helmet']
            confidence = rider_data['confidence']
            
            if has_helmet:
                color = (0, 255, 0)  
                status_text = f"Rider {rider_id}: Helmet ({confidence:.2f})"
            else:
                color = (0, 0, 255)  
                status_text = f"Rider {rider_id}: No Helmet ({confidence:.2f})"
                
                # Add alert indicator for non-helmeted riders with high confidence
                if confidence > CONFIDENCE_THRESHOLD:
                    cv2.putText(output_frame, "⚠️ VIOLATION", (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            thickness = max(1, int(confidence * 3))
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            draw_label(output_frame, status_text, (x1, y1), color)
        
        display_frame = resize_to_fit(output_frame)
        out.write(output_frame)
        cv2.imshow("Video Detection", display_frame)
        
        # Print progress every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_rate = frame_count / elapsed
            print(f"Processed {frame_count} frames. FPS: {fps_rate:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Violation log saved to {log_file}")
    print(f"Violation images saved in {violation_dir}")
    
    return output_path
def process_live(f=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output = process_frame(frame, f)
        cv2.imshow("Live Detection", output)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Select input type:")
    print("1. Image")
    print("2. Video")
    print("3. Live Camera Capture")
    choice = input("Enter choice (1/2/3): ")

    if choice == '1':
        path = input("Enter image path: ").strip()
        if os.path.exists(path):
            process_image(path)
    elif choice == '2':
        path = input("Enter video path: ").strip()
        if os.path.exists(path):
            process_video(path)
    elif choice == '3':
        process_live(f=True)
    elif choice.strip() == '3':
        process_live()
    else:
        print("Invalid choice")
        pass

if __name__ == "__main__":
    main()
