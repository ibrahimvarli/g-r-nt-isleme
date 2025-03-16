import cv2
import numpy as np
import os
from math import hypot

# Function to detect face and facial landmarks using OpenCV
def detect_face(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    try:
        # Load the pre-trained face detector from OpenCV using os.path.join for proper path handling
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            return None, None
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None
        
        # Get the first face detected
        (x, y, w, h) = faces[0]
        
        # Create simulated landmarks (simplified version)
        # In a real scenario, you would use a proper landmark detector
        points = []
        
        # Jaw points (0-16)
        for i in range(17):
            jaw_x = x + int(i * w / 16)
            jaw_y = y + h - int(h / 8)
            points.append((jaw_x, jaw_y))
        
        # Eyebrow points (17-26)
        for i in range(5):
            # Left eyebrow
            left_eyebrow_x = x + int(w / 4) + int(i * w / 10)
            left_eyebrow_y = y + int(h / 4)
            points.append((left_eyebrow_x, left_eyebrow_y))
        
        for i in range(5):
            # Right eyebrow
            right_eyebrow_x = x + int(w / 2) + int(i * w / 10)
            right_eyebrow_y = y + int(h / 4)
            points.append((right_eyebrow_x, right_eyebrow_y))
        
        # Nose points (27-35)
        nose_center_x = x + int(w / 2)
        for i in range(9):
            nose_x = nose_center_x
            nose_y = y + int(h / 3) + int(i * h / 15)
            points.append((nose_x, nose_y))
        
        # Eye points (36-47)
        # Left eye
        left_eye_center_x = x + int(w / 3)
        left_eye_center_y = y + int(h / 3)
        for i in range(6):
            angle = i * 60
            radius = w / 12
            eye_x = left_eye_center_x + int(radius * np.cos(np.radians(angle)))
            eye_y = left_eye_center_y + int(radius * np.sin(np.radians(angle)))
            points.append((eye_x, eye_y))
        
        # Right eye
        right_eye_center_x = x + int(2 * w / 3)
        right_eye_center_y = y + int(h / 3)
        for i in range(6):
            angle = i * 60
            radius = w / 12
            eye_x = right_eye_center_x + int(radius * np.cos(np.radians(angle)))
            eye_y = right_eye_center_y + int(radius * np.sin(np.radians(angle)))
            points.append((eye_x, eye_y))
        
        # Mouth points (48-67)
        mouth_center_x = x + int(w / 2)
        mouth_center_y = y + int(3 * h / 4)
        
        # Outer lip
        for i in range(12):
            angle = i * 30
            radius = w / 6
            lip_x = mouth_center_x + int(radius * np.cos(np.radians(angle)))
            lip_y = mouth_center_y + int(radius * np.sin(np.radians(angle)))
            points.append((lip_x, lip_y))
        
        # Inner lip
        for i in range(8):
            angle = i * 45
            radius = w / 10
            lip_x = mouth_center_x + int(radius * np.cos(np.radians(angle)))
            lip_y = mouth_center_y + int(radius * np.sin(np.radians(angle)))
            points.append((lip_x, lip_y))
        
        return (x, y, w, h), points
    
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None, None

# Function to draw landmarks on frame
def draw_landmarks(frame, points):
    if points is None:
        return
    
    for point in points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)

# Function to create a simple 3D model from landmarks
def create_face_model(points):
    if points is None:
        return np.zeros((500, 500, 3), np.uint8)
    
    # Create a blank image for the 3D model visualization
    model_img = np.zeros((500, 500, 3), np.uint8)
    
    # Draw the face outline
    for i in range(16):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    
    # Draw the eyebrows
    for i in range(17, 21):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    for i in range(22, 26):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    
    # Draw the nose
    for i in range(27, 35):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    
    # Draw the eyes
    for i in range(36, 41):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    cv2.line(model_img, points[41], points[36], (0, 255, 0), 2)
    
    for i in range(42, 47):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    cv2.line(model_img, points[47], points[42], (0, 255, 0), 2)
    
    # Draw the lips
    for i in range(48, 59):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    cv2.line(model_img, points[59], points[48], (0, 255, 0), 2)
    
    for i in range(60, 67):
        cv2.line(model_img, points[i], points[i+1], (0, 255, 0), 2)
    cv2.line(model_img, points[67], points[60], (0, 255, 0), 2)
    
    return model_img

# Function to calculate depth for 3D model (simplified)
def calculate_depth(points):
    if points is None:
        return []
    
    # Calculate the distance between the eyes as a reference
    left_eye = (sum([points[36][0], points[37][0], points[38][0], points[39][0], points[40][0], points[41][0]]) / 6,
                sum([points[36][1], points[37][1], points[38][1], points[39][1], points[40][1], points[41][1]]) / 6)
    right_eye = (sum([points[42][0], points[43][0], points[44][0], points[45][0], points[46][0], points[47][0]]) / 6,
                 sum([points[42][1], points[43][1], points[44][1], points[45][1], points[46][1], points[47][1]]) / 6)
    
    eye_distance = hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    
    # Calculate depth for each point (simplified model)
    depth_points = []
    for point in points:
        # Nose and center face points have more depth
        if point in points[27:36]:
            depth = 30
        # Eyes and eyebrows have medium depth
        elif point in points[17:27] or point in points[36:48]:
            depth = 15
        # Jaw and lips have less depth
        else:
            depth = 5
            
        depth_points.append((point[0], point[1], depth))
    
    return depth_points

# Main function
def main():
    try:
        # Start video capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video capture device")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Detect face and get landmarks
            face_rect, points = detect_face(frame)
            
            if face_rect is not None:
                # Draw landmarks on the frame
                draw_landmarks(frame, points)
                
                # Create 3D model
                model_img = create_face_model(points)
                
                # Calculate depth points for 3D model
                depth_points = calculate_depth(points)
                
                # Display depth information
                depth_img = np.zeros((500, 500, 3), np.uint8)
                for i, (x, y, z) in enumerate(depth_points):
                    # Scale coordinates to fit in the depth image
                    x_scaled = int((x / frame.shape[1]) * 500)
                    y_scaled = int((y / frame.shape[0]) * 500)
                    # Color based on depth (blue to red)
                    color = (255 - z * 8, 0, z * 8)
                    cv2.circle(depth_img, (x_scaled, y_scaled), 2, color, -1)
                    
                # Draw face rectangle
                x, y, w, h = face_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Show the 3D model and depth visualization
                cv2.imshow("3D Face Model", model_img)
                cv2.imshow("Depth Visualization", depth_img)
            
            # Show the frame
            cv2.imshow("Face Detection", frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error in main function: {e}")
    finally:
        # Release resources
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Program terminated")

if __name__ == "__main__":
    main()