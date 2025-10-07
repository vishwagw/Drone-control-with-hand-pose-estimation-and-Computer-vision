import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json

class HandPoseEstimator:
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the hand pose estimator
        
        Args:
            static_image_mode: If True, treats input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark connections for drawing
        self.connections = self.mp_hands.HAND_CONNECTIONS
        
        # Landmark names for reference
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Process a single image and extract hand landmarks
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (annotated_image, hand_landmarks_list)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.connections,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                hand_landmarks_list.append(landmarks)
        
        return annotated_image, hand_landmarks_list

    def process_video_stream(self, source=0):
        """
        Process video stream from camera or video file
        
        Args:
            source: Camera index (0 for default) or video file path
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        print("Press 'q' to quit, 's' to save current frame landmarks")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            annotated_frame, hand_landmarks = self.process_image(frame)
            
            # Display information
            cv2.putText(annotated_frame, f"Hands detected: {len(hand_landmarks)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Hand Pose Estimation', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and hand_landmarks:
                self.save_landmarks(hand_landmarks, f"landmarks_{cv2.getTickCount()}.json")
                print("Landmarks saved!")
        
        cap.release()
        cv2.destroyAllWindows()

    def calculate_hand_angles(self, landmarks: List) -> dict:
        """
        Calculate angles between fingers and hand orientation
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Dictionary containing calculated angles
        """
        if len(landmarks) != 21:
            return {}
        
        angles = {}
        
        # Extract key points
        wrist = np.array([landmarks[0]['x'], landmarks[0]['y']])
        thumb_tip = np.array([landmarks[4]['x'], landmarks[4]['y']])
        index_tip = np.array([landmarks[8]['x'], landmarks[8]['y']])
        middle_tip = np.array([landmarks[12]['x'], landmarks[12]['y']])
        ring_tip = np.array([landmarks[16]['x'], landmarks[16]['y']])
        pinky_tip = np.array([landmarks[20]['x'], landmarks[20]['y']])
        
        # Calculate angles between fingers
        def angle_between_vectors(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        
        # Finger vectors from wrist
        thumb_vec = thumb_tip - wrist
        index_vec = index_tip - wrist
        middle_vec = middle_tip - wrist
        ring_vec = ring_tip - wrist
        pinky_vec = pinky_tip - wrist
        
        angles['thumb_index'] = angle_between_vectors(thumb_vec, index_vec)
        angles['index_middle'] = angle_between_vectors(index_vec, middle_vec)
        angles['middle_ring'] = angle_between_vectors(middle_vec, ring_vec)
        angles['ring_pinky'] = angle_between_vectors(ring_vec, pinky_vec)
        
        return angles

    def detect_gestures(self, landmarks: List) -> str:
        """
        Simple gesture recognition based on finger positions
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Detected gesture name
        """
        if len(landmarks) != 21:
            return "Unknown"
        
        # Get fingertip and joint positions
        thumb_tip = landmarks[4]['y']
        thumb_joint = landmarks[3]['y']
        
        index_tip = landmarks[8]['y']
        index_joint = landmarks[6]['y']
        
        middle_tip = landmarks[12]['y']
        middle_joint = landmarks[10]['y']
        
        ring_tip = landmarks[16]['y']
        ring_joint = landmarks[14]['y']
        
        pinky_tip = landmarks[20]['y']
        pinky_joint = landmarks[18]['y']
        
        # Check if fingers are extended (tip is above joint)
        fingers_up = []
        
        # Thumb (special case - check x coordinate)
        if landmarks[4]['x'] > landmarks[3]['x']:  # Right hand
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers
        fingers_up.append(1 if index_tip < index_joint else 0)
        fingers_up.append(1 if middle_tip < middle_joint else 0)
        fingers_up.append(1 if ring_tip < ring_joint else 0)
        fingers_up.append(1 if pinky_tip < pinky_joint else 0)
        
        # Gesture recognition
        total_fingers = sum(fingers_up)
        
        if total_fingers == 0:
            return "Fist"
        elif total_fingers == 1:
            if fingers_up[1]:
                return "Point"
            elif fingers_up[0]:
                return "Thumbs Up"
        elif total_fingers == 2:
            if fingers_up[1] and fingers_up[2]:
                return "Peace"
            elif fingers_up[0] and fingers_up[4]:
                return "Rock"
        elif total_fingers == 5:
            return "Open Hand"
        else:
            return f"{total_fingers} Fingers"

    def save_landmarks(self, hand_landmarks_list: List, filename: str):
        """
        Save hand landmarks to JSON file
        
        Args:
            hand_landmarks_list: List of hand landmarks
            filename: Output filename
        """
        data = {
            'timestamp': cv2.getTickCount(),
            'hands': hand_landmarks_list
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_landmarks(self, filename: str) -> List:
        """
        Load hand landmarks from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            List of hand landmarks
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['hands']

    def visualize_3d_landmarks(self, landmarks: List):
        """
        Create 3D visualization of hand landmarks
        
        Args:
            landmarks: List of hand landmarks
        """
        if len(landmarks) != 21:
            print("Invalid number of landmarks")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x = [landmark['x'] for landmark in landmarks]
        y = [landmark['y'] for landmark in landmarks]
        z = [landmark['z'] for landmark in landmarks]
        
        # Plot points
        ax.scatter(x, y, z, c='red', s=50)
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        for connection in connections:
            start, end = connection
            ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'b-')
        
        # Labels and formatting
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Hand Landmarks')
        
        plt.show()

# Example usage and demo functions
def demo_image_processing():
    """Demo function for processing static images"""
    estimator = HandPoseEstimator(static_image_mode=True)
    
    # Load an image (replace with your image path)
    # image = cv2.imread('hand_image.jpg')
    # if image is not None:
    #     annotated_image, landmarks = estimator.process_image(image)
    #     
    #     if landmarks:
    #         print(f"Detected {len(landmarks)} hands")
    #         for i, hand in enumerate(landmarks):
    #             gesture = estimator.detect_gestures(hand)
    #             angles = estimator.calculate_hand_angles(hand)
    #             print(f"Hand {i+1}: {gesture}")
    #             print(f"Angles: {angles}")
    #     
    #     cv2.imshow('Result', annotated_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    print("Image processing demo - uncomment code and provide image path")

def demo_video_stream():
    """Demo function for real-time video processing"""
    estimator = HandPoseEstimator()
    estimator.process_video_stream()

if __name__ == "__main__":
    print("Hand Pose Estimation System")
    print("1. Run demo_video_stream() for real-time detection")
    print("2. Run demo_image_processing() for static image processing")
    
    # Uncomment to run demos
    # demo_video_stream()
    # demo_image_processing()