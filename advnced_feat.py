import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from collections import deque
import time

class AdvancedHandPoseSystem:
    def __init__(self, base_estimator):
        """
        Advanced features built on top of base hand pose estimator
        
        Args:
            base_estimator: Instance of HandPoseEstimator
        """
        self.base_estimator = base_estimator
        self.gesture_classifier = None
        self.landmark_history = deque(maxlen=30)  # Store last 30 frames
        self.gesture_history = deque(maxlen=10)   # Store last 10 gestures
        
    def extract_features(self, landmarks):
        """
        Extract comprehensive features from hand landmarks
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            Feature vector
        """
        if len(landmarks) != 21:
            return np.zeros(84)  # Return zero vector if invalid
        
        features = []
        
        # 1. Raw coordinates (normalized)
        coords = []
        for landmark in landmarks:
            coords.extend([landmark['x'], landmark['y'], landmark['z']])
        
        # Normalize coordinates relative to wrist
        wrist = np.array([landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']])
        normalized_coords = []
        for i in range(0, len(coords), 3):
            point = np.array(coords[i:i+3])
            normalized = point - wrist
            normalized_coords.extend(normalized)
        
        features.extend(normalized_coords)  # 63 features
        
        # 2. Distances between key points
        key_points = [0, 4, 8, 12, 16, 20]  # Wrist, thumb, index, middle, ring, pinky tips
        for i in range(len(key_points)):
            for j in range(i+1, len(key_points)):
                p1 = np.array([landmarks[key_points[i]]['x'], 
                              landmarks[key_points[i]]['y'], 
                              landmarks[key_points[i]]['z']])
                p2 = np.array([landmarks[key_points[j]]['x'], 
                              landmarks[key_points[j]]['y'], 
                              landmarks[key_points[j]]['z']])
                distance = np.linalg.norm(p2 - p1)
                features.append(distance)  # 15 features
        
        # 3. Finger curl ratios
        finger_ratios = self.calculate_finger_curl_ratios(landmarks)
        features.extend(finger_ratios)  # 5 features
        
        # 4. Hand span (distance from thumb to pinky)
        thumb_tip = np.array([landmarks[4]['x'], landmarks[4]['y'], landmarks[4]['z']])
        pinky_tip = np.array([landmarks[20]['x'], landmarks[20]['y'], landmarks[20]['z']])
        hand_span = np.linalg.norm(pinky_tip - thumb_tip)
        features.append(hand_span)  # 1 feature
        
        return np.array(features)
    
    def calculate_finger_curl_ratios(self, landmarks):
        """
        Calculate how much each finger is curled (0 = straight, 1 = fully curled)
        """
        ratios = []
        
        # Finger tip and base indices
        fingers = [
            [2, 3, 4],    # Thumb
            [5, 6, 7, 8], # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20]  # Pinky
        ]
        
        for finger in fingers:
            if len(finger) == 3:  # Thumb
                base = np.array([landmarks[finger[0]]['x'], landmarks[finger[0]]['y']])
                tip = np.array([landmarks[finger[2]]['x'], landmarks[finger[2]]['y']])
                straight_distance = np.linalg.norm(tip - base)
                
                # Calculate actual path distance
                path_distance = 0
                for i in range(len(finger) - 1):
                    p1 = np.array([landmarks[finger[i]]['x'], landmarks[finger[i]]['y']])
                    p2 = np.array([landmarks[finger[i+1]]['x'], landmarks[finger[i+1]]['y']])
                    path_distance += np.linalg.norm(p2 - p1)
                
                curl_ratio = 1 - (straight_distance / path_distance) if path_distance > 0 else 0
                ratios.append(curl_ratio)
            else:  # Other fingers
                base = np.array([landmarks[finger[0]]['x'], landmarks[finger[0]]['y']])
                tip = np.array([landmarks[finger[3]]['x'], landmarks[finger[3]]['y']])
                straight_distance = np.linalg.norm(tip - base)
                
                path_distance = 0
                for i in range(len(finger) - 1):
                    p1 = np.array([landmarks[finger[i]]['x'], landmarks[finger[i]]['y']])
                    p2 = np.array([landmarks[finger[i+1]]['x'], landmarks[finger[i+1]]['y']])
                    path_distance += np.linalg.norm(p2 - p1)
                
                curl_ratio = 1 - (straight_distance / path_distance) if path_distance > 0 else 0
                ratios.append(curl_ratio)
        
        return ratios
    
    def train_gesture_classifier(self, training_data, labels):
        """
        Train a custom gesture classifier
        
        Args:
            training_data: List of landmark sequences
            labels: Corresponding gesture labels
        """
        features = []
        for landmarks in training_data:
            if landmarks:  # Check if landmarks exist
                feature_vector = self.extract_features(landmarks[0])  # Use first hand
                features.append(feature_vector)
            else:
                features.append(np.zeros(84))  # Zero vector for missing data
        
        X = np.array(features)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest classifier
        self.gesture_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gesture_classifier.fit(X_train, y_train)
        
        # Evaluate
        accuracy = self.gesture_classifier.score(X_test, y_test)
        print(f"Gesture classifier accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def predict_gesture(self, landmarks):
        """
        Predict gesture using trained classifier
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            Predicted gesture and confidence
        """
        if self.gesture_classifier is None:
            return "No trained classifier", 0.0
        
        if not landmarks:
            return "No hand detected", 0.0
        
        features = self.extract_features(landmarks[0]).reshape(1, -1)
        prediction = self.gesture_classifier.predict(features)[0]
        probabilities = self.gesture_classifier.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def save_classifier(self, filename):
        """Save trained classifier to file"""
        if self.gesture_classifier:
            with open(filename, 'wb') as f:
                pickle.dump(self.gesture_classifier, f)
    
    def load_classifier(self, filename):
        """Load trained classifier from file"""
        with open(filename, 'rb') as f:
            self.gesture_classifier = pickle.load(f)
    
    def smooth_landmarks(self, landmarks, alpha=0.7):
        """
        Apply temporal smoothing to landmarks
        
        Args:
            landmarks: Current frame landmarks
            alpha: Smoothing factor (0-1)
            
        Returns:
            Smoothed landmarks
        """
        if not self.landmark_history or not landmarks:
            self.landmark_history.append(landmarks)
            return landmarks
        
        # Get previous landmarks
        prev_landmarks = self.landmark_history[-1]
        
        if not prev_landmarks or len(prev_landmarks) != len(landmarks):
            self.landmark_history.append(landmarks)
            return landmarks
        
        # Apply exponential smoothing
        smoothed_landmarks = []
        for i, hand in enumerate(landmarks):
            if i < len(prev_landmarks):
                smoothed_hand = []
                for j, landmark in enumerate(hand):
                    if j < len(prev_landmarks[i]):
                        prev_landmark = prev_landmarks[i][j]
                        smoothed_landmark = {
                            'x': alpha * landmark['x'] + (1 - alpha) * prev_landmark['x'],
                            'y': alpha * landmark['y'] + (1 - alpha) * prev_landmark['y'],
                            'z': alpha * landmark['z'] + (1 - alpha) * prev_landmark['z']
                        }
                        smoothed_hand.append(smoothed_landmark)
                    else:
                        smoothed_hand.append(landmark)
                smoothed_landmarks.append(smoothed_hand)
            else:
                smoothed_landmarks.append(hand)
        
        self.landmark_history.append(smoothed_landmarks)
        return smoothed_landmarks
    
    def detect_hand_motion(self, landmarks):
        """
        Detect hand motion patterns
        
        Args:
            landmarks: Current landmarks
            
        Returns:
            Motion characteristics
        """
        if len(self.landmark_history) < 5:
            return {"velocity": 0, "direction": "stationary", "acceleration": 0}
        
        if not landmarks:
            return {"velocity": 0, "direction": "stationary", "acceleration": 0}
        
        # Calculate wrist position over time
        wrist_positions = []
        for hist_landmarks in list(self.landmark_history)[-5:]:
            if hist_landmarks and len(hist_landmarks) > 0:
                wrist = hist_landmarks[0][0]  # First hand, wrist landmark
                wrist_positions.append([wrist['x'], wrist['y']])
        
        if len(wrist_positions) < 2:
            return {"velocity": 0, "direction": "stationary", "acceleration": 0}
        
        # Calculate velocity
        velocities = []
        for i in range(1, len(wrist_positions)):
            pos_diff = np.array(wrist_positions[i]) - np.array(wrist_positions[i-1])
            velocity = np.linalg.norm(pos_diff)
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities)
        
        # Calculate acceleration
        if len(velocities) > 1:
            acceleration = velocities[-1] - velocities[-2]
        else:
            acceleration = 0
        
        # Determine direction
        if avg_velocity < 0.01:
            direction = "stationary"
        else:
            total_displacement = np.array(wrist_positions[-1]) - np.array(wrist_positions[0])
            if abs(total_displacement[0]) > abs(total_displacement[1]):
                direction = "horizontal"
            else:
                direction = "vertical"
        
        return {
            "velocity": avg_velocity,
            "direction": direction,
            "acceleration": acceleration
        }
    
    def create_training_interface(self):
        """
        Interactive interface for collecting training data
        """
        print("Gesture Training Interface")
        print("Instructions:")
        print("- Press 'c' to capture gesture")
        print("- Press 'n' for next gesture")
        print("- Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        training_data = []
        labels = []
        current_gesture = input("Enter gesture name: ")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, landmarks = self.base_estimator.process_image(frame)
            
            # Display info
            cv2.putText(annotated_frame, f"Current gesture: {current_gesture}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Samples collected: {len([l for l in labels if l == current_gesture])}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Training Interface', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and landmarks:
                training_data.append(landmarks)
                labels.append(current_gesture)
                print(f"Captured sample for {current_gesture}")
            elif key == ord('n'):
                current_gesture = input("Enter next gesture name (or 'done' to finish): ")
                if current_gesture.lower() == 'done':
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if training_data:
            print(f"Training classifier with {len(training_data)} samples...")
            accuracy = self.train_gesture_classifier(training_data, labels)
            save_choice = input("Save classifier? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Enter filename: ")
                self.save_classifier(filename)
                print(f"Classifier saved as {filename}")
        
        return training_data, labels

# Example usage
def advanced_demo():
    """Demo of advanced features"""
    from hand_pose_estimate import HandPoseEstimator  # Import base estimator
    
    base_estimator = HandPoseEstimator()
    advanced_system = AdvancedHandPoseSystem(base_estimator)
    
    # Create training interface
    # training_data, labels = advanced_system.create_training_interface()
    
    print("Advanced Hand Pose System ready!")
    print("Features available:")
    print("- Custom gesture training")
    print("- Motion detection")
    print("- Landmark smoothing")
    print("- Feature extraction")

if __name__ == "__main__":
    advanced_demo()