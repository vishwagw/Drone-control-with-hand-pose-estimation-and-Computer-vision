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
            landmarks: Hand landmarks (single hand - list of 21 landmarks)
            
        Returns:
            Feature vector
        """
        # Fix: Handle different landmark formats
        if landmarks is None:
            return np.zeros(84)
            
        # Handle case where landmarks is a list of hands
        if isinstance(landmarks, list) and len(landmarks) > 0:
            if isinstance(landmarks[0], list):  # List of hands
                landmarks = landmarks[0]  # Take first hand
        
        # Ensure we have exactly 21 landmarks
        if len(landmarks) != 21:
            return np.zeros(84)  # Return zero vector if invalid
        
        features = []
        
        # 1. Raw coordinates (normalized)
        coords = []
        for landmark in landmarks:
            # Fix: Handle different landmark formats (dict vs object)
            if isinstance(landmark, dict):
                coords.extend([landmark['x'], landmark['y'], landmark.get('z', 0)])
            else:  # Assume it's an object with x, y, z attributes
                coords.extend([getattr(landmark, 'x', 0), 
                              getattr(landmark, 'y', 0), 
                              getattr(landmark, 'z', 0)])
        
        # Normalize coordinates relative to wrist
        wrist = np.array(coords[0:3])  # First landmark is wrist
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
                try:
                    # Fix: Safe coordinate extraction
                    p1_coords = self._get_landmark_coords(landmarks[key_points[i]])
                    p2_coords = self._get_landmark_coords(landmarks[key_points[j]])
                    
                    if p1_coords is not None and p2_coords is not None:
                        distance = np.linalg.norm(p2_coords - p1_coords)
                        features.append(distance)
                    else:
                        features.append(0.0)
                except (IndexError, KeyError, AttributeError):
                    features.append(0.0)  # 15 features total
        
        # 3. Finger curl ratios
        try:
            finger_ratios = self.calculate_finger_curl_ratios(landmarks)
            features.extend(finger_ratios)  # 5 features
        except Exception:
            features.extend([0.0] * 5)  # Default values if calculation fails
        
        # 4. Hand span (distance from thumb to pinky)
        try:
            thumb_coords = self._get_landmark_coords(landmarks[4])
            pinky_coords = self._get_landmark_coords(landmarks[20])
            
            if thumb_coords is not None and pinky_coords is not None:
                hand_span = np.linalg.norm(pinky_coords - thumb_coords)
                features.append(hand_span)
            else:
                features.append(0.0)
        except (IndexError, KeyError, AttributeError):
            features.append(0.0)  # 1 feature
        
        # Ensure we have exactly 84 features
        while len(features) < 84:
            features.append(0.0)
        
        return np.array(features[:84])  # Truncate if too many features
    
    def _get_landmark_coords(self, landmark):
        """
        Safely extract x, y, z coordinates from landmark
        
        Args:
            landmark: Landmark object or dict
            
        Returns:
            numpy array of coordinates or None if failed
        """
        try:
            if isinstance(landmark, dict):
                return np.array([landmark['x'], landmark['y'], landmark.get('z', 0)])
            else:  # Assume it's an object with attributes
                return np.array([getattr(landmark, 'x', 0), 
                               getattr(landmark, 'y', 0), 
                               getattr(landmark, 'z', 0)])
        except (KeyError, AttributeError):
            return None
    
    def calculate_finger_curl_ratios(self, landmarks):
        """
        Calculate how much each finger is curled (0 = straight, 1 = fully curled)
        """
        ratios = []
        
        # Finger tip and base indices
        fingers = [
            [2, 3, 4],        # Thumb
            [5, 6, 7, 8],     # Index
            [9, 10, 11, 12],  # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20]  # Pinky
        ]
        
        for finger in fingers:
            try:
                if len(finger) == 3:  # Thumb
                    base_coords = self._get_landmark_coords(landmarks[finger[0]])
                    tip_coords = self._get_landmark_coords(landmarks[finger[2]])
                    
                    if base_coords is None or tip_coords is None:
                        ratios.append(0.0)
                        continue
                        
                    straight_distance = np.linalg.norm(tip_coords[:2] - base_coords[:2])
                    
                    # Calculate actual path distance
                    path_distance = 0
                    for i in range(len(finger) - 1):
                        p1_coords = self._get_landmark_coords(landmarks[finger[i]])
                        p2_coords = self._get_landmark_coords(landmarks[finger[i+1]])
                        
                        if p1_coords is not None and p2_coords is not None:
                            path_distance += np.linalg.norm(p2_coords[:2] - p1_coords[:2])
                    
                    curl_ratio = 1 - (straight_distance / path_distance) if path_distance > 0 else 0
                    ratios.append(max(0, min(1, curl_ratio)))  # Clamp between 0 and 1
                    
                else:  # Other fingers (4 joints)
                    base_coords = self._get_landmark_coords(landmarks[finger[0]])
                    tip_coords = self._get_landmark_coords(landmarks[finger[3]])
                    
                    if base_coords is None or tip_coords is None:
                        ratios.append(0.0)
                        continue
                        
                    straight_distance = np.linalg.norm(tip_coords[:2] - base_coords[:2])
                    
                    path_distance = 0
                    for i in range(len(finger) - 1):
                        p1_coords = self._get_landmark_coords(landmarks[finger[i]])
                        p2_coords = self._get_landmark_coords(landmarks[finger[i+1]])
                        
                        if p1_coords is not None and p2_coords is not None:
                            path_distance += np.linalg.norm(p2_coords[:2] - p1_coords[:2])
                    
                    curl_ratio = 1 - (straight_distance / path_distance) if path_distance > 0 else 0
                    ratios.append(max(0, min(1, curl_ratio)))
                    
            except (IndexError, KeyError, AttributeError):
                ratios.append(0.0)  # Default value if calculation fails
        
        return ratios
    
    def train_gesture_classifier(self, training_data, labels):
        """
        Train a custom gesture classifier
        
        Args:
            training_data: List of landmark sequences
            labels: Corresponding gesture labels
        """
        features = []
        valid_labels = []
        
        for i, landmarks in enumerate(training_data):
            try:
                if landmarks:  # Check if landmarks exist
                    # Handle different data formats
                    if isinstance(landmarks, list) and len(landmarks) > 0:
                        if isinstance(landmarks[0], list):  # Multiple hands
                            feature_vector = self.extract_features(landmarks[0])  # Use first hand
                        else:  # Single hand
                            feature_vector = self.extract_features(landmarks)
                    else:
                        feature_vector = np.zeros(84)
                    
                    features.append(feature_vector)
                    valid_labels.append(labels[i])
                else:
                    features.append(np.zeros(84))  # Zero vector for missing data
                    valid_labels.append(labels[i])
            except Exception as e:
                print(f"Error processing training sample {i}: {e}")
                features.append(np.zeros(84))
                valid_labels.append(labels[i])
        
        if not features:
            print("No valid training data found!")
            return 0.0
        
        X = np.array(features)
        y = np.array(valid_labels)
        
        # Check if we have enough data
        if len(X) < 2:
            print("Not enough training data!")
            return 0.0
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except ValueError:
            # If not enough samples for split, use all data for training
            X_train, X_test, y_train, y_test = X, X, y, y
        
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
        
        try:
            # Handle different landmark formats
            if isinstance(landmarks, list) and len(landmarks) > 0:
                if isinstance(landmarks[0], list):  # Multiple hands
                    features = self.extract_features(landmarks[0]).reshape(1, -1)
                else:  # Single hand
                    features = self.extract_features(landmarks).reshape(1, -1)
            else:
                return "Invalid landmark format", 0.0
            
            prediction = self.gesture_classifier.predict(features)[0]
            probabilities = self.gesture_classifier.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error in gesture prediction: {e}")
            return "Prediction error", 0.0
    
    def save_classifier(self, filename):
        """Save trained classifier to file"""
        if self.gesture_classifier:
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(self.gesture_classifier, f)
                print(f"Classifier saved successfully to {filename}")
            except Exception as e:
                print(f"Error saving classifier: {e}")
    
    def load_classifier(self, filename):
        """Load trained classifier from file"""
        try:
            with open(filename, 'rb') as f:
                self.gesture_classifier = pickle.load(f)
            print(f"Classifier loaded successfully from {filename}")
        except Exception as e:
            print(f"Error loading classifier: {e}")
    
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
        
        try:
            # Apply exponential smoothing
            smoothed_landmarks = []
            for i, hand in enumerate(landmarks):
                if i < len(prev_landmarks):
                    smoothed_hand = []
                    for j, landmark in enumerate(hand):
                        if j < len(prev_landmarks[i]):
                            prev_landmark = prev_landmarks[i][j]
                            
                            # Safe coordinate access
                            curr_coords = self._get_landmark_coords(landmark)
                            prev_coords = self._get_landmark_coords(prev_landmark)
                            
                            if curr_coords is not None and prev_coords is not None:
                                smoothed_coords = alpha * curr_coords + (1 - alpha) * prev_coords
                                smoothed_landmark = {
                                    'x': smoothed_coords[0],
                                    'y': smoothed_coords[1],
                                    'z': smoothed_coords[2]
                                }
                            else:
                                smoothed_landmark = landmark  # Use current if prev unavailable
                            
                            smoothed_hand.append(smoothed_landmark)
                        else:
                            smoothed_hand.append(landmark)
                    smoothed_landmarks.append(smoothed_hand)
                else:
                    smoothed_landmarks.append(hand)
            
            self.landmark_history.append(smoothed_landmarks)
            return smoothed_landmarks
            
        except Exception as e:
            print(f"Error in landmark smoothing: {e}")
            self.landmark_history.append(landmarks)
            return landmarks
    
    def detect_hand_motion(self, landmarks):
        """
        Detect hand motion patterns
        
        Args:
            landmarks: Current landmarks
            
        Returns:
            Motion characteristics
        """
        default_motion = {"velocity": 0, "direction": "stationary", "acceleration": 0}
        
        if len(self.landmark_history) < 5:
            return default_motion
        
        if not landmarks:
            return default_motion
        
        try:
            # Calculate wrist position over time
            wrist_positions = []
            for hist_landmarks in list(self.landmark_history)[-5:]:
                if hist_landmarks and len(hist_landmarks) > 0:
                    # Handle different landmark formats
                    wrist_landmark = None
                    if isinstance(hist_landmarks[0], list):  # Multiple hands
                        if len(hist_landmarks[0]) > 0:
                            wrist_landmark = hist_landmarks[0][0]  # First hand, wrist landmark
                    else:  # Single hand
                        wrist_landmark = hist_landmarks[0]  # Wrist landmark
                    
                    if wrist_landmark:
                        wrist_coords = self._get_landmark_coords(wrist_landmark)
                        if wrist_coords is not None:
                            wrist_positions.append(wrist_coords[:2])  # Only x, y
            
            if len(wrist_positions) < 2:
                return default_motion
            
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
                "velocity": float(avg_velocity),
                "direction": direction,
                "acceleration": float(acceleration)
            }
            
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return default_motion
    
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
        if not cap.isOpened():
            print("Error: Could not open camera")
            return [], []
            
        training_data = []
        labels = []
        current_gesture = input("Enter gesture name: ")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            try:
                # Process frame - check if base_estimator has process_image method
                if hasattr(self.base_estimator, 'process_image'):
                    annotated_frame, landmarks = self.base_estimator.process_image(frame)
                else:
                    # Fallback: try other common method names
                    if hasattr(self.base_estimator, 'process'):
                        result = self.base_estimator.process(frame)
                        if isinstance(result, tuple):
                            annotated_frame, landmarks = result
                        else:
                            annotated_frame = frame
                            landmarks = result
                    else:
                        print("Error: base_estimator doesn't have expected methods")
                        annotated_frame = frame
                        landmarks = None
                
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
                        
            except Exception as e:
                print(f"Error in training interface: {e}")
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
        
        return training_data, labels

# Example usage
def advanced_demo():
    """Demo of advanced features"""
    try:
        from hand_pose_estimate import HandPoseEstimator  # Import base estimator
        
        base_estimator = HandPoseEstimator()
        advanced_system = AdvancedHandPoseSystem(base_estimator)
        
        print("Advanced Hand Pose System ready!")
        print("Features available:")
        print("- Custom gesture training")
        print("- Motion detection")
        print("- Landmark smoothing")
        print("- Feature extraction")
        
        # Uncomment to start training interface
        # training_data, labels = advanced_system.create_training_interface()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure hand_pose_estimate.py is in the same directory")
    except Exception as e:
        print(f"Error initializing system: {e}")

if __name__ == "__main__":
    advanced_demo()