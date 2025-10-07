import cv2
import mediapipe as mp
import time
import platform
import asyncio
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,  # Focus on one hand for cleaner control
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)

# Initialize webcam
cap = cv2.VideoCapture(0)

class GestureRecognizer:
    def __init__(self):
        self.current_gesture = "none"
        self.gesture_confidence = 0.0
        self.hand_center = (0, 0)
        self.gesture_history = []
        self.history_size = 5
        
    def get_finger_positions(self, landmarks):
        """Extract finger tip and pip positions"""
        # Finger landmark indices (tip, pip)
        fingers = {
            'thumb': (4, 3),
            'index': (8, 6),
            'middle': (12, 10),
            'ring': (16, 14),
            'pinky': (20, 18)
        }
        
        finger_positions = {}
        for finger, (tip_idx, pip_idx) in fingers.items():
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            finger_positions[finger] = {
                'tip': (tip.x, tip.y, tip.z),
                'pip': (pip.x, pip.y, pip.z)
            }
        
        return finger_positions
    
    def is_finger_extended(self, finger_pos, finger_name):
        """Check if a finger is extended"""
        tip = finger_pos['tip']
        pip = finger_pos['pip']
        
        if finger_name == 'thumb':
            # Thumb extension is based on x-coordinate difference
            return abs(tip[0] - pip[0]) > 0.04
        else:
            # Other fingers: tip should be above pip
            return tip[1] < pip[1] - 0.02
    
    def count_extended_fingers(self, finger_positions):
        """Count how many fingers are extended"""
        extended = 0
        extended_fingers = []
        
        for finger, pos in finger_positions.items():
            if self.is_finger_extended(pos, finger):
                extended += 1
                extended_fingers.append(finger)
        
        return extended, extended_fingers
    
    def calculate_hand_center(self, landmarks):
        """Calculate the center point of the hand"""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return (center_x, center_y)
    
    def get_hand_orientation(self, landmarks):
        """Get hand orientation based on wrist and middle finger"""
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        # Calculate angle
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        angle = math.degrees(math.atan2(dy, dx))
        
        return angle
    
    def detect_pointing_direction(self, landmarks):
        """Detect pointing direction based on index finger"""
        index_tip = landmarks[8]
        index_mcp = landmarks[5]  # metacarpal
        
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y
        
        # Determine direction
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "up" if dy < 0 else "down"
    
    def is_fist(self, finger_positions):
        """Check if hand is making a fist"""
        extended_count, _ = self.count_extended_fingers(finger_positions)
        return extended_count == 0
    
    def is_open_palm(self, finger_positions):
        """Check if hand is open palm"""
        extended_count, _ = self.count_extended_fingers(finger_positions)
        return extended_count >= 4
    
    def is_peace_sign(self, finger_positions):
        """Check for peace sign (index and middle finger extended)"""
        extended_count, extended_fingers = self.count_extended_fingers(finger_positions)
        return (extended_count == 2 and 
                'index' in extended_fingers and 
                'middle' in extended_fingers)
    
    def is_thumbs_up(self, finger_positions):
        """Check for thumbs up"""
        extended_count, extended_fingers = self.count_extended_fingers(finger_positions)
        return (extended_count == 1 and 'thumb' in extended_fingers)
    
    def is_pointing(self, finger_positions):
        """Check if pointing with index finger"""
        extended_count, extended_fingers = self.count_extended_fingers(finger_positions)
        return (extended_count == 1 and 'index' in extended_fingers)
    
    def is_ok_sign(self, landmarks):
        """Check for OK sign (thumb and index finger touching)"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger tips
        distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + 
                           (thumb_tip.y - index_tip.y)**2)
        
        return distance < 0.05  # Threshold for "touching"
    
    def recognize_gesture(self, landmarks):
        """Main gesture recognition function"""
        finger_positions = self.get_finger_positions(landmarks)
        self.hand_center = self.calculate_hand_center(landmarks)
        
        gesture = "none"
        confidence = 0.0
        additional_info = {}
        
        # Check for various gestures
        if self.is_fist(finger_positions):
            gesture = "fist"
            confidence = 0.9
            additional_info["command"] = "land"
            
        elif self.is_open_palm(finger_positions):
            gesture = "open_palm"
            confidence = 0.9
            additional_info["command"] = "hover"
            
        elif self.is_peace_sign(finger_positions):
            gesture = "peace"
            confidence = 0.9
            additional_info["command"] = "takeoff"
            
        elif self.is_thumbs_up(finger_positions):
            gesture = "thumbs_up"
            confidence = 0.8
            additional_info["command"] = "ascend"
            
        elif self.is_pointing(finger_positions):
            direction = self.detect_pointing_direction(landmarks)
            gesture = f"pointing_{direction}"
            confidence = 0.8
            additional_info["command"] = f"move_{direction}"
            additional_info["direction"] = direction
            
        elif self.is_ok_sign(landmarks):
            gesture = "ok_sign"
            confidence = 0.7
            additional_info["command"] = "descend"
            
        else:
            # Count fingers for number-based commands
            extended_count, extended_fingers = self.count_extended_fingers(finger_positions)
            if extended_count > 0:
                gesture = f"{extended_count}_fingers"
                confidence = 0.6
                additional_info["command"] = f"speed_level_{extended_count}"
        
        # Update gesture history for stability
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Use most common gesture in recent history
        if len(self.gesture_history) >= 3:
            most_common = max(set(self.gesture_history), key=self.gesture_history.count)
            if self.gesture_history.count(most_common) >= 2:
                self.current_gesture = most_common
                self.gesture_confidence = confidence
                return most_common, confidence, additional_info
        
        self.current_gesture = gesture
        self.gesture_confidence = confidence
        return gesture, confidence, additional_info

# Initialize gesture recognizer
gesture_recognizer = GestureRecognizer()

def setup():
    # Ensure webcam is opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    return True

def draw_gesture_info(frame, gesture, confidence, additional_info, hand_center):
    """Draw gesture information on the frame"""
    height, width = frame.shape[:2]
    
    # Draw gesture text
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw command
    if "command" in additional_info:
        command = additional_info["command"]
        cv2.putText(frame, f"Command: {command}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Draw hand center
    if hand_center:
        center_pixel = (int(hand_center[0] * width), int(hand_center[1] * height))
        cv2.circle(frame, center_pixel, 10, (0, 0, 255), -1)
        
        # Draw position info
        cv2.putText(frame, f"Hand Center: ({hand_center[0]:.2f}, {hand_center[1]:.2f})", 
                    (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw control zones
    draw_control_zones(frame)

def draw_control_zones(frame):
    """Draw visual zones for different control areas"""
    height, width = frame.shape[:2]
    
    # Draw center zone (hover zone)
    center_x, center_y = width // 2, height // 2
    cv2.rectangle(frame, (center_x - 100, center_y - 75), 
                  (center_x + 100, center_y + 75), (255, 255, 0), 2)
    cv2.putText(frame, "HOVER ZONE", (center_x - 80, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw directional indicators
    # Up zone
    cv2.putText(frame, "UP", (center_x - 15, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Down zone
    cv2.putText(frame, "DOWN", (center_x - 25, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Left zone
    cv2.putText(frame, "LEFT", (20, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Right zone
    cv2.putText(frame, "RIGHT", (width - 80, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def process_drone_command(gesture, confidence, additional_info, hand_center):
    """Process the recognized gesture into drone commands"""
    if confidence < 0.6:  # Confidence threshold
        return
    
    command = additional_info.get("command", "none")
    
    # Here you would interface with your drone's SDK
    # For demonstration, we'll just print the commands
    print(f"DRONE COMMAND: {command}")
    
    if command == "takeoff":
        print("🚁 Initiating takeoff sequence...")
    elif command == "land":
        print("🛬 Landing drone...")
    elif command == "hover":
        print("⏸️  Hovering in place...")
    elif command == "ascend":
        print("⬆️  Ascending...")
    elif command == "descend":
        print("⬇️  Descending...")
    elif command.startswith("move_"):
        direction = additional_info.get("direction", "unknown")
        print(f"➡️  Moving {direction}...")
    elif command.startswith("speed_level_"):
        level = command.split("_")[-1]
        print(f"🚀 Setting speed level to {level}")
    
    # Position-based control using hand center
    if hand_center:
        # Map hand position to movement commands
        # Center of frame is (0.5, 0.5)
        x_offset = hand_center[0] - 0.5
        y_offset = hand_center[1] - 0.5
        
        # Only act if hand is significantly off-center
        if abs(x_offset) > 0.15 or abs(y_offset) > 0.15:
            if abs(x_offset) > abs(y_offset):
                direction = "right" if x_offset > 0 else "left"
                intensity = abs(x_offset) * 100
                print(f"📍 Position control: Move {direction} (intensity: {intensity:.1f}%)")
            else:
                direction = "forward" if y_offset < 0 else "backward"
                intensity = abs(y_offset) * 100
                print(f"📍 Position control: Move {direction} (intensity: {intensity:.1f}%)")

def update_loop():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand detection
    results = hands.process(frame_rgb)
    
    # Convert back to BGR for rendering
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    gesture = "none"
    confidence = 0.0
    additional_info = {}
    hand_center = None
    
    # Draw hand landmarks and recognize gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Recognize gesture
            gesture, confidence, additional_info = gesture_recognizer.recognize_gesture(hand_landmarks.landmark)
            hand_center = gesture_recognizer.hand_center
            
            # Process drone command
            process_drone_command(gesture, confidence, additional_info, hand_center)
    
    # Draw gesture information
    draw_gesture_info(frame, gesture, confidence, additional_info, hand_center)
    
    # Display the frame
    cv2.imshow('Hand Gesture Drone Control', frame)

async def main():
    if setup():
        print("Hand Gesture Drone Control Started!")
        print("Gestures:")
        print("✋ Open Palm - Hover")
        print("✊ Fist - Land")
        print("✌️  Peace Sign - Takeoff")
        print("👍 Thumbs Up - Ascend")
        print("👌 OK Sign - Descend")
        print("👉 Pointing - Move in direction")
        print("🖐️  1-5 Fingers - Speed levels")
        print("📍 Hand position - Positional control")
        print("\nPress 'q' to quit")
        
        while True:
            update_loop()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
            await asyncio.sleep(1.0 / 30)  # Control frame rate (~30 FPS)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())