import cv2
import mediapipe as mp
import time
import platform
import asyncio

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

def setup():
    # Ensure webcam is opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    return True

def update_loop():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand detection
    results = hands.process(frame_rgb)
    
    # Convert back to BGR for rendering
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    
    # Display the frame
    cv2.imshow('Hand Pose Estimation', frame)

async def main():
    if setup():
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