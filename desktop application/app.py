import threading
import time
import webview
import cv2
import numpy as np
import base64

try:
    import mediapipe as mp
except Exception:
    mp = None


class DroneController:
    def __init__(self, window):
        self.window = window
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        t = threading.Thread(target=self._run_cam, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def _classify_pose(self, hand_landmarks, handedness=None):
        # Simple heuristic-based classifier using MediaPipe landmarks.
        # Returns one of: 'hover','land','forward','back','left','right','ascend','descend'
        if hand_landmarks is None:
            return 'no_hand'

        lm = hand_landmarks.landmark
        # Indices for tips and pip joints
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        fingers_up = []
        for t, p in zip(tips, pips):
            fingers_up.append(lm[t].y < lm[p].y)

        # Thumb heuristic: compare tip x to ip x depending on hand
        thumb_up = False
        try:
            if handedness and 'Right' in handedness.classification[0].label:
                thumb_up = lm[4].x > lm[3].x
            else:
                thumb_up = lm[4].x < lm[3].x
        except Exception:
            thumb_up = lm[4].y < lm[3].y

        total_up = sum(1 for v in fingers_up if v) + (1 if thumb_up else 0)

        # Map heuristics to commands
        if total_up == 0:
            return 'land'
        if total_up == 5:
            return 'hover'
        # index only
        if fingers_up[0] and not any(fingers_up[1:]) and not thumb_up:
            return 'forward'
        # index+middle
        if fingers_up[0] and fingers_up[1] and not any(fingers_up[2:]) and not thumb_up:
            return 'back'
        # left/right using index tip x relative to wrist
        wrist_x = lm[0].x
        index_x = lm[8].x
        if fingers_up[0] and not any(fingers_up[1:]):
            if index_x < wrist_x - 0.04:
                return 'left'
            if index_x > wrist_x + 0.04:
                return 'right'
            return 'forward'
        # thumb up/down
        if thumb_up and not any(fingers_up):
            if lm[4].y < lm[2].y:
                return 'ascend'
            else:
                return 'descend'

        # fallback
        if fingers_up[0] and fingers_up[1]:
            return 'back'

        return 'hover'

    def _run_cam(self):
        if mp is None:
            print('MediaPipe not installed. Install requirements and try again.')
            return

        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        last_cmd = None
        last_send = 0
        last_frame_send = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            cmd = 'no_hand'
            if results.multi_hand_landmarks:
                # classify first detected hand
                cmd = self._classify_pose(results.multi_hand_landmarks[0], getattr(results, 'multi_handedness', None))

            # throttle JS command calls to ~10 Hz
            now = time.time()
            if cmd != last_cmd or now - last_send > 0.1:
                try:
                    js = f"window.onCommand && window.onCommand('{cmd}')"
                    self.window.evaluate_js(js)
                except Exception:
                    pass
                last_cmd = cmd
                last_send = now

            # Prepare annotated preview frame (send ~6-8 FPS)
            if now - last_frame_send > 0.14:
                try:
                    annotated = frame.copy()
                    if results.multi_hand_landmarks:
                        for hl in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(annotated, hl, mp_hands.HAND_CONNECTIONS)

                    # encode to JPEG
                    ret2, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ret2:
                        b64 = base64.b64encode(buf).decode('ascii')
                        data_url = 'data:image/jpeg;base64,' + b64
                        js = f"window.onFrame && window.onFrame('{data_url}')"
                        try:
                            self.window.evaluate_js(js)
                        except Exception:
                            pass
                except Exception:
                    pass
                last_frame_send = now

            # Small sleep to avoid maxing CPU
            time.sleep(0.02)

        cap.release()


def main():
    html_path = 'web/index.html'
    window = webview.create_window('HandPose Drone Control', html_path, width=900, height=700)

    controller = DroneController(window)

    # Start camera/controller after window is created
    def on_loaded():
        controller.start()

    # webview provides 'loaded' event via start callback argument
    webview.start(on_loaded)


if __name__ == '__main__':
    main()
