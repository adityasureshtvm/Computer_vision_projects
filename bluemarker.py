#drawing
import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not available")
    exit()

# Create canvas
canvas = None

# Default draw settings
draw_color = (255, 0, 0)  # Blue
brush_thickness = 5

# HSV range for blue color tracking (change if needed)
lower_color = np.array([100, 150, 0])
upper_color = np.array([140, 255, 255])

points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            center = (x + w//2, y + h//2)
            points.append(center)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Draw trail on canvas
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            cv2.line(canvas, points[i - 1], points[i], draw_color, brush_thickness)

    # Overlay canvas on webcam feed
    combined = cv2.add(frame, canvas)

    # Display instructions
    cv2.putText(combined, "Press: r/g/b/y = Color | e = Eraser | +/- = Thickness", (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(combined, "c = Clear | s = Save | q = Quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Show output
    cv2.imshow("Virtual Painter", combined)
    cv2.imshow("Color Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    # Handle key events
    if key == ord('r'):
        draw_color = (0, 0, 255)
    elif key == ord('g'):
        draw_color = (0, 255, 0)
    elif key == ord('b'):
        draw_color = (255, 0, 0)
    elif key == ord('y'):
        draw_color = (0, 255, 255)
    elif key == ord('e'):
        draw_color = (0, 0, 0)  # Eraser (black)
    elif key == ord('+') or key == ord('='):
        brush_thickness += 1
    elif key == ord('-') and brush_thickness > 1:
        brush_thickness -= 1
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        points = []
    elif key == ord('s'):
        cv2.imwrite("virtual_painting.png", canvas)
        print("Saved as virtual_painting.png")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
