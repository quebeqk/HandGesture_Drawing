#Libraries
import cv2
import mediapipe as mp
import numpy as np

#Access camera
cap=cv2.VideoCapture(0)

# Camera frame resolution(H*w*color channel RGB)
frame_shape = (480, 640, 3)

# Load Model
#creates an instance of the hand tracking model from the Mediapipe library
hands = mp.solutions.hands
# initializes the hand tracking model
hand_landmark = hands.Hands(max_num_hands=1)
#imports the drawing utilities to draw landmarks and connections on the detected hands.
draw = mp.solutions.drawing_utils


curr_tool = 'draw'
start_point = None
prevxy = None
#creates a NumPy array mask filled with zeros, with the shape defined by frame_shape. 
#This array will be used to permanently draw things on the canvas.
mask = np.zeros(frame_shape,dtype='uint8') 

drawing_color = (125, 100, 140) #default drawing color(BGR)
draw_thickness = 4  #thickness of the drawing

# Define a list of colors(BGR)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
current_color_index = 0  # Initialize the current color index

# Read toolbar image
tools = cv2.imread("tool.png")
#converts the data type of the tools image to 'uint8'.
tools = tools.astype('uint8')

midCol = 640 // 2
max_row = 50
min_col = midCol-125
max_col = midCol+125

#Some utility functions
#Check if the distance between 2 points is less than 60px
def get_is_clicked(point1,point2):
    (x1,y1)=point1
    (x2,y2)=point2

    dis=(x1-x2)**2+(y1-y2)**2
    dis=np.sqrt(dis)
    if dis<30:
        return True
    else:
        return False

#Return tool based on column location
def get_Tool(point,prev_tool):
    (x,y)=point

    if x>min_col and x<max_col and y<max_row:
        if x < min_col:
            return
        elif x < 50 + min_col:
            curr_tool = "line"
        elif x<100 + min_col:
            curr_tool = "rectangle"
        elif x < 150 + min_col:
            curr_tool ="draw"
        elif x<200 + min_col:
            curr_tool = "circle"
        else:
            curr_tool = "erase"
        return curr_tool
    else:
        return prev_tool


while True:
    #Read/Show frame's from camera
    success,frame=cap.read()  #BGR
    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    #Preprocess image ie. converting the BGR image to RGB
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Processing the frame with MediaPipe Hands to detect hand landmarks
    results=hand_landmark.process(rgb)

    if results.multi_hand_landmarks:
        for all_landmarks in results.multi_hand_landmarks:
            draw.draw_landmarks(frame,all_landmarks,hands.HAND_CONNECTIONS)
            
            # index finger location
            x=int(all_landmarks.landmark[8].x*frame_shape[1])
            y=int(all_landmarks.landmark[8].y*frame_shape[0])

            # Middle finger location
            middle_x =int(all_landmarks.landmark[12].x * frame_shape[1])
            middle_y = int(all_landmarks.landmark[12].y * frame_shape[0])

            # Get thumb and index finger locations
            thumb_x = int(all_landmarks.landmark[4].x * frame_shape[1])
            thumb_y = int(all_landmarks.landmark[4].y * frame_shape[0])

            #pinky finger location
            pinky_x = int(all_landmarks.landmark[20].x * frame_shape[1])
            pinky_y = int(all_landmarks.landmark[20].y * frame_shape[0])

            is_clicked=get_is_clicked((x,y),(middle_x,middle_y))
            curr_tool=get_Tool((x,y),curr_tool)

            if curr_tool == "erase" and ((thumb_x - x) ** 2 + (thumb_y - y) ** 2) ** 0.5<30:
                # Clear the canvas by filling it with a blank color (white)
                mask = np.zeros(frame_shape, dtype='uint8')

            # Check if the pinky finger is close to the thumb
            if thumb_x != 0 and thumb_y != 0 and pinky_x != 0 and pinky_y != 0:
                distance = ((thumb_x - pinky_x) ** 2 + (thumb_y - pinky_y) ** 2) ** 0.5
                if distance < 30:
                    # Switch to the next color
                    current_color_index = (current_color_index + 1) % len(colors)
                    drawing_color = colors[current_color_index]


            # Select tool and draw for that
            if curr_tool == 'line':
                    if is_clicked and start_point is None:
                        start_point = (x, y)
                    elif is_clicked:
                        cv2.line(frame, start_point, (x, y), drawing_color, draw_thickness)
                    elif is_clicked == False and start_point:
                        cv2.line(mask, start_point, (x, y), drawing_color, draw_thickness)
                        start_point = None

            if curr_tool == 'draw':
                # Connect previous and current index finger locations
                if is_clicked and prevxy!=None:
                    cv2.line(mask, prevxy, (x, y), drawing_color, draw_thickness)

            elif curr_tool == 'rectangle':
                if is_clicked and start_point==None:
                    start_point = (x, y)
                
                elif is_clicked:
                    cv2.rectangle(frame, start_point, (x, y), drawing_color, draw_thickness)
                
                elif is_clicked==False and start_point:
                    cv2.rectangle(mask, start_point, (x, y), drawing_color, draw_thickness)
                    start_point=None
            
            elif curr_tool=='circle':
                if is_clicked and start_point==None:
                    start_point = (x, y)
                
                if start_point:
                    rad = int(((start_point[0]-x)**2 + (start_point[1]-y)**2)**0.5)
                if is_clicked:
                    cv2.circle(frame, start_point, rad, drawing_color, draw_thickness)
                
                if is_clicked==False and start_point:
                    cv2.circle(mask, start_point, rad, drawing_color, draw_thickness)
                
                    start_point=None
            
            elif curr_tool == "erase":
                cv2.circle(frame, (x, y), 30, (0,0,0), -1) # -1 means fill
                if is_clicked:
                    cv2.circle(mask, (x, y), 30, 0, -1)

            # Connect previous and current index finger locations
            # if prevxy is not None:
            #     # Draw line on the mask
            #     cv2.line(mask, prevxy, (x, y), drawing_color, draw_thickness)
            # keeps track of the previous finger position for drawing continuous lines.
            prevxy = (x, y) 

    # Merge resized_frame and resized_mask
    frame = np.where(mask, mask,frame)

    frame[0:max_row, min_col:max_col] = tools

   # Create a full-screen window
    cv2.namedWindow('live', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('live', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display the merged frame
    cv2.imshow('live', frame)

    # Check for ESC key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the VideoCapture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
