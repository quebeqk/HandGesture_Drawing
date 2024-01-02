# Hand Gesture Drawing Application

This Python script uses OpenCV and MediaPipe to create a hand gesture drawing application. It allows users to draw on the screen using their hand gestures captured through a webcam.

## Features

- Tracks hand landmarks using MediaPipe to detect finger positions and gestures.
- Supports drawing lines, rectangles, circles, and erasing on the screen.
- Automatically switches drawing colors based on finger proximity for a more intuitive user experience.
- Provides a full-screen drawing interface for a more immersive drawing experience.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe

## Installation

1. Clone the repository:

2. Install the required dependencies using pip:

## Usage

1. Navigate to the project directory:

2. Run the script:

3. Use your hand gestures to draw on the screen. The application tracks your hand movements and allows you to draw in real-time.

## Controls

- When your middle and index finger come closer, you can draw and navigate to the drawing area to select and draw lines, rectangles, and circles on the screen.
- As soon as the index finger and middle finger are not closer, drawing is disabled.
- Use your thumb and pinky finger to switch between different drawing colors.
- Use the eraser tool by bringing your middle finger close to the drawing area and then use the index and middle finger close to each other to erase.
  - In eraser mode, if the index finger and thumb come closer, the entire screen is cleared.
- Press the ESC key to exit the application.
