import cv2
import os
import argparse
import mediapipe as mp

def get_output_filename(input_path):
    """
    Generate output filename by changing the extension to .jpg
    
    Args:
        input_path (str): Path to input video file
    
    Returns:
        str: Output path with .jpg extension
    """
    # Get the directory and filename without extension
    directory = os.path.dirname(input_path)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create output path with .jpg extension
    return os.path.join(directory, f"{filename}.jpg")

def extract_face_frame(video_path, start_minutes=10, frame_skip=10):
    """
    Extract a frame from video where a face occupies approximately half the frame.
    Starts from specified number of minutes into the video.
    
    Args:
        video_path (str): Path to input video file
        start_minutes (int): Minutes to skip from start of video
        frame_skip (int): Number of frames to skip between processing
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Generate output path from input filename
    output_path = get_output_filename(video_path)
    
    # Replace face cascade with MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 1 for full range detection up to 5 meters
        min_detection_confidence=0.7  # Increased confidence threshold
    )
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Create window for display
    cv2.namedWindow('Processing Frame', cv2.WINDOW_NORMAL)
    
    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height
    
    # Calculate start frame (10 minutes from start)
    start_frame = int(start_minutes * 60 * fps)
    if start_frame >= total_frames:
        print(f"Error: Video is shorter than {start_minutes} minutes")
        return False
    
    # Seek to start position
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Video Info: {frame_width}x{frame_height} at {fps}fps")
    print(f"Total frames: {total_frames}")
    print(f"Starting from frame {start_frame} ({start_minutes} minutes in)")
    print(f"Processing every {frame_skip}th frame...")
    print(f"Output will be saved as: {output_path}")
    print("Press 'q' to quit, 's' to skip to next frame, 'p' to pause/unpause")
    
    frame_count = start_frame
    paused = False
    while True:
        if not paused:
            ret, frame = video.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames according to frame_skip parameter
            if (frame_count - start_frame) % frame_skip != 0:
                continue
        
        # Process frame (whether paused or not)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        display_frame = frame.copy()
        
        # Add pause indicator
        if paused:
            cv2.putText(display_frame, "PAUSED", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates and detection confidence
                bbox = detection.location_data.relative_bounding_box
                detection_confidence = detection.score[0]  # Overall face detection confidence
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)
                
                face_area = w * h
                face_ratio = face_area / frame_area
                
                # Check if both eyes are detected
                left_eye = detection.location_data.relative_keypoints[0]  # Left eye keypoint
                right_eye = detection.location_data.relative_keypoints[1]  # Right eye keypoint
                
                # Convert eye coordinates to pixel values
                left_eye_x = int(left_eye.x * frame_width)
                left_eye_y = int(left_eye.y * frame_height)
                right_eye_x = int(right_eye.x * frame_width)
                right_eye_y = int(right_eye.y * frame_height)
                
                # Draw eyes on frame
                cv2.circle(display_frame, (left_eye_x, left_eye_y), 3, (0, 255, 255), -1)
                cv2.circle(display_frame, (right_eye_x, right_eye_y), 3, (0, 255, 255), -1)
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display face ratio
                ratio_text = f"Face ratio: {face_ratio:.2%}"
                cv2.putText(display_frame, ratio_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check if face occupies approximately half the frame AND detection confidence is high
                if 0.03 <= face_ratio and detection_confidence > 0.6:
                    # Save the frame as JPEG
                    cv2.imwrite(output_path, frame)
                    print(f"\nSuccessfully saved frame {frame_count} to {output_path}")
                    print(f"Face ratio: {face_ratio:.2%}")
                    print(f"Time position: {frame_count/fps/60:.2f} minutes")
                    
                    # Display "SAVED" on frame
                    cv2.putText(display_frame, "SAVED!", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Processing Frame', display_frame)
                    cv2.waitKey(1000)  # Show the saved frame for 1 second
                    
                    video.release()
                    cv2.destroyAllWindows()
                    return True
                
                # If paused, display detailed detection info
                if paused:
                    print("\nDetection Details:")
                    print(f"Face ratio: {face_ratio:.2%}")
                    print(f"Detection confidence: {detection_confidence:.2%}")
                    print(f"Face bounding box: x={x}, y={y}, w={w}, h={h}")
                    print(f"Left eye position: ({left_eye_x}, {left_eye_y})")
                    print(f"Right eye position: ({right_eye_x}, {right_eye_y})")
                    print(f"Current frame: {frame_count}/{total_frames}")
                    print(f"Time position: {frame_count/fps/60:.2f} minutes")
                    print("Press 'p' to resume processing")
        
        # Display frame number and progress
        current_minute = frame_count/fps/60
        progress = ((frame_count - start_frame) / (total_frames - start_frame)) * 100
        progress_text = f"Time: {current_minute:.1f}min Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
        cv2.putText(display_frame, progress_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Processing Frame', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            print("\nProcessing cancelled by user")
            break
        elif key == ord('s'):
            continue
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("\n=== Processing PAUSED ===")
            else:
                print("\n=== Processing RESUMED ===")
    
    video.release()
    cv2.destroyAllWindows()
    print("\nNo suitable frame found with face occupying half the frame")
    return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract a frame with a face from a video file')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('-s', '--skip', 
                        type=int,
                        default=10,
                        help='Number of frames to skip between processing (default: 10)')
    parser.add_argument('-m', '--start-minutes', 
                        type=int,
                        default=10,
                        help='Minutes to skip from start of video (default: 10)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Video file '{args.input_video}' not found")
        return
    
    # Process the video
    extract_face_frame(args.input_video, args.start_minutes, args.skip)

if __name__ == "__main__":
    main()
