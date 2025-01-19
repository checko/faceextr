import cv2
import os
import argparse

def extract_face_frame(video_path, output_path, start_minutes=10, frame_skip=10):
    """
    Extract a frame from video where a face occupies approximately half the frame.
    Starts from specified number of minutes into the video.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path to save the output JPEG
        start_minutes (int): Minutes to skip from start of video
        frame_skip (int): Number of frames to skip between processing
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
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
    print("Press 'q' to quit, 's' to skip to next frame")
    
    frame_count = start_frame
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Skip frames according to frame_skip parameter
        if (frame_count - start_frame) % frame_skip != 0:
            continue
            
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around detected faces and show face ratio
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            face_area = w * h
            face_ratio = face_area / frame_area
            
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display face ratio
            ratio_text = f"Face ratio: {face_ratio:.2%}"
            cv2.putText(display_frame, ratio_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if face occupies approximately half the frame (40-60% range)
            if 0.2 <= face_ratio :
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
        
        # Display frame number and progress
        current_minute = frame_count/fps/60
        progress = ((frame_count - start_frame) / (total_frames - start_frame)) * 100
        progress_text = f"Time: {current_minute:.1f}min Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
        cv2.putText(display_frame, progress_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Processing Frame', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nProcessing cancelled by user")
            break
        elif key == ord('s'):
            continue
    
    video.release()
    cv2.destroyAllWindows()
    print("\nNo suitable frame found with face occupying half the frame")
    return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract a frame with a face from a video file')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('-o', '--output', 
                        default='face_frame.jpg',
                        help='Path for the output JPEG file (default: face_frame.jpg)')
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
    extract_face_frame(args.input_video, args.output, args.start_minutes, args.skip)

if __name__ == "__main__":
    main()
