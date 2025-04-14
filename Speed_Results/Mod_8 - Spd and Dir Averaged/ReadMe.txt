This is a modification of speed_estimation.py file, which can be found in ultralytics/solutions folder. Copy the code below into the current speed_estimation.py, then call on speed estimation as directed in YOLOxSpeedEstimation or in: https://docs.ultralytics.com/guides/speed-estimation/#speedestimator-arguments.

With this, when speed estimation is called, the display shows the speed of detected objects in pixels per frame and the direction of the object as an angle in degrees. Both are calculated from the centroids of the bounding boxes, considering the movement between a certain number of frames and then averaged over a number of recorded speeds and directions. 

The modification changes most of the 'process' function in speed_estimation.py, as well as adds a few new dictionaries to __init__ and a new function called get_direction_and_angle.
The modifications in the process function are the following:
    - Extract the centre of the bounding box as x and y coordinates and store them as the current_position
    - in -- Speed Estimation :
        - Get pos_start and pos_end as the first and last positions of an object between a certain number             of frames (25)
        - Compute the displacement between the two positions and convert it into speed in pixels per frame,           by dividing the displacement with the number of frames
        - Take the average of the last ten 'speeds' and store it
    - in --- Direction Estimation:
        - Same process, the angle is calculated using the function get_direction_and_angle

The code for speed_estimation.py:

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from time import time

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

from shapely.geometry import Point, Polygon

class SpeedEstimator(BaseSolution):
    """
    A class to estimate the speed of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for estimating object speeds using
    tracking data in video streams.

    Attributes:
        spd (Dict[int, float]): Dictionary storing speed data for tracked objects.
        trkd_ids (List[int]): List of tracked object IDs that have already been speed-estimated.
        trk_pt (Dict[int, float]): Dictionary storing previous timestamps for tracked objects.
        trk_pp (Dict[int, Tuple[float, float]]): Dictionary storing previous positions for tracked objects.
        region (List[Tuple[int, int]]): List of points defining the speed estimation region.
        track_line (List[Tuple[float, float]]): List of points representing the object's track.
        r_s (LineString): LineString object representing the speed estimation region.

    Methods:
        initialize_region: Initializes the speed estimation region.
        process: Processes input frames to estimate object speeds.
        store_tracking_history: Stores the tracking history for an object.
        extract_tracks: Extracts tracks from the current frame.
        display_output: Displays the output with annotations.

    Examples:
        >>> estimator = SpeedEstimator()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = estimator.process(frame)
        >>> cv2.imshow("Speed Estimation", results.plot_im)
    """

    def __init__(self, **kwargs):
        """
        Initialize the SpeedEstimator object with speed estimation parameters and data structures.

        Args:
            **kwargs (Any): Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self.initialize_region()  # Initialize speed region

        self.spd = {}  # Dictionary for speed data
        self.trkd_ids = []  # List for already speed-estimated and tracked IDs
        self.trk_pt = {}  # Dictionary for tracks' previous timestamps
        self.trk_pp = {}  # Dictionary for tracks' previous positions
        self.angle_history = {}  # Dictionary to store angle history for averaging
        self.position_history = {}  # Dictionary to store position history for averaging
        self.spd_history = {}  # Dictionary to store speed history for averaging
        self.direction_history = {}  # For storing angles over time


    def get_direction_and_angle(self, prev_pos, curr_pos):
        """Calculate direction angle and arrow symbol between two points.
        
        Args:
            prev_pos : the position of the bounding box centroid from the previous frame
            curr_pos : the position of the bounding box centroid from the current frame 
        
        Returns:
            "→" : the arrow of the direction 
            angle_deg : the angle of the direction in degrees
        """
        dx, dy = curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]
        angle_rad = np.arctan2(-dy, dx)  # Negative y since image origin is top-left
        angle_deg = np.degrees(angle_rad) % 360

        # 8-directional mapping
        directions = {
            (337.5, 360): "→",
            (0, 22.5): "→",
            (22.5, 67.5): "↘",
            (67.5, 112.5): "↓",
            (112.5, 157.5): "↙",
            (157.5, 202.5): "←",
            (202.5, 247.5): "↖",
            (247.5, 292.5): "↑",
            (292.5, 337.5): "↗",
        }

        for (low, high), arrow in directions.items():
            if low <= angle_deg < high:
                return arrow, angle_deg
        return "→", angle_deg  # Default case

    def polygon_contains_point(self, point, region):
        """Check if a point is inside the defined region."""
        return Polygon(region).contains(Point(point))

    def process(self, im0):
        """
        Process an input frame to estimate object speeds based on tracking data.

        Args:
            im0 (np.ndarray): Input image for processing with shape (H, W, C) for RGB images.

        Returns:
            (SolutionResults): Contains processed image `plot_im` and `total_tracks` (number of tracked objects).

        Examples:
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = estimator.process(image)
        """
        self.extract_tracks(im0)  # Extract tracks
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        # Draw speed estimation region
        annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # Store track history

            # Initialize tracking data for new objects
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = np.array(self.track_line[-1], dtype=np.float32)  # Ensure float precision

            ### Modified code for updating the speed of objects in region, remembers the last 50 frames and gives their average, but also now the pixel per frame include decimals
            
            
            angle, arrow = None, ""

            # Check if the object is within the speed estimation region
            if self.polygon_contains_point(self.track_line[-1], self.region):  # Check if object is in region
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    # Get bounding box center instead of track_line
                    x_center, y_center, w, h = box[:4].tolist()  # Extract first four values
                    current_position = np.array([x_center, y_center], dtype=np.float32)
                    previous_position = self.trk_pp.get(track_id, current_position)  # Default to current position if not found

                    # Print positions for debugging
                    #print(f"Track ID {track_id}: Current Position: {current_position}, Previous Position: {previous_position}")

                    # Compute displacement
                    displacement = np.linalg.norm(current_position - previous_position)

                    # --- Direction & Speed Estimation over 10 Frames ---
                    # --- Initialize tracking dicts ---
                    if track_id not in self.position_history:
                        self.position_history[track_id] = []
                    if track_id not in self.spd_history:
                        self.spd_history[track_id] = []
                    if track_id not in self.angle_history:
                        self.angle_history[track_id] = []
                    
                    # --- Append position to both position and angle history ---
                    self.position_history[track_id].append(current_position)
                    self.angle_history[track_id].append(current_position)
                    
                    # Trim position history to 10
                    if len(self.position_history[track_id]) > 25:
                        self.position_history[track_id].pop(0)
                    
                    # Trim angle history to 100 (if needed)
                    if len(self.angle_history[track_id]) > 250:
                        self.angle_history[track_id].pop(0)
                    
                    # --- Speed Estimation (10-frame gap, 10-speed average) ---
                    if len(self.position_history[track_id]) == 25:
                        pos_start = self.position_history[track_id][0]
                        pos_end = self.position_history[track_id][-1]
                    
                        displacement = np.linalg.norm(pos_end - pos_start) / 25  # px/frame
                        self.spd_history[track_id].append(displacement)
                    
                        # Keep last 10 speed measurements
                        if len(self.spd_history[track_id]) > 10:
                            self.spd_history[track_id].pop(0)
                    
                        # Average speed over last 10 values
                        self.spd[track_id] = np.mean(self.spd_history[track_id])

                    # Compute direction
                    arrow, angle = self.get_direction_and_angle(previous_position, current_position)
                    
                    # --- Direction Estimation over 25 frames, then averaging over last 10 angles ---
                    arrow, angle = "", None
                    if len(self.angle_history[track_id]) >= 25:
                        dir_start = self.angle_history[track_id][-25]
                        dir_end = self.angle_history[track_id][-1]
                        arrow, raw_angle = self.get_direction_and_angle(dir_start, dir_end)
                    
                        # Initialize angle history if needed
                        if track_id not in self.direction_history:
                            self.direction_history[track_id] = []
                    
                        # Store latest angle
                        self.direction_history[track_id].append(raw_angle)
                    
                        # Keep only the last 10 angles
                        if len(self.direction_history[track_id]) > 10:
                            self.direction_history[track_id].pop(0)
                    
                        # Compute average angle
                        angle = np.mean(self.direction_history[track_id])
                    
                        print(f"Track ID {track_id}: Direction: {arrow}, Angle: {angle:.1f}")
                    else:
                        print(f"Track ID {track_id}: Direction: not available")


            # Update tracking data for next frame
            #self.trk_pt[track_id] = time()
            
            # --- Display only if speed is above threshold ---
            speed_val = self.spd.get(track_id, 0)
            if speed_val >= 0.2:
                if angle is not None:
                    print_label = f"ID {track_id}: {speed_val:.2f} px/frame, Angle: {angle:.1f}"
                else:
                    print_label = f"ID {track_id}: {speed_val:.2f} px/frame"
            
                annotator.box_label(box, label=print_label, color=colors(track_id, True))

            ### End of Mod
        
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return results with processed image and tracking summary
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
