"""
Configuration file for AAU Racing Autonomous RC Car
All tunable parameters in one place for easy adjustment
"""

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_CONFIG = {
    'resolution_x': 854,
    'resolution_y': 480,
    'fps': 30,
    'crop_top_ratio': 0.25,      # Crop top 25% of image
    'crop_bottom_ratio': 0.75,   # Keep until 75% of image
}

# ============================================================================
# COLOR DETECTION THRESHOLDS (HSV)
# ============================================================================
COLOR_THRESHOLDS = {
    'yellow_lower': [22, 110, 120],
    'yellow_upper': [33, 255, 255],
    'blue_lower': [100, 120, 60],
    'blue_upper': [135, 255, 255],
}

# ============================================================================
# PERCEPTION SETTINGS
# ============================================================================
PERCEPTION_CONFIG = {
    'min_contour_area': 30,           # Minimum pixels for valid cone
    'neighbor_distance': 25,          # Max pixels apart for same cone
    'max_depth': 6.0,                 # Max depth in meters
    'z_smoothing_window': 5,          # Frames for depth smoothing
    'depth_offset': 180,              # Pixel offset for depth sampling
}

# Morphology settings
MORPHOLOGY_CONFIG = {
    'kernel_size': (2, 3),
    'erosion_iterations': 2,
    'dilation_iterations': 10,
}

# Spatial filter settings (RealSense)
SPATIAL_FILTER = {
    'magnitude': 5,
    'smooth_alpha': 1,
    'smooth_delta': 50,
    'holes_fill': 0,
}

# ============================================================================
# LOGIC/PATH PLANNING SETTINGS
# ============================================================================
LOGIC_CONFIG = {
    'lookahead_distance': 0.5,    # meters
    'wheelbase': 0.3,              # meters (track width offset)
    'max_cone_pairs': 4,           # Max pairs to consider
}

# ============================================================================
# CONTROL SETTINGS
# ============================================================================
CONTROL_CONFIG = {
    'default_speed': 32,           # PWM value (0-255)
    'max_steering_angle': 30.0,    # degrees
    'arduino_port': '/dev/ttyACM0',
    'arduino_baud': 115200,
    'command_delay': 0.01,         # seconds between commands
}

# ============================================================================
# SAFETY SETTINGS
# ============================================================================
SAFETY_CONFIG = {
    'max_steering_angle': 30.0,            # Maximum steering in degrees
    'no_cone_timeout': 5.0,                # Seconds before warning
    'max_consecutive_no_cones': 15,        # Frames before alarm
    'emergency_stop_speed': 0,             # Speed when emergency stop
    'safe_crawl_speed': 20,                # Slow speed when uncertain
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',               # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'stats_interval': 30,          # Print stats every N loops
    'performance_window': 30,      # Frames for FPS calculation
}

# ============================================================================
# SYSTEM SETTINGS
# ============================================================================
SYSTEM_CONFIG = {
    'max_init_retries': 3,
    'retry_delay': 2.0,            # seconds
    'camera_warmup_frames': 30,
}