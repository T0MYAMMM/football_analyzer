from __future__ import annotations
import cv2
import torch
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as pls
import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageOps
import supervision as sv
from typing import Tuple, List
import torchvision.transforms as T
import torchvision.transforms.functional as F

from supervision.annotators.utils import ColorLookup, resolve_color
#from supervision.draw.color import color

# Import custom utilities
#from utils.utils_tracking import get_center_of_bbox, get_bbox_width, get_foot_position
from examples.soccer.utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation, keypoint_aux_world_coords_2D, keypoint_world_coords_2D, line_world_coords_3D
from examples.soccer.utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict

device = 'cuda:0'

# geometry utilities
@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x

    @property
    def min_y(self) -> float:
        return self.y

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def max_y(self) -> float:
        return self.y + self.height

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )

    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))
            
def inference(frame, model_kp, model_l, kp_thresh, line_thresh, h, w):
    #w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam = FramebyFrameCalib(iwidth = w, iheight = h, denormalize = True)

    transform_resize = T.Resize((540, 960))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(Image.fromarray(frame_rgb)).float().unsqueeze(0)
    
    if frame_tensor.size()[-1] != 960:
        frame_tensor = transform_resize(frame_tensor)
        
    frame_tensor = frame_tensor.to("cuda:0")

    with torch.no_grad():
        heatmaps = model_kp(frame_tensor)
        heatmaps_l = model_l(frame_tensor)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
    kp_dict = coords_to_dict(kp_coords, threshold = kp_thresh)
    lines_dict = coords_to_dict(line_coords, threshold = line_thresh)
    final_dict = complete_keypoints(kp_dict, lines_dict, w = frame_tensor.size(3), h = frame_tensor.size(2), normalize = True)

    cam.update(final_dict[0])
    results = cam.heuristic_voting()
    
    return results

def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P

def get_homography_from_projection(P, inverse=False):
        H = P[:, [0, 1, 3]]  # (3, 3)
        H = H / H[-1, -1]  # normalize homography

        if inverse:
            H_inv = np.linalg.inv(H)
            return H_inv / H_inv[-1, -1]
        else:
            return H

def inverse_projection_matrix(P):
    H_inv = get_homography_from_projection(P, inverse=True)
    return H_inv

def get_center_of_bbox(rect):
    print(f'rect: {rect}')
    x_center = rect.x + rect.width / 2
    y_center = rect.y + rect.height / 2
    return x_center, y_center

def project_to_field_map(bbox_centers: np.ndarray, P_inv: np.ndarray) -> np.ndarray:
    """
    Projects bounding box centers to the field map using the inverse projection matrix.

    Args:
        bbox_centers (np.ndarray): An array of shape (n, 2) containing the [x, y] coordinates of the bounding box centers.
        P_inv (np.ndarray): The inverse projection matrix.

    Returns:
        np.ndarray: An array of shape (n, 2) containing the projected [x, y] coordinates on the field map.
    """
    # Convert the bbox centers to homogeneous coordinates
    homogeneous_coords = np.hstack((bbox_centers, np.ones((bbox_centers.shape[0], 1))))

    # Apply the inverse projection matrix
    world_coords = P_inv @ homogeneous_coords.T

    # Normalize the homogeneous coordinates
    world_coords /= world_coords[-1, :]

    # Return the x and y coordinates
    return world_coords[:-1, :].T

def map_bbox_to_field(detections: List[sv.Detection], P_inv):
    field_coords = []
    bbox_centers = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    coords_list = project_to_field_map(bbox_centers, P_inv)
    for x, y in coords_list:
        field_coords.append((x,y))
    return field_coords

def visualize_field_and_players(detections: sv.Detections, color_lookup: np.ndarray, H_inv) -> np.ndarray:
    keypoint_world_coords_2D = [[x - 52.5, y - 34] for x, y in [
        [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
        [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34], [105., 30.34],
        [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66], [0., 43.16], [5.5, 43.16],
        [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16], [88.5, 54.16], [105., 54.16], [0., 68.],
        [52.5, 68.], [105., 68.], [16.5, 26.68], [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15],
        [88.5, 41.31], [19.99, 32.29], [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46],
        [61.31, 36.46], [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
        [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.], [88.5, 34.], [94., 34.]
    ]]

    line_world_coords_3D = [[[x1 - 52.5, y1 - 34, z1], [x2 - 52.5, y2 - 34, z2]] for [[x1, y1, z1], [x2, y2, z2]] in [
        [[0., 54.16, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [0., 13.84, 0.]],
        [[88.5, 54.16, 0.], [105., 54.16, 0.]], [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], [[88.5, 13.84, 0.], [105., 13.84, 0.]],
        [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]], [[0., 30.34, 0.], [0., 30.34, -2.44]],
        [[105., 37.66, -2.44], [105., 30.34, -2.44]], [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],
        [[52.5, 0., 0.], [52.5, 68, 0.]], [[0., 68., 0.], [105., 68., 0.]], [[0., 0., 0.], [0., 68., 0.]], [[105., 0., 0.], [105., 68., 0.]],
        [[0., 0., 0.], [105., 0., 0.]], [[0., 43.16, 0.], [5.5, 43.16, 0.]], [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
        [[5.5, 24.84, 0.], [0., 24.84, 0.]], [[99.5, 43.16, 0.], [105., 43.16, 0.]], [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
        [[99.5, 24.84, 0.], [105., 24.84, 0.]]
    ]]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('lightgreen')

    # Draw the field lines and ellipses (arcs)
    for line in line_world_coords_3D:
        (x1, y1, z1), (x2, y2, z2) = line
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=3)

    # Draw the center circle
    center_circle = plt.Circle((0, 0), 9.15, color="white", fill=False, linewidth=3)
    ax.add_patch(center_circle)

    # Draw the penalty box arcs
    penalty_arc_left = pls.Arc((-52.5 + 11, 0), width=18.3, height=18.3, angle=-90, theta1=37, theta2=143, color='white', linewidth=3)
    penalty_arc_right = pls.Arc((52.5 - 11, 0), width=18.3, height=18.3, angle=-90, theta1=217, theta2=323, color='white', linewidth=3)
    ax.add_patch(penalty_arc_left)
    ax.add_patch(penalty_arc_right)

    COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

    # Draw player coordinates
    bbox_centers = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    coords_list = project_to_field_map(bbox_centers, H_inv)

    for detection_idx, (x, y) in enumerate(coords_list):
        color = resolve_color(
            color=sv.ColorPalette.from_hex(COLORS),
            detections=detections,
            detection_idx=detection_idx,
            color_lookup=color_lookup
        )
        ax.scatter(x, -y, color=color.as_hex(), zorder=5, s=80)

    # Draw the key points
    keypoints_x, keypoints_y = zip(*keypoint_world_coords_2D)
    ax.scatter(keypoints_x, keypoints_y, color='red', label='Key Points', s=30, alpha=0.3, zorder=10)

    ax.set_xlim(-52.5, 52.5)
    ax.set_ylim(-34, 34)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_title("Football Field Key Points and Lines")
    ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    img = ImageOps.exif_transpose(img)  # Correct the orientation if needed
    img_np = np.array(img.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_np

def project(frame, P, show_field_mode: str ='both'):
    """
    Project keypoints and lines onto the frame based on the projection matrix P.

    Args:
        frame (np.ndarray): The frame to draw on.
        P (np.ndarray): The projection matrix.
        show_mode (str): The mode to show ('keypoints', 'lines', 'both').

    Returns:
        np.ndarray: The frame with keypoints and/or lines drawn.
    """
    # Draw lines and ellipses if show_mode is 'lines' or 'both'
    if show_field_mode in ['lines', 'both']:
        for line in line_world_coords_3D:
            w1 = line[0]
            w2 = line[1]
            i1 = P @ np.array([w1[0], w1[1], w1[2], 1])
            i2 = P @ np.array([w2[0], w2[1], w2[2], 1])
            i1 /= i1[-1]
            i2 /= i2[-1]
            frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), (255, 255, 255), 2)

        # Draw ellipses
        r = 9.15
        pts1, pts2, pts3 = [], [], []

        # Right penalty arc
        base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
        for ang in np.linspace(37, 143, 50):
            ang = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts1.append([ipos[0], ipos[1]])

        # Left penalty arc
        base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
        for ang in np.linspace(217, 323, 200):
            ang = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts2.append([ipos[0], ipos[1]])

        # Central circle
        base_pos = np.array([52.5-105/2, 68/2-68/2, 0., 0.])
        for ang in np.linspace(0, 360, 500):
            ang = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts3.append([ipos[0], ipos[1]])

        XEllipse1 = np.array(pts1, np.int32)
        XEllipse2 = np.array(pts2, np.int32)
        XEllipse3 = np.array(pts3, np.int32)

        frame = cv2.polylines(frame, [XEllipse1], False, (255, 255, 255), 3)
        frame = cv2.polylines(frame, [XEllipse2], False, (255, 255, 255), 3)
        frame = cv2.polylines(frame, [XEllipse3], False, (255, 255, 255), 3)

    # Draw keypoints if show_mode is 'keypoints' or 'both'
    if show_field_mode in ['keypoints', 'both']:
        for kp in keypoint_world_coords_2D:
            point = P @ np.array([kp[0], kp[1], 0, 1])  # Assuming z-coordinate is 0 for keypoints
            point /= point[-1]
            frame = cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(0, 0, 255), thickness=-1)

    return frame


    """
    A class for drawing dots on an image at specific coordinates based on provided
    detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        radius: int = 4,
        position: Position = Position.BOTTOM_CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        outline_thickness: int = 0,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            radius (int): Radius of the drawn dots.
            position (Position): The anchor position for placing the dot.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            outline_thickness (int): Thickness of the outline of the dot.
        """
        self.color: Union[Color, ColorPalette] = color
        self.radius: int = radius
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup
        self.outline_thickness = outline_thickness

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
        H_inv: np.ndarray = None
    ) -> ImageType:
        """
        Annotates the given scene with dots based on the provided detections.

        Args:
            scene (ImageType): The image where dots will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.
            H_inv (np.ndarray): The inverse projection matrix for transforming points.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)
        """
        xy = detections.get_anchors_coordinates(anchor=self.position)

        for detection_idx in range(len(detections)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            # Transform the point if H_inv is provided
            if H_inv is not None:
                center = project_point_to_field_map(xy[detection_idx], H_inv)
            else:
                center = (int(xy[detection_idx, 0]), int(xy[detection_idx, 1]))

            cv2.circle(scene, center, self.radius, color.as_bgr(), -1)
            if self.outline_thickness:
                cv2.circle(
                    scene, center, self.radius, (0, 0, 0), self.outline_thickness
                )
        return scene