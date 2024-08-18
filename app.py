import cv2
import tempfile
import yaml
import torch
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from examples.soccer.main import Mode, main

class FootballAnalyzer:
    def __init__(self):
        self.video_path = None
        self.team_info = None
        self.selected_team_info = None
        self.tempf = None

    def set_video_path(self, demo_selected, input_video_file):
        demo_vid_paths = {
            "Demo 1": 'examples/soccer/data/demo1.mp4',
            "Demo 2": 'examples/soccer/data/demo2.mp4'
        }
        demo_vid_path = demo_vid_paths[demo_selected]

        self.tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        if not input_video_file:
            self.tempf.name = demo_vid_path
            demo_vid = open(self.tempf.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Demo video')
            st.sidebar.video(demo_bytes)
        else:
            self.tempf.write(input_video_file.read())
            demo_vid = open(self.tempf.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input video')
            st.sidebar.video(demo_bytes)

    def set_team_info(self, demo_selected):
        demo_team_info = {
            "Demo 1": {"team1_name": "White",
                       "team2_name": "Red",
                       "team1_p_color": '#1E2530',
                       "team1_gk_color": '#F5FD15',
                       "team2_p_color": '#FBFCFA',
                       "team2_gk_color": '#B1FCC4',
                       },
            "Demo 2": {"team1_name": "Black",
                       "team2_name": "Red",
                       "team1_p_color": '#29478A',
                       "team1_gk_color": '#DC6258',
                       "team2_p_color": '#90C8FF',
                       "team2_gk_color": '#BCC703',
                       }
        }

        self.selected_team_info = demo_team_info[demo_selected]

    def run(self):
        st.set_page_config(page_title="Smart Football", layout="wide", initial_sidebar_state="expanded")
        st.title("Smart Football: Football Broadcast Video Analyzer")
        st.subheader(":red[Smart City and Community Innovation Center]")
        
        # Sidebar Setup
        st.sidebar.title("Main Settings")
        demo_selected = st.sidebar.radio(label="Select Demo Video", options=["Demo 1", "Demo 2"], horizontal=True)

        st.sidebar.markdown('---')
        st.sidebar.subheader("Video Upload")
        input_video_file = st.sidebar.file_uploader(label='Upload a video file', type=['mp4', 'mov', 'avi', 'm4v', 'asf'])

        st.sidebar.markdown('---')
        st.sidebar.subheader("Selected Video ")
        self.set_video_path(demo_selected, input_video_file)
        self.set_team_info(demo_selected)

        # Page Setup
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Tutorials", 
            "Player Detection", 
            "Player Tracking", 
            "Field Detection", 
            "Ball Detection",
            "Team Classification",
            "Player Localization"
        ])

        self.setup_tutorial_tab(tab1)
        self.setup_player_detection_tab(tab2)
        self.setup_player_tracking_tab(tab3)
        self.setup_field_detection_tab(tab4)
        self.setup_ball_detection_tab(tab5)
        self.setup_team_classification_tab(tab6)
        self.setup_player_localization_tab(tab7)

    def setup_tutorial_tab(self, tab):
        with tab:
            st.header(':blue[Welcome!]')

            st.subheader('Main Application Functionalities:', divider='blue')
            st.markdown("""
                1. Football players, referee, and ball detection.
                2. Players team prediction.
                3. Estimation of players and ball positions on a tactical map.
                4. Ball Tracking.
            """)
            st.subheader('How to use?', divider='blue')
            st.markdown("""
                **There are two demo videos that are automatically loaded when you start the app, alongside the recommended settings and hyperparameters**
                1. Upload a video to analyze, using the sidebar menu "Browse files" button.
                2. Enter the team names that correspond to the uploaded video in the text fields in the sidebar menu.
                3. Access the "Team colors" tab in the main page.
                4. Select a frame where players and goalkeepers from both teams can be detected.
                5. Follow the instruction on the page to pick each team colors.
                6. Go to the "Model Hyperparameters & Detection" tab, adjust hyperparameters and select the annotation options. (Default hyperparameters are recommended)
                7. Run Detection!
                8. If "save outputs" option was selected, the saved video can be found in the "outputs" directory
            """)
            st.write("Version 0.0.1")

    def setup_player_detection_tab(self, tab):
        with tab:
            st.header("Player Detection")

            st.markdown("""
                **Model Infrastructure**: 
                The player detection model is based on the YOLOv8x  architecture, which is well-known for its speed and accuracy in object detection tasks. The model is trained to detect football players in broadcast video footage.

                **Pipeline Process**: 
                1. The video frames are fed into the YOLO model.
                2. The model predicts bounding boxes and class probabilities for players in each frame.
                3. The bounding boxes are annotated on the video frames.
                
                **Benefits and Challenges**:
                - **Benefits**: High accuracy, real-time detection, robust to different camera angles.
                - **Challenges**: Requires good computational resources (GPU recommended), may struggle with occlusions and small player sizes.
            """)

            t2col1, t2col2 = st.columns([1, 1])

            with t2col1:
                player_detection_conf = st.slider(
                    'Player Detection Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.5,
                    key='pd_obj_conf'
                )

            with t2col2:
                save_output = st.checkbox(label='Save Output', value=False, key='pd_save_output')
                output_file_name = st.text_input(
                    label='File Name (Optional)',
                    placeholder='Enter output video file name.',
                    key='pd_output_file_name'
                ) if save_output else None

            st.markdown("---")

            ccol1, ccol2, ccol3, ccol4 = st.columns([1.5, 1, 1, 1])

            with ccol1:
                st.write('')

            with ccol2:
                start_detection = st.button(label='Start Detection', key='pd_start_detection')

            with ccol3:
                stop_btn_state = not start_detection
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state, key='pd_stop_detection')

            with ccol4:
                st.write('')

        stframe = st.empty()
        cap = cv2.VideoCapture(self.tempf.name)
        status = False

        if start_detection and not stop_detection:
            st.toast(f'Detection Started!')
            main(
                source_video_path=self.tempf.name,
                target_video_path=f'examples/soccer/data/{output_file_name}.mp4' if save_output else 'examples/soccer/data/pd_output.mp4',
                device='cuda:0',
                mode=Mode.PLAYER_DETECTION,
                player_detection_conf=player_detection_conf,
                stframe=stframe,
            )
        else:
            try:
                cap.release()
            except:
                pass
        if status:
            st.toast(f'Detection Completed!')
            cap.release()

    def setup_player_tracking_tab(self, tab):
        with tab:
            st.header("Player Tracking")

            st.markdown("""
                **Model Infrastructure**:
                Player tracking integrates object detection with a tracking algorithm (ByteTrack) to maintain player identities across frames.

                **Pipeline Process**:
                1. Detect players in each frame using YOLO.
                2. Track the detected players using ByteTrack to maintain their identities.
                3. Annotate tracked players on the video frames.

                **Benefits and Challenges**:
                - **Benefits**: Maintains player identities, handles occlusions and re-identification, provides consistent tracking.
                - **Challenges**: Requires high computational power, potential tracking drift in crowded scenes.
            """)

            t2col1, t2col2 = st.columns([1, 1])

            with t2col1:
                player_detection_conf = st.slider(
                    'Player Detection Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.5,
                    key='pt_obj_conf'
                )

            with t2col2:
                save_output = st.checkbox(label='Save Output', value=False, key='pt_save_output')
                output_file_name = st.text_input(
                    label='File Name (Optional)',
                    placeholder='Enter output video file name.',
                    key='pt_output_file_name'
                ) if save_output else None

            st.markdown("---")

            ccol1, ccol2, ccol3, ccol4 = st.columns([1.5, 1, 1, 1])

            with ccol1:
                st.write('')

            with ccol2:
                start_detection = st.button(label='Start Tracking', key='pt_start_detection')

            with ccol3:
                stop_btn_state = not start_detection
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state, key='pt_stop_detection')

            with ccol4:
                st.write('')

        stframe = st.empty()
        cap = cv2.VideoCapture(self.tempf.name)
        status = False

        if start_detection and not stop_detection:
            st.toast(f'Detection Started!')
            main(
                source_video_path=self.tempf.name,
                target_video_path=f'examples/soccer/data/{output_file_name}.mp4' if save_output else 'examples/soccer/data/pt_output.mp4',
                device='cuda:0',
                mode=Mode.PLAYER_TRACKING,
                player_detection_conf=player_detection_conf,
                stframe=stframe,
            )
        else:
            try:
                cap.release()
            except:
                pass
        if status:
            st.toast(f'Detection Completed!')
            cap.release()

    def setup_field_detection_tab(self, tab):
        with tab:
            st.header("Field Detection")

            st.markdown("""
                **Model Infrastructure**:
                The field detection model uses keypoint and line detection algorithms to identify the field's boundaries and key points.

                **Pipeline Process**:
                1. The video frames are processed to detect keypoints and lines on the field.
                2. The detected points and lines are used to annotate the field structure on the video frames.

                **Benefits and Challenges**:
                - **Benefits**: Accurate field structure detection, useful for tactical analysis.
                - **Challenges**: May require high computational resources, sensitive to lighting and camera angles.
            """)


            t2col1, t2col2 = st.columns([1, 1])

            with t2col1:
                keypoints_detection_conf = st.slider(
                    'Field Segmentation (Key Points) Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.7,
                    key='fd_keypoints_conf'
                )

                lines_detection_conf = st.slider(
                    'Field Segmentation (Lines) Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.7,
                    key='fd_lines_conf'
                )

            with t2col2:
                st.write('Annotation Settings')

                show_field_mode = st.radio(
                    "Select Annotation Display",
                    options=["keypoints", "lines", "both"],
                    key='fd_show_mode'
                )

                save_output = st.checkbox(label='Save Output', value=False, key='fd_save_output')
                output_file_name = st.text_input(
                    label='File Name (Optional)',
                    placeholder='Enter output video file name.',
                    key='fd_output_file_name'
                ) if save_output else None

            st.markdown("---")

            ccol1, ccol2, ccol3, ccol4 = st.columns([1.5, 1, 1, 1])

            with ccol1:
                st.write('')

            with ccol2:
                start_detection = st.button(label='Start Detection', key='fd_start_detection')

            with ccol3:
                stop_btn_state = not start_detection
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state, key='fd_stop_detection')

            with ccol4:
                st.write('')

        stframe = st.empty()
        cap = cv2.VideoCapture(self.tempf.name)
        status = False

        if start_detection and not stop_detection:
            st.toast(f'Detection Started!')
            main(
                source_video_path=self.tempf.name,
                target_video_path=f'examples/soccer/data/{output_file_name}.mp4' if save_output else 'examples/soccer/data/fd_output.mp4',
                device='cuda:0',
                mode=Mode.PITCH_DETECTION,
                keypoints_detection_conf=keypoints_detection_conf,
                lines_detection_conf=lines_detection_conf,
                show_field_mode=show_field_mode,
                stframe=stframe,
            )
        else:
            try:
                cap.release()
            except:
                pass
        if status:
            st.toast(f'Detection Completed!')
            cap.release()

    def setup_ball_detection_tab(self, tab):
        with tab:
            t2col1, t2col2 = st.columns([1, 1])

            with t2col1:
                ball_detection_conf = st.slider(
                    'Ball Detection Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.6,
                    key='bd_ball_conf'
                )

            with t2col2:
                save_output = st.checkbox(label='Save Output', value=False, key='bd_save_output')
                output_file_name = st.text_input(
                    label='File Name (Optional)',
                    placeholder='Enter output video file name.',
                    key='bd_output_file_name'
                ) if save_output else None

            st.markdown("---")

            ccol1, ccol2, ccol3, ccol4 = st.columns([1.5, 1, 1, 1])

            with ccol1:
                st.write('')

            with ccol2:
                start_detection = st.button(label='Start Detection', key='bd_start_detection')

            with ccol3:
                stop_btn_state = not start_detection
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state, key='bd_stop_detection')

            with ccol4:
                st.write('')

        stframe = st.empty()
        cap = cv2.VideoCapture(self.tempf.name)
        status = False

        if start_detection and not stop_detection:
            st.toast(f'Detection Started!')
            main(
                source_video_path=self.tempf.name,
                target_video_path=f'examples/soccer/data/{output_file_name}.mp4' if save_output else 'examples/soccer/data/bd_output.mp4',
                device='cuda:0',
                mode=Mode.BALL_DETECTION,
                ball_detection_conf=ball_detection_conf,
                stframe=stframe,
            )
        else:
            try:
                cap.release()
            except:
                pass
        if status:
            st.toast(f'Detection Completed!')
            cap.release()

    def setup_team_classification_tab(self, tab):
        with tab:
            st.header("Team Classification")

            st.markdown("""
                **Model Infrastructure**:
                The team classification model uses a combination of object detection and color classification to identify players and their respective teams.

                **Pipeline Process**:
                1. Detect players in each frame using YOLO.
                2. Extract player images and classify team based on uniform color.
                3. Annotate detected players with their team colors.

                **Benefits and Challenges**:
                - **Benefits**: Effective in distinguishing teams, useful for tactical analysis.
                - **Challenges**: Sensitive to uniform colors, may require additional training for new teams.
            """)

            t2col1, t2col2 = st.columns([1, 1])

            with t2col1:
                player_detection_conf = st.slider(
                    'Player Detection Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.5,
                    key='tc_obj_conf'
                )

            with t2col2:
                save_output = st.checkbox(label='Save Output', value=False, key='tc_save_output')
                output_file_name = st.text_input(
                    label='File Name (Optional)',
                    placeholder='Enter output video file name.',
                    key='tc_output_file_name'
                ) if save_output else None

            st.markdown("---")

            ccol1, ccol2, ccol3, ccol4 = st.columns([1.5, 1, 1, 1])

            with ccol1:
                st.write('')

            with ccol2:
                start_detection = st.button(label='Start Detection', key='tc_start_detection')

            with ccol3:
                stop_btn_state = not start_detection
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state, key='tc_stop_detection')

            with ccol4:
                st.write('')

        stframe = st.empty()
        cap = cv2.VideoCapture(self.tempf.name)
        status = False

        if start_detection and not stop_detection:
            st.toast(f'Detection Started!')
            main(
                source_video_path=self.tempf.name,
                target_video_path=f'examples/soccer/data/{output_file_name}.mp4' if save_output else 'examples/soccer/data/tc_output.mp4',
                device='cuda:0',
                mode=Mode.TEAM_CLASSIFICATION,
                player_detection_conf=player_detection_conf,
                stframe=stframe,
            )
        else:
            try:
                cap.release()
            except:
                pass
        if status:
            st.toast(f'Detection Completed!')
            cap.release()

    def setup_player_localization_tab(self, tab):
        with tab:
            st.header("Player Localization")

            st.markdown("""
                **Model Infrastructure**:
                Player localization combines object detection and field segmentation to place players accurately on a tactical map.

                **Pipeline Process**:
                1. Detect players and key field points in each frame.
                2. Use detected key points to transform player positions to a top-down view.
                3. Annotate player positions on the tactical map.

                **Benefits and Challenges**:
                - **Benefits**: Accurate player localization, useful for tactical analysis and strategy planning.
                - **Challenges**: Requires precise field segmentation, computationally intensive.
            """)

            t2col1, t2col2 = st.columns([1, 1])

            with t2col1:
                player_detection_conf = st.slider(
                    'Player Detection Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.5,
                    key='pl_obj_conf'
                )

                ball_detection_conf = st.slider(
                    'Ball Detection Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.6,
                    key='pl_ball_conf'
                )

                keypoints_detection_conf = st.slider(
                    'Field Segmentation (Key Points) Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.7,
                    key='pl_keypoints_conf'
                )

                lines_detection_conf = st.slider(
                    'Field Segmentation (Lines) Confidence Threshold',
                    min_value=0.0, max_value=1.0, value=0.7,
                    key='pl_lines_conf'
                )

            with t2col2:
                show_field_mode = st.radio(
                    "Select Annotation Display",
                    options=["keypoints", "lines", "both"],      
                    key='pl_show_mode'
                )

                show_o = st.checkbox(label="Show Players Detections", value=True, key='pl_show_o')
                show_ot = st.checkbox(label="Show Players Trackings", value=True, key='pt_show_o')

                save_output = st.checkbox(label='Save Output', value=False, key='pl_save_output')
                output_file_name = st.text_input(
                    label='File Name (Optional)',
                    placeholder='Enter output video file name.',
                    key='pl_output_file_name'
                ) if save_output else None

            st.markdown("---")

            ccol1, ccol2, ccol3, ccol4 = st.columns([1.5, 1, 1, 1])

            with ccol1:
                st.write('')

            with ccol2:
                start_detection = st.button(label='Start Detection', key='pl_start_detection')

            with ccol3:
                stop_btn_state = not start_detection
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state, key='pl_stop_detection')

            with ccol4:
                st.write('')

        stframe = st.empty()
        st.sidebar.markdown('---')

        stframe_map = st.empty()
        cap = cv2.VideoCapture(self.tempf.name)
        status = False

        if start_detection and not stop_detection:
            st.toast(f'Detection Started!')
            main(
                source_video_path=self.tempf.name,
                target_video_path=f'examples/soccer/data/{output_file_name}.mp4' if save_output else 'examples/soccer/data/pl_output.mp4',
                target_video_map_path=f'examples/soccer/data/{output_file_name}_map.mp4' if save_output else 'examples/soccer/data/pl_output_map.mp4',
                device='cuda:0',
                mode=Mode.LOCALIZATION,
                player_detection_conf=player_detection_conf,
                ball_detection_conf=ball_detection_conf,
                keypoints_detection_conf=keypoints_detection_conf,
                lines_detection_conf=lines_detection_conf,
                show_field_mode=show_field_mode,
                stframe=stframe,
                stframe_map=stframe_map
            )
        else:
            try:
                cap.release()
            except:
                pass
        if status:
            st.toast(f'Detection Completed!')
            cap.release()



if __name__ == '__main__':
    analyzer = FootballAnalyzer()
    try:
        analyzer.run()
    except SystemExit:
        pass
