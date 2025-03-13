from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement import CameraMovEstimator
from view_transformer import ViewTransformer
from speed_dist_estimator import SpeedDistanceEstimator

def main():
    # Read video
    video_path = "input_videos/input_vid1.mp4"
    video_frames = read_video(video_path)

    # Initialize tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_obj_tracks(video_frames)
    # Get Object Positions
    tracker.add_pos_to_track(tracks)

    # Camera Movement Estimation
    cam_mov_estimator = CameraMovEstimator(video_frames[0])
    cam_mov_per_frame = cam_mov_estimator.get_cam_mov(video_frames)
    cam_mov_estimator.add_adjust_pos_to_tracks(tracks, cam_mov_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_pos(tracks["ball"])

    # Speed and distance estimator
    frame_window = 5
    speed_and_distance_estimator = SpeedDistanceEstimator(frame_window)
    speed_and_distance_estimator.add_speed_dist_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw Camera movement
    output_video_frames = cam_mov_estimator.draw_cam_mov(output_video_frames, cam_mov_per_frame)

    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_dist(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()
