import cv2

def extract_and_show_frames(video_path, frame_rate=1):
    """
    从视频中提取每一帧并直接显示出来。

    :param video_path: 视频文件的路径
    :param frame_rate: 每秒提取的帧数（默认为1帧/秒）
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 获取视频的帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")

    # 计算每帧之间的间隔
    frame_interval = int(video_fps / frame_rate)

    frame_count = 0

    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果读取成功
        if ret:
            # 检查是否需要显示当前帧
            if frame_count % frame_interval == 0:
                # 显示帧
                cv2.imshow('Frame', frame)

                # 按下 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
        else:
            # 如果读取失败，退出循环
            break

    # 释放视频捕获对象
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

# 示例用法
video_path = 'C:/Users/Lenovo/20250222_135506.mp4'
frame_rate = 1  # 每秒提取1帧

extract_and_show_frames(video_path, frame_rate)

# 示例用法

