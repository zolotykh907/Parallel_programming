import argparse
import threading
import queue
import time
import cv2
from ultralytics import YOLO
from operator import itemgetter

def multythreading(path_in_video, count_thr, path_out_video):
    out_lock = threading.Lock()
    event_stop = threading.Event()
    max_buf_size = 30
    buf=[]

    writer = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_out_video, writer, 60.0, (640, 480))

    def write_video(arr_img):
        sorted_arr_img = sorted(arr_img, key = itemgetter(1))
        for elem in sorted_arr_img:
            out.write(elem[0])

    def fun_thread_read(path_video: str, frame_queue: queue.Queue, event_stop: threading.Event):
        cap = cv2.VideoCapture(path_video)
        id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame!")
                break
            while frame_queue.qsize() > max_buf_size:
                time.sleep(0.01)
            frame_queue.put((frame, id))
            id+=1
            time.sleep(0.0001)
        event_stop.set()

    def fun_thread_safe_predict(frame_queue: queue.Queue, event_stop: threading.Event):
        local_model = YOLO(model="yolov8s-pose.pt")
        while True:
            try:
                frame, id = frame_queue.get(timeout=1)
                results = local_model.predict(source=frame, device='cpu')

                for r in results:
                    frame = r.plot()
                    buf.append((frame, id))
                    if len(buf)>max_buf_size:
                        with out_lock:
                            write_video(buf)
                            buf.clear()

            except queue.Empty:
                if event_stop.is_set():
                    print(f'Thread {threading.get_ident()} final!')
                    break

        if buf:
            with out_lock:
                write_video(buf)
        out.release()

    threads = []
    frame_queue = queue.Queue()

    thread_read = threading.Thread(target=fun_thread_read, args=(path_in_video, frame_queue, event_stop,))
    thread_read.start()

    start_t = time.monotonic()

    for _ in range(count_thr):
        threads.append(threading.Thread(target=fun_thread_safe_predict, args=(frame_queue, event_stop,)))

    for thr in threads:
        thr.start()

    thread_read.join()
    for thr in threads:
        thr.join()

    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')
    print(f'Relusts save to {path_out_video}')

def one_thread(path_in_video, path_out_video):
    video_path = path_in_video
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_out_video, fourcc, 60.0, (640, 480))

    local_model = YOLO(model="yolov8s-pose.pt")

    start_t = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame!")
            break

        results = local_model.predict(source=frame, device='cpu')
        for i, r in enumerate(results):
            frame = r.plot()

        out.write(frame)

    out.release()

    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('in_name', type=str, help='path to video')
    parser.add_argument('count_thr', type=int, help='count of threading')
    parser.add_argument('out_name', type=str, help='path to output video')

    args = parser.parse_args()

    if args.count_thr == 1:
        one_thread(args.in_name, args.out_name)
    else:
        multythreading(args.in_name, args.count_thr, args.out_name)

arg_parser()

