import time
import json
from pathlib import Path
from typing import Union, Optional
from ultralytics import YOLOWorld

def calculate_fps(
        model_path: str,
        source: Union[str, int] = "data/sample.mp4",
        num_frames: int = 100,
        save_path: str = "fps_results.json",
        prompt: Optional[str] = None
) -> float:
    """
    Calculate FPS (Frames Per Second) for a YOLO-World model and save results to a JSON file.
    Supports custom object detection via prompt.

    Args:
        model_path (str): Path to the YOLO model weights (e.g., 'yolov8s-worldv2.pt').
        source (Union[str, int]): Path to the video/image or webcam ID.
        num_frames (int): Number of frames to process for FPS calculation.
        save_path (str): Path to save the FPS results as JSON.
        prompt (Optional[str]): Text prompt for YOLO-World (e.g., "person, car").

    Returns:
        float: Calculated FPS.
    """
    fps: float = 0.0
    frame_count: int = 0
    elapsed_time: float = 0.0

    try:

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file {model_path} does not exist.")


        if isinstance(source, str):
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file {source} does not exist.")
        else:
            source_path = source

        try:
            model = YOLOWorld(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        start_time: float = time.time()

        predict_args = {"source": str(source_path), "stream": True}
        if prompt:
            predict_args["prompts"] = prompt
            print(f"Using YOLO-World with prompts: {prompt}")

        try:
            for result in model.predict(**predict_args):
                frame_count += 1
                if num_frames > 0 and frame_count >= num_frames:
                    break
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        end_time: float = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

    except FileNotFoundError as e:
        print(f"File error: {str(e)}")
    except RuntimeError as e:
        print(f"Runtime error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


    results: dict = {
        "model": str(model_path),
        "source": str(source),
        "frames_processed": frame_count,
        "elapsed_time_sec": elapsed_time,
        "fps": fps,
        "prompt": prompt if prompt else "Standard YOLO detection",
        "status": "success" if frame_count > 0 else "failed"
    }

    try:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_obj, "w") as f:
            json.dump(results, f, indent=4)
        print(f"FPS: {fps:.2f}, Results saved to {save_path}")
    except Exception as e:
        print(f"Failed to save results: {str(e)}")

    return fps


if __name__ == "__main__":
    calculate_fps(
        "ultralytics/data/yolov8s-worldv2.pt",
        source="ultralytics/ultralytics/data/sample.mp4",
        num_frames=100,
        prompt="person"
    )

''' 
==============fps_resilts.json==============
{
    "model": "ultralytics/data/yolov8s-worldv2.pt",
    "source": "ultralytics/ultralytics/data/sample.mp4",
    "frames_processed": 100,
    "elapsed_time_sec": 34.28576946258545,
    "fps": 2.9166619728084444,
    "prompt": "person",
    "status": "success"
}
'''
