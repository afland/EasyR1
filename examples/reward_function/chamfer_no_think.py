import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any
import cadquery as cq
import trimesh
import multiprocessing
import re
import os
import uuid

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply standard transformations to normalize the mesh to unit cube.
    
    Args:
        mesh: Input trimesh mesh
        
    Returns:
        trimesh.Trimesh: Transformed mesh centered in unit cube with max dimension = 1
    """
    # Current transformations (commented out)
    # mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 100 / 2))
    # mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    
    # Center at origin and scale to fit in unit cube
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # Center at origin
    mesh.apply_scale(2.0 / max(mesh.extents))  # Scale to fit in unit cube
    
    return mesh


# Helper function to run in a separate process
def _execute_cadquery_in_process(code: str, queue: multiprocessing.Queue, tessellate_tolerance: float, tessellate_angular_tolerance: float):
    """
    Executes CADQuery code and tessellates the result in a separate process.
    Puts (True, (vertices, faces)) or (False, error_message) into the queue.
    """
    namespace = {'cq': cq}
    try:
        exec(code, namespace)
        result = namespace.get('r')

        if result is None:
            queue.put((False, "CADQuery code must create a variable called 'r' with the final shape", None))
            return

        if isinstance(result, cq.Workplane):
            result = result.val()
        
        if not isinstance(result, cq.Shape):
            queue.put((False, "Result must be a CADQuery Shape", None))
            return

        # Perform tessellation in the subprocess
        vertices_tuples, faces_tuples = result.tessellate(tessellate_tolerance, tessellate_angular_tolerance)
        # Convert CQ Vectors to simple tuples for pickling, though they might be picklable directly.
        # This is a safer bet for cross-process communication.
        vertices = [(v.x, v.y, v.z) for v in vertices_tuples]
        # Faces are already lists of indices, should be fine.
        queue.put((True, (vertices, faces_tuples), None))

    except Exception as e:
        queue.put((False, f"Exception during CADQuery execution or tessellation: {str(e)}", None))

def execute_cadquery_code(code: str, timeout_seconds: float = 0.2, tessellate_tolerance: float = 0.001, tessellate_angular_tolerance: float = 0.1) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    """
    Execute CADQuery code in a separate process with a timeout.
    The tessellation also happens in the separate process.
    
    Args:
        code: CADQuery code string
        timeout_seconds: Timeout for the execution.
        tessellate_tolerance: Linear tolerance for tessellation.
        tessellate_angular_tolerance: Angular tolerance for tessellation.
        
    Returns:
        A tuple (vertices, faces) if successful.
        Vertices is a list of (x,y,z) tuples.
        Faces is a list of lists of vertex indices.
        
    Raises:
        TimeoutError: If code execution exceeds timeout.
        ValueError: If code execution fails, result variable 'r' is not found,
                    or result is not a cq.Shape, or other errors.
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_cadquery_in_process,
        args=(code, queue, tessellate_tolerance, tessellate_angular_tolerance)
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate() # Send SIGTERM
        process.join() # Wait for termination
        raise TimeoutError(f"CADQuery execution timed out after {timeout_seconds} seconds")

    try:
        # Get result from queue, with a short timeout in case the process died before putting anything
        success, result_data, _ = queue.get(timeout=0.1) 
        if success:
            vertices, faces = result_data
            return vertices, faces
        else:
            # result_data is the error message string in case of failure
            raise ValueError(f"CADQuery processing error: {result_data}")
    except Exception as e:
        raise ValueError(f"Error in CADQuery subprocess: {str(e)}")
    finally:
        # Ensure the process is joined if it wasn't terminated due to timeout
        if process.is_alive():
            process.join() # Should not happen if logic above is correct
        # Close the queue
        queue.close()
        queue.join_thread() # Ensure all data in buffer is flushed


def cadquery_to_pointcloud(cadquery_code: str, n_points: int = 8192) -> np.ndarray:
    """Convert CADQuery code to a point cloud.
    
    Args:
        cadquery_code: String containing CADQuery code
        n_points: Number of points to sample from the surface
        
    Returns:
        np.ndarray: Point cloud of shape (n_points, 3)
    """
    # Execute the CADQuery code and get tessellated vertices/faces
    vertices, faces = execute_cadquery_code(cadquery_code, timeout_seconds=0.5) # Increased timeout for safety with process overhead
    
    # Convert CADQuery shape to mesh using tessellate
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces) # Construct Trimesh from returned data
    
    # Normalize the mesh
    mesh = normalize_mesh(mesh)
    
    # Sample points from the surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    
    return points

def compute_chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Compute the chamfer distance between two point clouds.
    
    Args:
        pred_points: Predicted point cloud of shape (N, 3)
        gt_points: Ground truth point cloud of shape (M, 3)
        
    Returns:
        float: Chamfer distance between the point clouds
    """
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd


def chamfer_reward(pred_points: np.ndarray, gt_points: np.ndarray, alpha: float = 1.0) -> float:
    """Compute reward based on chamfer distance.
    
    Args:
        pred_points: Predicted point cloud of shape (N, 3)
        gt_points: Ground truth point cloud of shape (M, 3)
        alpha: Scaling factor for the exponential reward
        
    Returns:
        float: Reward value between 0 and 1
    """
    cd = compute_chamfer_distance(pred_points, gt_points)
    return np.exp(-alpha * cd)

def compute_score(predicts: List[str], ground_truths: List[np.ndarray], format_weight: float = 0.2) -> List[Dict[str, float]]:
    """Compute scores for a list of predictions and ground truths.
    
    Args:
        predicts: List of predicted strings, potentially containing CADQuery code and special tags.
        ground_truths: List of ground truth identifiers (used to load pre-computed point clouds).
        format_weight: Weight given to the format score in the overall score.
        compilation_weight: Weight given to the compilation score in the overall score.
        
    Returns:
        List[Dict[str, float]]: List of score dictionaries containing:
            - overall: Weighted average of accuracy, format, and compilation scores.
            - format: Score based on the presence of <think> and <answer> tags (0.0 or 1.0).
            - accuracy: Chamfer-based reward (0.0 if <answer> tags are missing/empty, or on CAD processing error).
            - compilation: 1.0 if CADQuery code compiles, 0.0 otherwise.
    """
    scores = []
    for predict, gt_points in zip(predicts, ground_truths):
        accuracy_score = 0.0
        compilation_score = 0.0

        try:
            pred_points = cadquery_to_pointcloud(predict)
            accuracy_score = chamfer_reward(pred_points, gt_points, alpha=11)
            compilation_score = 1.0
            print(f"Success! Accuracy score: {accuracy_score}")
            print(f"Good CAD code:\n{predict}")
            if accuracy_score > 0.8:
                random_id = str(uuid.uuid4())
                output_dir = os.path.expanduser('~/npys')
                os.makedirs(output_dir, exist_ok=True)
                file_path_gt = os.path.join(output_dir, f'{random_id}_gt.npy')
                file_path_pr = os.path.join(output_dir, f'{random_id}_pr.npy')
                np.save(file_path_gt, gt_points)
                np.save(file_path_pr, pred_points)
        except Exception as e:
            print(f"Error processing CAD code: {e}")

        scores.append({
            "overall": compilation_score * format_weight + (1 - format_weight) * accuracy_score,
            "format": compilation_score, 
            "accuracy": accuracy_score
        })
    return scores