import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any, Optional
import cadquery as cq
import trimesh
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError as ConcurrentTimeoutError
# import re
# import os
# import uuid

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


def _execute_cadquery_worker(code: str) -> Tuple[bool, Any]:
    """
    Worker function that executes CADQuery code and tessellates the result.
    This runs in a worker process.
    
    Returns:
        Tuple of (success: bool, result: vertices and faces tuple or error message)
    """
    namespace = {'cq': cq}
    try:
        exec(code, namespace)
        result = namespace.get('r')

        if result is None:
            return False, "CADQuery code must create a variable called 'r' with the final shape"

        if isinstance(result, cq.Workplane):
            result = result.val()
        
        if not isinstance(result, cq.Shape):
            return False, "Result must be a CADQuery Shape"

        # Perform tessellation
        vertices_tuples, faces_tuples = result.tessellate(0.001, 0.1)
        # Convert CQ Vectors to simple tuples for pickling
        vertices = [(v.x, v.y, v.z) for v in vertices_tuples]
        
        return True, (vertices, faces_tuples)

    except Exception as e:
        return False, f"Exception during CADQuery execution or tessellation: {str(e)}"


class CADQueryWorkerPool:
    """
    A worker pool for executing CADQuery code with timeout handling.
    Workers are only replaced when they timeout, not after each task.
    """
    
    def __init__(self, max_workers: Optional[int] = None, timeout_seconds: float = 5.0):
        """
        Initialize the worker pool.
        
        Args:
            max_workers: Maximum number of worker processes. If None, defaults to min(32, CPU count)
            timeout_seconds: Timeout for individual CADQuery executions
        """
        if max_workers is None:
            # For Intel Xeon Platinum 8480 with 56 cores, we can use more workers
            # but cap at 32 to avoid excessive overhead
            max_workers = min(16, multiprocessing.cpu_count())
        
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.executor = None
        self._start_executor()
    
    def _start_executor(self):
        """Start or restart the process pool executor."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
        
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers
        )
    
    def execute_batch(self, cadquery_codes: List[str]) -> Tuple[List[Tuple[bool, Any]], int]:
        """
        Execute a batch of CADQuery code strings in parallel.
        
        Args:
            cadquery_codes: List of CADQuery code strings
            
        Returns:
            A tuple containing:
                - List of (success, result) tuples
                - Integer count of timeouts in this batch
        """
        # Submit all tasks
        futures = []
        for code in cadquery_codes:
            future = self.executor.submit(_execute_cadquery_worker, code)
            futures.append(future)
        
        # Collect results with timeout handling
        results = []
        timed_out_futures = []
        timeout_count = 0
        
        for future in futures:
            try:
                result = future.result(timeout=self.timeout_seconds)
                results.append(result)
            except ConcurrentTimeoutError:
                timeout_count += 1
                results.append((False, f"CADQuery execution timed out after {self.timeout_seconds} seconds"))
                timed_out_futures.append(future)
            except Exception as e:
                results.append((False, f"CADQuery execution failed: {str(e)}"))
        
        # If we had timeouts, we need to restart the executor to clean up hung processes
        if timed_out_futures:
            for future in timed_out_futures:
                future.cancel()
            self._start_executor()
        
        return results, timeout_count
    
    def shutdown(self):
        """Shutdown the worker pool."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

# Global worker pool instance
_global_worker_pool = None

def get_worker_pool() -> CADQueryWorkerPool:
    """Get or create the global worker pool."""
    global _global_worker_pool
    if _global_worker_pool is None:
        _global_worker_pool = CADQueryWorkerPool()
    return _global_worker_pool

def shutdown_worker_pool():
    """Shutdown the global worker pool."""
    global _global_worker_pool
    if _global_worker_pool is not None:
        _global_worker_pool.shutdown()
        _global_worker_pool = None


def cadquery_codes_to_pointclouds_batch(cadquery_codes: List[str], n_points: int = 8192) -> Tuple[List[Optional[np.ndarray]], int]:
    """Convert a batch of CADQuery codes to point clouds efficiently.
    
    Args:
        cadquery_codes: List of CADQuery code strings
        n_points: Number of points to sample from each surface
        
    Returns:
        A tuple containing:
            - List of point clouds (np.ndarray of shape (n_points, 3)) or None if failed
            - Integer count of timeouts in this batch
    """
    pool = get_worker_pool()
    results, timeout_count = pool.execute_batch(cadquery_codes)
    
    point_clouds = []
    for success, result_data in results:
        if success:
            try:
                vertices, faces = result_data
                # Convert to mesh and normalize
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh = normalize_mesh(mesh)
                
                # Sample points from the surface
                points, _ = trimesh.sample.sample_surface(mesh, n_points)
                point_clouds.append(points)
            except Exception as e:
                point_clouds.append(None)
        else:
            point_clouds.append(None)
    return point_clouds, timeout_count

def compute_chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd

def chamfer_reward(pred_points: np.ndarray, gt_points: np.ndarray, alpha: float = 1.0) -> float:
    cd = compute_chamfer_distance(pred_points, gt_points)
    return np.exp(-alpha * cd)

def compute_score(predicts: List[str], ground_truths: List[np.ndarray], format_weight: float = 0.2) -> List[Dict[str, float]]:
    """Compute scores for a list of predictions and ground truths using batch processing.
    
    Args:
        predicts: List of predicted strings containing CADQuery code.
        ground_truths: List of ground truth point clouds.
        format_weight: Weight given to the format score in the overall score.
        
    Returns:
        List[Dict[str, float]]: List of score dictionaries containing:
            - overall: Weighted average of accuracy and format scores.
            - format: Compilation score (1.0 if CADQuery code compiles, 0.0 otherwise).
            - accuracy: Chamfer-based reward (0.0 if compilation fails).
    """
    
    # Process all CADQuery codes in parallel
    pred_point_clouds, timeout_count = cadquery_codes_to_pointclouds_batch(predicts)
    
    # Compute scores
    scores = []
    success_count = 0
    
    for pred_points, gt_points in zip(pred_point_clouds, ground_truths):
        accuracy_score = 0.0
        compilation_score = 0.0

        if pred_points is not None:
            success_count += 1
            accuracy_score = chamfer_reward(pred_points, gt_points, alpha=12)
            compilation_score = 1.0

        scores.append({
            "overall": compilation_score * format_weight + (1 - format_weight) * accuracy_score,
            "format": compilation_score, 
            "accuracy": accuracy_score
        })
    
    print("Batch processing completed!")
    print(f"{success_count}/{len(predicts)} successful")
    print(f"{timeout_count}/{len(predicts)} timed out")
    return scores