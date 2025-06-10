import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any, Optional
import cadquery as cq
import trimesh
import multiprocessing
import json
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
            max_workers = min(32, multiprocessing.cpu_count())
        print(f"Using {max_workers} workers for CADQuery batch execution")
        
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
        Divides into smaller batches equal to worker count to prevent blocking.
        
        Args:
            cadquery_codes: List of CADQuery code strings
            
        Returns:
            A tuple containing:
                - List of (success, result) tuples
                - Integer count of timeouts in this batch
        """
        all_results = []
        total_timeout_count = 0
        
        # Process in smaller batches to prevent blocking
        for i in range(0, len(cadquery_codes), self.max_workers):
            minibatch = cadquery_codes[i:i + self.max_workers]
            
            # Submit all tasks in this batch
            futures = []
            for code in minibatch:
                future = self.executor.submit(_execute_cadquery_worker, code)
                futures.append(future)
            
            # Collect results with timeout handling
            minibatch_results = []
            timed_out_futures = []
            minibatch_timeout_count = 0
            
            for i, future in enumerate(futures):
                # Give first task full timeout, others get minimal timeout to collect completed results
                timeout = self.timeout_seconds if i == 0 else 1.0
                
                try:
                    result = future.result(timeout=timeout)
                    minibatch_results.append(result)
                except ConcurrentTimeoutError:
                    minibatch_timeout_count += 1
                    minibatch_results.append((False, f"CADQuery execution timed out after {self.timeout_seconds} seconds"))
                    timed_out_futures.append(future)
                except Exception as e:
                    minibatch_results.append((False, f"CADQuery execution failed: {str(e)}"))
            
            # Add batch results to overall results
            all_results.extend(minibatch_results)
            total_timeout_count += minibatch_timeout_count
            
            # If any timeouts occurred, restart workers for next batch
            if minibatch_timeout_count > 0:
                self._start_executor()
        
        return all_results, total_timeout_count
    
    def shutdown(self):
        """Shutdown the worker pool."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

def cadquery_codes_to_pointclouds_batch(cadquery_codes: List[str]) -> Tuple[List[Optional[np.ndarray]], int]:
    """Convert a batch of CADQuery codes to point clouds efficiently.
    
    Args:
        cadquery_codes: List of CADQuery code strings
        
    Returns:
        A tuple containing:
            - Integer count of timeouts in this batch
    """
    pool = CADQueryWorkerPool()
    results, timeout_count = pool.execute_batch(cadquery_codes)
    pool.shutdown()
    
    point_clouds = []
    for success, result_data in results:
        if success:
            try:
                vertices, faces = result_data
                # Convert to mesh and normalize
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh = normalize_mesh(mesh)
                
                # Sample points from the surface
                points, _ = trimesh.sample.sample_surface(mesh, 8192)
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

def compute_score(predicts: List[str], ground_truths: List[Tuple[Any, Any, np.ndarray]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """Compute scores for a list of predictions and ground truths using batch processing.
    
    Args:
        predicts: List of predicted strings containing CADQuery code.
        ground_truths: List of ground truth tuples, each as (obj_id, turn, points).
        format_weight: Weight given to the format score in the overall score.
        
    Returns:
        List[Dict[str, float]]: List of score dictionaries containing:
            - overall: Weighted average of accuracy and format scores.
            - format: Compilation score (1.0 if CADQuery code compiles, 0.0 otherwise).
            - accuracy: Chamfer-based reward (0.0 if compilation fails).
    """

    print(f"Computing reward for {len(predicts)} predictions")

    pred_point_clouds, timeout_count = cadquery_codes_to_pointclouds_batch(predicts)

    # Compute scores
    scores = []
    json_results = []
    success_count = 0
    chamfer_distances = []
    
    for predict, pred_points, (obj_id, turn, gt_points) in zip(predicts, pred_point_clouds, ground_truths):
        accuracy_score = 0.0
        compilation_score = 0.0
        chamfer_distance = -1.0

        if pred_points is not None:
            success_count += 1
            chamfer_distance = compute_chamfer_distance(pred_points, gt_points)
            accuracy_score = chamfer_reward(pred_points, gt_points, alpha=40)
            compilation_score = 1.0
            chamfer_distances.append(chamfer_distance)

        scores.append({
            "overall": compilation_score * format_weight + (1 - format_weight) * accuracy_score,
            "format": compilation_score, 
            "accuracy": accuracy_score
        })
        json_results.append({
            'obj_id': obj_id,
            'turn': turn,
            'predicted_cadquery': predict,
            'chamfer_distance': chamfer_distance
        })
    
    # Always save to ~/benchmark_results.json
    import os
    results_json_path = os.path.expanduser('~/benchmark_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print("Batch reward processing completed!")
    print(f"{success_count}/{len(predicts)} successful")
    print(f"{timeout_count}/{len(predicts)} timed out")
    if chamfer_distances:
        mean_chamfer_distance = np.mean(chamfer_distances)
        print(f"Mean chamfer distance: {mean_chamfer_distance:.6f}")
    else:
        print("Mean chamfer distance: N/A (no successful compilations)")
    print(f"Results saved to {results_json_path}")
    return scores