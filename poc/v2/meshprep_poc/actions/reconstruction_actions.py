# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Surface Reconstruction Actions for Extremely Fragmented Meshes.

These actions treat fragmented meshes as point clouds and reconstruct
the surface from scratch. Essential for meshes with 1000+ bodies that
cannot be repaired with traditional methods.

Key approaches:
1. Screened Poisson Reconstruction (Open3D) - Best for organic shapes
2. Point cloud to mesh with normal estimation
3. Shrinkwrap reconstruction - Envelope-based approach
4. Morphological voxel reconstruction - Gap filling for fragments
"""

import logging
import numpy as np
import trimesh

from . import register_action

logger = logging.getLogger(__name__)

# Check for Open3D availability
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not available - some reconstruction actions disabled")

# Check for scipy availability (for morphological operations)
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import error logging
try:
    from .error_logging import log_action_failure
    ERROR_LOGGING_AVAILABLE = True
except ImportError:
    ERROR_LOGGING_AVAILABLE = False
    log_action_failure = None


def _log_reconstruction_failure(action_name: str, error_message: str, mesh: trimesh.Trimesh) -> None:
    """Log a reconstruction action failure."""
    if ERROR_LOGGING_AVAILABLE and log_action_failure:
        log_action_failure(
            action_name=action_name,
            error_message=error_message,
            mesh=mesh,
            action_type="reconstruction",
        )


def _trimesh_to_open3d_pointcloud(mesh: trimesh.Trimesh) -> "o3d.geometry.PointCloud":
    """Convert trimesh vertices to Open3D point cloud with estimated normals."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required for this operation")
    
    # Get all vertices from the mesh (treating as point cloud)
    points = np.asarray(mesh.vertices)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals - critical for reconstruction
    # Use larger radius for fragmented meshes
    bbox = mesh.bounds
    bbox_size = bbox[1] - bbox[0]
    search_radius = np.max(bbox_size) / 50  # 2% of bbox
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius, 
            max_nn=30
        )
    )
    
    # Orient normals consistently (important for Poisson)
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    return pcd


def _open3d_mesh_to_trimesh(o3d_mesh: "o3d.geometry.TriangleMesh") -> trimesh.Trimesh:
    """Convert Open3D mesh to trimesh."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@register_action(
    name="open3d_screened_poisson",
    description="Screened Poisson surface reconstruction (best for fragmented meshes)",
    parameters={"depth": 9, "width": 0, "scale": 1.1, "linear_fit": False},
    risk_level="high"
)
def action_open3d_screened_poisson(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Screened Poisson Surface Reconstruction using Open3D.
    
    This is the gold standard for reconstructing surfaces from point clouds.
    Treats the fragmented mesh as a point cloud, estimates normals, and
    reconstructs a smooth, watertight surface.
    
    Best for:
    - Extremely fragmented meshes (1000+ bodies)
    - Noisy scan data
    - Meshes that need complete surface reconstruction
    
    Args:
        mesh: Input mesh (will be treated as point cloud)
        params:
            - depth: Octree depth (8-12, higher = more detail, default 9)
            - width: Target width of finest octree cells (0 = auto)
            - scale: Ratio of cube diameter to bounding box (default 1.1)
            - linear_fit: Use linear interpolation (default False)
    
    Returns:
        Reconstructed watertight mesh
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, falling back to original mesh")
        return mesh.copy()
    
    depth = params.get("depth", 9)
    width = params.get("width", 0)
    scale = params.get("scale", 1.1)
    linear_fit = params.get("linear_fit", False)
    
    try:
        # Convert to point cloud
        logger.info(f"Converting {len(mesh.vertices)} vertices to point cloud")
        pcd = _trimesh_to_open3d_pointcloud(mesh)
        
        logger.info(f"Running Screened Poisson reconstruction (depth={depth})")
        
        # Run Poisson reconstruction
        reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )
        
        # Remove low-density vertices (artifacts outside the point cloud)
        if len(densities) > 0:
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)  # Remove bottom 1%
            vertices_to_remove = densities < density_threshold
            reconstructed.remove_vertices_by_mask(vertices_to_remove)
        
        # Convert back to trimesh
        result = _open3d_mesh_to_trimesh(reconstructed)
        
        logger.info(f"Reconstruction complete: {len(result.vertices)} vertices, {len(result.faces)} faces")
        
        return result
        
    except Exception as e:
        logger.error(f"Screened Poisson reconstruction failed: {e}")
        _log_reconstruction_failure("reconstruct_poisson", str(e), mesh)
        return mesh.copy()


@register_action(
    name="open3d_ball_pivoting",
    description="Ball pivoting surface reconstruction",
    parameters={"radii_factor": 1.0},
    risk_level="high"
)
def action_open3d_ball_pivoting(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Ball Pivoting Algorithm surface reconstruction.
    
    Rolls a ball over the point cloud to create triangles.
    Good for point clouds with relatively uniform sampling.
    
    Args:
        mesh: Input mesh (treated as point cloud)
        params:
            - radii_factor: Multiplier for auto-calculated radii (default 1.0)
    
    Returns:
        Reconstructed mesh
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, falling back to original mesh")
        return mesh.copy()
    
    radii_factor = params.get("radii_factor", 1.0)
    
    try:
        # Convert to point cloud
        pcd = _trimesh_to_open3d_pointcloud(mesh)
        
        # Compute average point spacing
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        
        # Calculate radii based on point spacing
        radii = [avg_dist * radii_factor * mult for mult in [0.5, 1.0, 2.0, 4.0]]
        
        logger.info(f"Running Ball Pivoting with radii: {radii}")
        
        reconstructed = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        
        result = _open3d_mesh_to_trimesh(reconstructed)
        
        logger.info(f"Ball pivoting complete: {len(result.vertices)} vertices, {len(result.faces)} faces")
        
        return result
        
    except Exception as e:
        logger.error(f"Ball pivoting reconstruction failed: {e}")
        _log_reconstruction_failure("reconstruct_ball_pivot", str(e), mesh)
        return mesh.copy()


@register_action(
    name="open3d_alpha_shape",
    description="Alpha shape surface reconstruction",
    parameters={"alpha": None},
    risk_level="high"
)
def action_open3d_alpha_shape(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Alpha Shape surface reconstruction.
    
    Creates a shape from point cloud using alpha complex.
    Good for point clouds with varying density.
    
    Args:
        mesh: Input mesh (treated as point cloud)
        params:
            - alpha: Alpha value (None = auto-calculate optimal)
    
    Returns:
        Reconstructed mesh
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Open3D not available, falling back to original mesh")
        return mesh.copy()
    
    alpha = params.get("alpha")
    
    try:
        pcd = _trimesh_to_open3d_pointcloud(mesh)
        
        # Auto-calculate alpha if not provided
        if alpha is None:
            distances = pcd.compute_nearest_neighbor_distance()
            alpha = np.mean(distances) * 2.0
        
        logger.info(f"Running Alpha Shape with alpha={alpha}")
        
        reconstructed = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            alpha
        )
        
        result = _open3d_mesh_to_trimesh(reconstructed)
        
        logger.info(f"Alpha shape complete: {len(result.vertices)} vertices, {len(result.faces)} faces")
        
        return result
        
    except Exception as e:
        logger.error(f"Alpha shape reconstruction failed: {e}")
        _log_reconstruction_failure("reconstruct_alpha_shape", str(e), mesh)
        return mesh.copy()


@register_action(
    name="morphological_voxel_reconstruct",
    description="Voxel reconstruction with morphological gap filling",
    parameters={"resolution": 100, "dilation_iterations": 2, "erosion_iterations": 1},
    risk_level="high"
)
def action_morphological_voxel_reconstruct(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Advanced voxel reconstruction with morphological operations.
    
    Voxelizes the mesh, then applies morphological dilation to
    connect nearby fragments, followed by erosion to restore
    the approximate original shape.
    
    Best for:
    - Fragments that should connect but have small gaps
    - Noisy meshes with holes
    
    Args:
        mesh: Input mesh
        params:
            - resolution: Voxel grid resolution (default 100)
            - dilation_iterations: Dilation to fill gaps (default 2)
            - erosion_iterations: Erosion to restore shape (default 1)
    
    Returns:
        Reconstructed mesh
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available for morphological operations, using basic voxelization")
        # Fall back to basic voxelization
        try:
            bbox = mesh.bounds
            bbox_size = bbox[1] - bbox[0]
            pitch = max(bbox_size) / params.get("resolution", 100)
            voxels = mesh.voxelized(pitch=pitch)
            return voxels.marching_cubes
        except Exception as e:
            logger.error(f"Basic voxelization failed: {e}")
            return mesh.copy()
    
    resolution = params.get("resolution", 100)
    dilation_iter = params.get("dilation_iterations", 2)
    erosion_iter = params.get("erosion_iterations", 1)
    
    try:
        # Calculate voxel pitch
        bbox = mesh.bounds
        bbox_size = bbox[1] - bbox[0]
        pitch = max(bbox_size) / resolution
        
        logger.info(f"Voxelizing with resolution {resolution} (pitch={pitch:.4f})")
        
        # Voxelize
        voxels = mesh.voxelized(pitch=pitch)
        voxel_matrix = voxels.matrix.copy()
        
        logger.info(f"Initial voxels: {voxel_matrix.sum()} occupied")
        
        # Morphological closing: dilation followed by erosion
        # This fills small gaps between fragments
        if dilation_iter > 0:
            structure = ndimage.generate_binary_structure(3, 1)  # 6-connected
            voxel_matrix = ndimage.binary_dilation(
                voxel_matrix, 
                structure=structure, 
                iterations=dilation_iter
            )
            logger.info(f"After dilation ({dilation_iter}x): {voxel_matrix.sum()} occupied")
        
        if erosion_iter > 0:
            voxel_matrix = ndimage.binary_erosion(
                voxel_matrix, 
                structure=structure, 
                iterations=erosion_iter
            )
            logger.info(f"After erosion ({erosion_iter}x): {voxel_matrix.sum()} occupied")
        
        # Fill internal holes
        voxel_matrix = ndimage.binary_fill_holes(voxel_matrix)
        logger.info(f"After fill holes: {voxel_matrix.sum()} occupied")
        
        # Create new voxel grid and extract mesh
        new_voxels = trimesh.voxel.VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(voxel_matrix),
            transform=voxels.transform
        )
        
        result = new_voxels.marching_cubes
        
        logger.info(f"Reconstruction complete: {len(result.vertices)} vertices, {len(result.faces)} faces")
        
        return result
        
    except Exception as e:
        logger.error(f"Morphological voxel reconstruction failed: {e}")
        _log_reconstruction_failure("reconstruct_voxel_morph", str(e), mesh)
        return mesh.copy()


@register_action(
    name="shrinkwrap_reconstruct",
    description="Shrinkwrap envelope reconstruction",
    parameters={"subdivision_level": 3, "iterations": 50, "method": "project"},
    risk_level="high"
)
def action_shrinkwrap_reconstruct(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Shrinkwrap surface reconstruction.
    
    Creates a simple envelope mesh (icosphere) and iteratively
    projects it onto the fragment point cloud, similar to
    Blender's shrinkwrap modifier or Materialise Magics.
    
    Best for:
    - Complex fragmented meshes where you want to preserve concavities
    - Meshes where voxelization loses too much detail
    
    Args:
        mesh: Input mesh
        params:
            - subdivision_level: Icosphere subdivisions (2-5, default 3)
            - iterations: Projection iterations (default 50)
            - method: 'project' or 'nearest' (default 'project')
    
    Returns:
        Shrinkwrapped mesh
    """
    subdiv_level = params.get("subdivision_level", 3)
    iterations = params.get("iterations", 50)
    method = params.get("method", "project")
    
    try:
        # Create a starting mesh - icosphere scaled to bounding box
        ico = trimesh.creation.icosphere(subdivisions=subdiv_level)
        
        # Scale to mesh bounding box with some padding
        bbox = mesh.bounds
        center = (bbox[0] + bbox[1]) / 2
        size = bbox[1] - bbox[0]
        
        ico.vertices = ico.vertices * (max(size) * 0.6) + center
        
        logger.info(f"Starting shrinkwrap with {len(ico.vertices)} vertices, {iterations} iterations")
        
        # Get target points (all vertices from fragments)
        target_points = mesh.vertices
        
        # Build KD-tree for fast nearest neighbor lookup
        from scipy.spatial import cKDTree
        tree = cKDTree(target_points)
        
        # Iterative projection
        for i in range(iterations):
            # Find nearest target point for each vertex
            distances, indices = tree.query(ico.vertices)
            
            # Move vertices toward targets
            # Use decreasing step size for convergence
            step = 0.5 * (1 - i / iterations)
            
            if method == "project":
                # Project along vertex normals toward target
                ico.vertices = ico.vertices + step * (target_points[indices] - ico.vertices)
            else:  # nearest
                # Simply move toward nearest point
                ico.vertices = ico.vertices * (1 - step) + target_points[indices] * step
            
            # Smooth to prevent self-intersection (every 10 iterations)
            if i % 10 == 9:
                trimesh.smoothing.filter_laplacian(ico, lamb=0.3, iterations=1)
        
        logger.info(f"Shrinkwrap complete: {len(ico.vertices)} vertices, {len(ico.faces)} faces")
        
        return ico
        
    except Exception as e:
        logger.error(f"Shrinkwrap reconstruction failed: {e}")
        _log_reconstruction_failure("reconstruct_shrinkwrap", str(e), mesh)
        return mesh.copy()


@register_action(
    name="fragment_aware_reconstruct",
    description="Intelligent fragment analysis and reconstruction",
    parameters={"min_fragment_faces": 10, "merge_distance": None},
    risk_level="high"
)
def action_fragment_aware_reconstruct(mesh: trimesh.Trimesh, params: dict) -> trimesh.Trimesh:
    """
    Fragment-aware reconstruction.
    
    Analyzes fragments to determine the best reconstruction strategy:
    1. If fragments have consistent orientation - use Poisson
    2. If fragments are noisy/random - use voxelization
    3. If fragments form a clear boundary - use alpha wrapping
    
    Args:
        mesh: Input mesh
        params:
            - min_fragment_faces: Minimum faces to consider a fragment (default 10)
            - merge_distance: Distance to merge nearby vertices (auto if None)
    
    Returns:
        Reconstructed mesh using the best detected strategy
    """
    min_faces = params.get("min_fragment_faces", 10)
    merge_dist = params.get("merge_distance")
    
    try:
        # Split into components
        components = mesh.split(only_watertight=False)
        num_components = len(components)
        
        logger.info(f"Analyzing {num_components} fragments")
        
        # Filter tiny fragments (noise)
        significant_components = [c for c in components if len(c.faces) >= min_faces]
        noise_count = num_components - len(significant_components)
        
        if noise_count > 0:
            logger.info(f"Filtered {noise_count} noise fragments (< {min_faces} faces)")
        
        if len(significant_components) == 0:
            logger.warning("No significant fragments found")
            return mesh.copy()
        
        # Analyze fragment characteristics
        total_vertices = sum(len(c.vertices) for c in significant_components)
        
        # Check if we have normals and if they're consistent
        has_consistent_normals = True
        try:
            combined_normals = np.vstack([c.vertex_normals for c in significant_components])
            normal_variance = np.var(combined_normals, axis=0).mean()
            has_consistent_normals = normal_variance < 0.5
        except Exception:
            has_consistent_normals = False
        
        # Choose strategy
        if num_components > 1000:
            # Extreme fragmentation - use voxelization with morphological ops
            logger.info("Strategy: Morphological voxel reconstruction (extreme fragmentation)")
            return action_morphological_voxel_reconstruct(mesh, {
                "resolution": 150,
                "dilation_iterations": 3,
                "erosion_iterations": 2
            })
        elif has_consistent_normals and OPEN3D_AVAILABLE:
            # Good normals - use Poisson
            logger.info("Strategy: Screened Poisson (consistent normals detected)")
            return action_open3d_screened_poisson(mesh, {"depth": 9})
        else:
            # Fall back to morphological voxel
            logger.info("Strategy: Morphological voxel reconstruction (fallback)")
            return action_morphological_voxel_reconstruct(mesh, {
                "resolution": 100,
                "dilation_iterations": 2,
                "erosion_iterations": 1
            })
        
    except Exception as e:
        logger.error(f"Fragment-aware reconstruction failed: {e}")
        _log_reconstruction_failure("reconstruct_fragments", str(e), mesh)
        return mesh.copy()
