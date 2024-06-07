"""
Takes in vine-only point clouds and produces skeletal line segments with radii.
"""

import argparse
import multiprocessing
import numpy
import sys
from pathlib import Path
from collections import Counter
from scipy.sparse import csgraph
from utils.cloud_utils import *
from tqdm import tqdm 

from utils.graph_utils import (
    calculate_line_components,
    calculate_mst,
    close_mst_cycles,
    consolidate_lines,
    consolidate_vis_lines,
    construct_graph,
    visualize_graph,
    visualize_as_mesh,
)


# Filters that get applied to the point cloud (name and kwargs). The filter
# operations should either be a function of the open3d point cloud, or the
# smoothing() function from cloud_utils
FILTERS = (
    ("smoothing", {"radius": 0.01}),
    ("remove_statistical_outlier", {"nb_neighbors": 30, "std_ratio": 0.75}),
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--barb-threshold",
        help="Length along MST path beneath which we call a branch a barb (m).",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--close-cycles",
        help="Whether to close cycles in the MST mesh.",
        action="store_true",
    )
    parser.add_argument(
        "--cloud-paths",
        help="Path or paths to cloud files (space-separated if multiple)."
        " Originally .ply point clouds were used, but anything handled by"
        " open3d.io.read_point_cloud will work. Clouds will be added together"
        " if there are multiple files.",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ignorable-length",
        help="Length along MST path beneath which discard the cluster (m).",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--min-point-count",
        help="Number of points below which we will discard the cluster.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--prior-radius",
        help="Prior on vine radius (m).",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--save-dir",
        help="Directory to save intermediate and final files in.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--segment-length",
        help="Length along the graph (m) to roughly split into line segments.",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--verbose",
        help="Adds series of debug printouts.",
        action="store_true",
    )
    parser.add_argument(
        "--vis-mst-mesh",
        help="Whether to visualize mesh of MST graph (slow).",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--vis-refine-process",
        help="Whether to visualize various sub-parts of the refinement (slow).",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--vis-subgraphs",
        help="Whether to export all subgraphs (based on topology), is most"
        " useful when debugging the subgraph process (slow).",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--voxel-size-initial",
        help="Size (m) to initially downsample cloud to.",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--weight-prior",
        help="In radius estimation, weight of 'prior' term (1 is equal weight"
        " to the point fitting component).",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--weight-smooth",
        help="In radius estimation, weight of 'smooth' term (1 is equal weight"
        " to the point fitting component).",
        type=float,
        default=0.1,
    )
    return parser.parse_args()

def main():
    args = _parse_args()

    for path in args.cloud_paths:
        assert path.is_file(), f"--cloud-paths {path} was not a valid file"

    assert (
        args.save_dir.is_dir()
    ), f"--save-dir {args.save_dir} was not a valid directory"

    for name in (
        "barb_threshold",
        "ignorable_length",
        "prior_radius",
        "segment_length",
        "voxel_size_initial",
        "weight_prior",
        "weight_smooth",
    ):
        value = getattr(args, name)
        assert value >= 0, f"Value for {name}: {value} invalid"

    assert (
        args.barb_threshold < args.ignorable_length
    ), "Barb threshold should be shorter than the ignorable length"

    parameters = {
        "seg_len": args.segment_length,
        "prior_radius": args.prior_radius,
        "prior_weight": args.weight_prior,
        "smooth_weight": args.weight_smooth,
    }

    ###########################################################################
    pt_cloud = load_ptclouds_npz(args.cloud_paths)

    cloud, graph, dist_graph = construct_graph(
        pt_clouds=pt_cloud,
        save_cloud_dir=args.save_dir,
        save_graph_dir=args.save_dir,
        filtering=FILTERS,
        use_density=False,
        final_voxel_size=args.voxel_size_initial,
    )
    visualize_graph(graph=graph, save_dir=args.save_dir)
    visualize_as_mesh(
        points=numpy.asarray(cloud.points),
        graph=graph,
        save_dir=args.save_dir,
        colormap={"vmax": numpy.max(graph), "cmap": "plasma"},
    )
    print("MST... ")
    mst = calculate_mst(
        graph=graph,
        dist_graph=dist_graph,
        points=numpy.asarray(cloud.points),
        save_dir=args.save_dir,
        viz_dir=args.save_dir,
        vis_mst_mesh=args.vis_mst_mesh,
    )
    print("Closing MST cycles...")
    cyclic_mst = close_mst_cycles(
        close_cycles=args.close_cycles,
        mst=mst,
        graph=dist_graph,
        points=numpy.asarray(cloud.points),
        save_dir=args.save_dir,
        viz_dir=args.save_dir,
        vis_mst_mesh=args.vis_mst_mesh,
    )

    print("Opening multiple threads for calculating lines...")
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method("spawn", force=True)
    else:
        multiprocessing.set_start_method("forkserver", force=True)
    # TODO: Check which number of processes is fastest, not just max
    pool = multiprocessing.Pool(processes=16)

    workers = []
    # Go through the biggest labels first
    _, labels = csgraph.connected_components(
        csgraph=cyclic_mst,
        directed=False,
        return_labels=True,
    )
    points = numpy.asarray(cloud.points)
    filtered_labels = [
        (count, label)
        for label, count in Counter(labels).items()
        if count > args.min_point_count
    ]
    filtered_labels = sorted(filtered_labels, reverse=True)

    print(f'>>> starting loop 1')
    for _, label in tqdm(filtered_labels):
        worker = pool.apply_async(
            calculate_line_components,
            args=(
                cyclic_mst,
                points,
                labels,
                label,
                args.save_dir,
                args.save_dir,
                parameters,
                args.vis_refine_process,
                args.vis_subgraphs,
                args.verbose,
                args.ignorable_length,
                args.barb_threshold,
            ),
        )
        workers.append(worker)

    print(f'>>> waiting for workers:  ', end='')
    for worker in workers:
        worker.get()
        print(f'*', end='')

    print(f'>>> starting consolidate_lines()')
    lines, _ = consolidate_lines(args.save_dir)

    print(f'>>> done')
    consolidate_vis_lines(args.save_dir)


if __name__ == "__main__":
    main()
