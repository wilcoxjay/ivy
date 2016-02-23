#
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
"""
Use DOT to layout a graph for cytoscape.js

TODO: add support for middle points in edges

This version calls dot as a subprocess, so it doesn't require any
python package, and just requires that graphviz is installed (dot
should run from command line).
"""

from __future__ import division
from collections import deque, defaultdict
import subprocess

from ivy_utils import topological_sort


def cubic_bezier_point(p0, p1, p2, p3, t):
    """
    https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Cubic_B.C3.A9zier_curves
    """
    a = (1.0 - t)**3
    b = 3.0 * t * (1.0 - t)**2
    c = 3.0 * t**2 * (1.0 - t)
    d = t**3
    return {
        "x": a * p0["x"] + b * p1["x"] + c * p2["x"] + d * p3["x"],
        "y": a * p0["y"] + b * p1["y"] + c * p2["y"] + d * p3["y"],
    }


def square_distance_to_segment(p, p1, p2):
    v0 = (p["x"] - p1["x"], p["y"] - p1["y"])
    v1 = (p2["x"] - p1["x"], p2["y"] - p1["y"])
    v0sq = v0[0] * v0[0] + v0[1] * v0[1]
    v1sq = v1[0] * v1[0] + v1[1] * v1[1]
    prod  = v0[0] * v1[0] + v0[1] * v1[1]
    v2sq = prod * prod / v1sq
    if prod < 0:
        return v0sq
    elif v2sq < v1sq:
        return v0sq - v2sq
    else:
        v3 = (v0[0] - v1[0], v0[1] - v1[1])
        return v3[0] * v3[0] + v3[1] * v3[1]


def approximate_cubic_bezier(p0, p1, p2, p3, threshold=1.0, limit=1024):
    """
    Return an series of points whose segments approximate the given
    bezier curve
    """
    threshold_squared = threshold ** 2
    points = {  # dict mapping t values to points
        0.0: p0,
        1.0: p3,
    }
    to_check = deque([(0.0, 1.0)])
    while len(to_check) > 0 and len(points) < limit:
        l, r = to_check.popleft()
        pl = points[l]
        pr = points[r]
        m = (l + r) / 2.0
        pm = cubic_bezier_point(p0, p1, p2, p3, m)
        if square_distance_to_segment(pm, pl, pr) > threshold_squared:
            points[m] = pm
            to_check.append((l, m))
            to_check.append((m, r))
    return [points[t] for t in sorted(points.keys())]


def get_approximation_points(bspline):
    """
    Retrurn a series of points whose segments approximate the given
    bspline
    """
    result = []
    for i in range(0, len(bspline) - 3, 3):
        result.extend(approximate_cubic_bezier(
            bspline[i], bspline[i+1], bspline[i+2], bspline[i+3],
            threshold=4.0,
            limit=100,
        )[:-1])
    result.append(bspline[-1])
    return result


def repr_elements(elements):
    from widget_cy_graph import CyGraphWidget
    return repr(CyGraphWidget()._trait_to_json(elements))


def dot_layout(cy_elements, transitive_edges, edge_weight=None):
    """
    Get a CyElements object and augment it (in-place) with positions,
    widths, heights, and spline data from a dot based layout.

    transitive_edges is a list of obj's of transitive edges.
    edge_weight can be a dictionay mapping edge obj's to numeric weights.
    If edge_weight is None, the default is 10 for transitive edges and 0 otherwise.

    For example, the values used for the leader demo are:
    transitive_edges=('reach', 'le')
    edge_weight={'reach': 10, 'le': 10, 'id': 1}

    Returns the object.
    """
    transitive_edges = frozenset(transitive_edges)
    if edge_weight is None:
        edge_weight = defaultdict(int, ((obj, 10) for obj in transitive_edges))

    elements = list(cy_elements.elements)
    # open('g_before.txt', 'w').write(repr_elements(elements))

    nodes_by_id = dict(
        (e["data"]["id"], e)
        for e in elements if e["group"] == "nodes"
    )

    # sort element to make transitive relations appear top to bottom
    order = [
        (nodes_by_id[e["data"]["source"]], nodes_by_id[e["data"]["target"]])
        for e in elements if
        e["group"] == "edges" and
        e["data"]["obj"] in transitive_edges
    ]
    elements = topological_sort(elements, order, lambda e: e["data"]["id"])

    # group nodes by clusters
    clusters = defaultdict(list)
    nodes_with_no_cluster = []
    for e in elements:
        if e["group"] == "nodes":
            if e["data"]["cluster"] is not None:
                clusters[e["data"]["cluster"]].append(e)
            else:
                nodes_with_no_cluster.append(e)

    # create the dot input

    g = 'digraph {\n'

    # add nodes to the graph

    def dot_node(e):
        # support unicode in labels, and handle newlines and double quotes (currently not handling other characters)
        return u'{} [label="{}"];\n'.format(
            e["data"]["id"],
            e["data"]["label"].replace('\n', '\\n').replace('"', '\\"'),
        )

    for e in nodes_with_no_cluster:
        g += '  ' + dot_node(e)
    for i, k in enumerate(sorted(clusters.keys())):
        g += '  subgraph cluster_{} {{\n'.format(i)
        for e in clusters[k]:
            g += '    ' + dot_node(e)
        g += '  }\n'

    # add edges to the graph
    # make edges unique by color
    edges = [e for e in elements if e["group"] == "edges"]
    for i, e in enumerate(edges):
        g += '  {} -> {} [weight={},color="#{:06d}"];\n'.format(
            e["data"]["source"],
            e["data"]["target"],
            edge_weight[e["data"]["obj"]],
            i,
        )

    g += '}\n'

    # open('g_before.dot', 'w').write(g.encode('utf8'))

    # now call dot using the plain output format
    p = subprocess.Popen(
        ['dot', '-Tplain'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True, # to search PATH on windows
    )
    dot_output, dot_error = p.communicate(g.encode('utf8'))
    assert p.returncode == 0 and dot_error == '', dot_error

    # open('g_after.dot', 'w').write(dot_output)
    # # write g.png
    # p = subprocess.Popen(
    #     ['dot', '-Tpng'],
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     shell=True, # to search PATH on windows
    # )
    # png_output, png_error = p.communicate(g.encode('utf8'))
    # assert p.returncode == 0 and png_error == '', png_error
    # open('g.png', 'w').write(png_output)

    # now parse the dot output
    lines = dot_output.splitlines()
    assert lines[0].startswith('graph ')
    assert lines[-1] == 'stop'
    lines = lines[1:-1]
    dot_result = {}  # nodes kept by id (string), edges kept by index in edges (int)
    for line in lines:
        sp = line.split()
        if sp[0] == 'node':
            dot_result[sp[1]] = dict(
                x=float(sp[2]),
                y=float(sp[3]),
                width=float(sp[4]),
                height=float(sp[5]),
            )
        elif sp[0] == 'edge':
            assert sp[-1][0] == '#', line
            i = int(sp[-1][1:])
            n = int(sp[3])
            dot_result[i] = [float(x) for x in sp[4:4 + 2 * n]]
        else:
            assert False, line

    # now add layout info to elements

    # add layout info to nodes: position, width, height
    for e in elements:
        if e["group"] == "nodes":
            attr = dot_result[e["data"]["id"]]
            e["position"] = {
                "x": 72.0 * attr['x'],
                "y": -72.0 * attr['y'],
            }
            e["data"]["width"] = 72.0 * attr['width']
            e["data"]["height"] = 72.0 * attr['height']

    # add layout info to edges: bspline and approxpoints
    for i, e in enumerate(edges):
        coords = dot_result[i]
        bspline = [
            {"x": 72.0 * coords[i], "y": -72.0 * coords[i + 1]}
            for i in range(0, len(coords), 2)
        ]
        e["data"]["bspline"] = bspline
        e["data"]["approxpoints"] = get_approximation_points(bspline)

    # open('g_after.txt', 'w').write(repr_elements(elements))
    return cy_elements
