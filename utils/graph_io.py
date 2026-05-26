import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely import wkt
import networkx as nx
import os
import random
from typing import List

import numpy as np
import torch


def load_graph_from_files(dpath, crs='EPSG:4326'):
    nodes = pd.read_csv(f'{dpath}/G_nodes.csv', index_col='osmid')
    edges = pd.read_csv(f'{dpath}/G_edges.csv', index_col=['u', 'v', 'key'])
    nodes['geometry'] = nodes['geometry'].apply(wkt.loads)
    edges['geometry'] = edges['geometry'].apply(wkt.loads)
    gdf_nodes = gpd.GeoDataFrame(nodes, geometry='geometry', crs=crs)
    gdf_edges = gpd.GeoDataFrame(edges, geometry='geometry', crs=crs)
    g = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
    return g, gdf_nodes, gdf_edges


def convert_to_line_graph(g, edge2idx):
    lg = nx.line_graph(g)
    dual_node_attrs = {e: g.get_edge_data(*e) for e in g.edges}
    nx.set_node_attributes(lg, dual_node_attrs)

    lg = nx.relabel.relabel_nodes(lg, edge2idx)
    sorted_lg = nx.MultiDiGraph()
    sorted_lg.add_nodes_from(sorted(lg.nodes(data=True)))
    sorted_lg.add_edges_from(lg.edges(data=True))
    return sorted_lg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_city_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def list_available_cities(data_root: str) -> List[str]:
    out = []
    if not os.path.exists(data_root):
        return out
    for name in sorted(os.listdir(data_root)):
        city_path = os.path.join(data_root, name)
        if not os.path.isdir(city_path):
            continue
        if os.path.exists(os.path.join(city_path, "G_nodes.csv")) and os.path.exists(os.path.join(city_path, "G_edges.csv")):
            out.append(name)
    return out


def stable_city_seed(base_seed: int, city: str) -> int:
    return int(base_seed + sum(ord(ch) for ch in city))


if __name__ == '__main__':
    for city in ['chengdu', 'porto', 'rome', 'sanfran']:
        city_path = f'data/{city}'
        g, nodes_gdf, edges_gdf = load_graph_from_files(city_path)
        lg = convert_to_line_graph(g, edges_gdf['edge_idx'].to_dict())
        print(city, lg)



