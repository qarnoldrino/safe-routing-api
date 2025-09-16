# app.py

from flask import Flask, request, jsonify
import networkx as nx
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import mapping
import requests
import os

# ---------------------------------------
# Helper: download files from Google Drive
# ---------------------------------------
def download_if_missing(drive_file_id: str, filename: str):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?export=download&id={drive_file_id}"
        print(f"Downloading {filename} from Google Drive (id={drive_file_id})...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {filename}.")

# ---------------------------------------
# Google Drive file IDs
# ---------------------------------------
GRAPHML_ID = "1otBqeeWgyLGFWWkxCs7MtVAd76Ob-WWW"
EDGES_ID   = "1L65iNRi09VxivAtqJTL8iUbO2P8C3yxd"
RISK_ID    = "1tNvOlMHI8WyFuYpKIcBkR8ZoQydFsHYI"

# ---------------------------------------
# Download large data files if missing
# ---------------------------------------
download_if_missing(GRAPHML_ID, "chicago_walk_simplified.graphml")
download_if_missing(EDGES_ID, "edges_gdf.parquet")
download_if_missing(RISK_ID, "risk_surface_monthly.parquet")

# ---------------------------------------
# Load graph and risk data once
# ---------------------------------------
print("Loading graph and data...")
G = nx.read_graphml("chicago_walk_simplified.graphml")
edges_gdf = gpd.read_parquet("edges_gdf.parquet")
risk_surface = pd.read_parquet("risk_surface_monthly.parquet")
print("Loaded network + risk surface.")

# Risk lookup
risk_lookup = risk_surface.set_index(["edge_id","bin_of_day"])["mu_hat_month"].to_dict()

# Patch edge IDs and lengths into graph
edge_map = edges_gdf[["u","v","key","edge_id","length_m"]]
lookup = edge_map.set_index(["u","v","key"]).to_dict("index")
for u, v, k, data in G.edges(keys=True, data=True):
    rec = lookup.get((int(u), int(v), int(k)))
    if rec:
        data["edge_id"] = int(rec["edge_id"])
        data["length_m"] = float(rec["length_m"])

# ---------------------------------------
# Routing logic
# ---------------------------------------
def route_eval(G, src, dst, bin_of_day, alpha):
    def edge_cost(u, v, data):
        mu = risk_lookup.get((data.get("edge_id"), bin_of_day), 0.0)
        return float(data.get("length_m", 1.0)) + alpha * mu

    path = nx.shortest_path(G, src, dst, weight=edge_cost)
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]

    # Compute simple metrics
    dist = 0.0
    pred_risk = 0.0
    # iterate edges
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v, 0)  # assuming key=0; or adapt if needed
        dist += float(edge_data.get("length_m", edge_data.get("length", 0.0)))
        pred_risk += risk_lookup.get((edge_data.get("edge_id"), bin_of_day), 0.0)

    return coords, dist, pred_risk

# ---------------------------------------
# Flask app
# ---------------------------------------
app = Flask(__name__)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/route", methods=["POST"])
def route():
    try:
        data = request.get_json(force=True)
        src_lat = float(data["src"]["lat"])
        src_lon = float(data["src"]["lon"])
        dst_lat = float(data["dst"]["lat"])
        dst_lon = float(data["dst"]["lon"])
        bin_of_day = int(data.get("bin", 2))
        alpha = float(data.get("alpha", 4))

        # Snap to nearest nodes
        src = ox.nearest_nodes(G, src_lon, src_lat)
        dst = ox.nearest_nodes(G, dst_lon, dst_lat)

        coords, dist, pred_risk = route_eval(G, src, dst, bin_of_day, alpha)

        geojson = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {}
        }

        return jsonify({
            "route": geojson,
            "metrics": {
                "distance_m": dist,
                "pred_risk": pred_risk,
                "alpha": alpha,
                "bin_of_day": bin_of_day
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
