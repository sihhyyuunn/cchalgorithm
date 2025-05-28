from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
import pickle
import json
import os

app = Flask(__name__, template_folder="templates")

# =========================
# 위치 좌표 로딩 함수
# =========================
def load_coords_from_json(path="locationinfoIc.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        coords = {}
        for item in raw_data.values():
            try:
                name = item["http://data.ex.co.kr:80/link/def/icName"][0]["value"]
                y = float(item["http://data.ex.co.kr:80/link/def/yValue"][0]["value"])
                x = float(item["http://data.ex.co.kr:80/link/def/xValue"][0]["value"])
                coords[name] = (y, x)
            except Exception:
                continue
        return coords

node_coords = load_coords_from_json()

# =========================
# pickle 불러오기
# =========================
def load_preprocessed():
    with open("graph.pkl", "rb") as f: graph = pickle.load(f)
    with open("order.pkl", "rb") as f: order = pickle.load(f)
    with open("shortcuts.pkl", "rb") as f: shortcuts = pickle.load(f)
    return graph, order, shortcuts

G, order, shortcuts = load_preprocessed()

# =========================
# CCH 구조 정의 및 로딩
# =========================
class CCH:
    def __init__(self, graph):
        self.graph = graph.copy()
        self.order = []
        self.shortcuts = {}
        self.customized_graph = nx.DiGraph()

    def set_order(self, order):
        self.order = order

    def add_shortcut(self, u, v, weight):
        if (u, v) not in self.shortcuts or self.shortcuts[(u, v)] > weight:
            self.shortcuts[(u, v)] = weight

    def contract_nodes(self, preserved_nodes=None):
        preserved_nodes = set(preserved_nodes) if preserved_nodes else set()
        for node in self.order:
            if node in preserved_nodes or node not in self.graph:
                continue
            in_edges = list(self.graph.in_edges(node, data=True))
            out_edges = list(self.graph.out_edges(node, data=True))
            for u, _, data_u in in_edges:
                for _, v, data_v in out_edges:
                    if u == v:
                        continue
                    total_weight = data_u['weight'] + data_v['weight']
                    self.add_shortcut(u, v, total_weight)
            self.graph.remove_node(node)

    def customize(self):
        self.customized_graph = nx.DiGraph()
        for (u, v), w in self.shortcuts.items():
            self.customized_graph.add_edge(u, v, weight=w)

    def query(self, source, target):
        return nx.bidirectional_dijkstra(self.customized_graph, source, target, weight='weight')

cch = CCH(G)
cch.set_order(order)
cch.shortcuts = shortcuts
cch.customize()

# =========================
# Flask 라우팅
# =========================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/route', methods=['POST'])
def get_route():
    data = request.get_json()
    start_id = str(data.get("start"))
    end_id = str(data.get("end"))

    if start_id not in G.nodes or end_id not in G.nodes:
        return jsonify({"error": f"입력한 노드ID가 그래프에 없습니다: {start_id} 또는 {end_id}"}), 400

    try:
        path_nodes, path_length = cch.query(start_id, end_id)
        coords = [node_coords.get(n, (0, 0)) for n in path_nodes]
        return jsonify({
            "start": start_id,
            "end": end_id,
            "length": path_length,
            "path": path_nodes,
            "coordinates": coords
        })
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
