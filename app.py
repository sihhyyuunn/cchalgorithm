from flask import Flask, render_template, request, jsonify
import networkx as nx
import pickle
import os
import json

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

location_map = load_coords_from_json()

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
# CCH 구조 정의
# =========================
class CCH:
    def __init__(self, graph, shortcuts):
        self.graph = graph.copy()
        self.shortcuts = shortcuts
        self.customized_graph = nx.DiGraph()
        for (u, v), w in self.shortcuts.items():
            self.customized_graph.add_edge(u, v, weight=w)

    def query(self, source, target):
        return nx.bidirectional_dijkstra(self.customized_graph, source, target, weight='weight')

cch = CCH(G, shortcuts)

# =========================
# Flask 라우팅
# =========================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/route', methods=['POST'])
def get_route():
    data = request.get_json()
    start_id = str(data.get("start")).strip()
    end_id = str(data.get("end")).strip()

    if start_id not in G.nodes or end_id not in G.nodes:
        return jsonify({"error": f"입력한 콘존명이 그래프에 없습니다: {start_id} 또는 {end_id}"}), 400

    try:
        path_nodes, path_length = cch.query(start_id, end_id)
        coords = [location_map.get(n, [0, 0]) for n in path_nodes]
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
