# from flask import Flask, render_template, request, jsonify
# import pickle
# import networkx as nx
# import json
# import os

# app = Flask(__name__, template_folder="templates")

# # ===================================
# # 위치 좌표 로딩 (콘존명 → 위경도)
# # ===================================
# def load_location_coords(path="locationinfoIc.json"):
#     with open(path, "r", encoding="utf-8") as f:
#         raw = json.load(f)
#         coords = {}
#         for item in raw.values():
#             try:
#                 name = item["http://data.ex.co.kr:80/link/def/icName"][0]["value"]
#                 y = float(item["http://data.ex.co.kr:80/link/def/yValue"][0]["value"])
#                 x = float(item["http://data.ex.co.kr:80/link/def/xValue"][0]["value"])
#                 coords[name.strip()] = (y, x)
#             except Exception:
#                 continue
#         return coords

# location_map = load_location_coords()

# # ===================================
# # .pkl 로딩
# # ===================================
# def load_preprocessed():
#     with open("graph.pkl", "rb") as f: G = pickle.load(f)
#     with open("order.pkl", "rb") as f: order = pickle.load(f)
#     with open("shortcuts.pkl", "rb") as f: shortcuts = pickle.load(f)
#     return G, order, shortcuts

# G, order, shortcuts = load_preprocessed()

# # ===================================
# # CCH 구조
# # ===================================
# class CCH:
#     def __init__(self, graph, shortcuts):
#         self.graph = graph
#         self.shortcuts = shortcuts
#         self.customized_graph = nx.DiGraph()
#         self.build_customized()

#     def build_customized(self):
#         for (u, v), weight in self.shortcuts.items():
#             self.customized_graph.add_edge(u, v, weight=weight)

#     def query(self, source, target):
#         return nx.bidirectional_dijkstra(self.customized_graph, source, target, weight='weight')

# cch = CCH(G, shortcuts)

# # ===================================
# # 라우팅
# # ===================================
# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/route', methods=['POST'])
# def get_route():
#     data = request.get_json()
#     start_id = str(data.get("start")).strip()
#     end_id = str(data.get("end")).strip()

#     if start_id not in G.nodes or end_id not in G.nodes:
#         return jsonify({"error": f"입력한 콘존명이 그래프에 없습니다: {start_id} 또는 {end_id}"}), 400

#     try:
#         path_nodes, path_length = cch.query(start_id, end_id)

#         # ✅ 오류 방지를 위한 좌표 처리
#         coords = []
#         for n in path_nodes:
#             coord = location_map.get(n)
#             if isinstance(coord, (list, tuple)) and len(coord) == 2:
#                 coords.append([float(coord[0]), float(coord[1])])
#             else:
#                 coords.append([0.0, 0.0])  # fallback

#         return jsonify({
#             "start": start_id,
#             "end": end_id,
#             "length": path_length,
#             "path": path_nodes,
#             "coordinates": coords
#         })

#     except Exception as e:
#         return jsonify({"error": f"서버 오류: {str(e)}"}), 500
        

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port)

from flask import Flask, render_template, request, jsonify
import pickle
import networkx as nx
import json
import os

app = Flask(__name__, template_folder="templates")

# ===================================
# 위치 좌표 로딩
# ===================================
def load_location_coords(path="locationinfoIc.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        coords = {}
        for item in raw.values():
            try:
                name = item["http://data.ex.co.kr:80/link/def/icName"][0]["value"]
                y = float(item["http://data.ex.co.kr:80/link/def/yValue"][0]["value"])
                x = float(item["http://data.ex.co.kr:80/link/def/xValue"][0]["value"])
                coords[name.strip()] = (y, x)
            except Exception:
                continue
        return coords

location_map = load_location_coords()

# ===================================
# pkl 로딩
# ===================================
def load_preprocessed():
    with open("graph.pkl", "rb") as f: G = pickle.load(f)
    with open("order.pkl", "rb") as f: order = pickle.load(f)
    with open("shortcuts.pkl", "rb") as f: shortcuts = pickle.load(f)
    return G, order, shortcuts

G, order, shortcuts = load_preprocessed()

# ===================================
# CCH 정의
# ===================================
class CCH:
    def __init__(self, graph, shortcuts):
        self.graph = graph
        self.shortcuts = shortcuts
        self.customized_graph = nx.DiGraph()
        self.build_customized()

    def build_customized(self):
        for (u, v), weight in self.shortcuts.items():
            self.customized_graph.add_edge(u, v, weight=weight)

    def query(self, source, target):
        return nx.bidirectional_dijkstra(self.customized_graph, source, target, weight='weight')

cch = CCH(G, shortcuts)

# ===================================
# 라우팅
# ===================================
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

        print("📍 [DEBUG] path_nodes =", path_nodes)

        coords = []
        for n in path_nodes:
            coord = location_map.get(n)
            print(f"🔎 노드: {n} | 좌표값: {coord} | 타입: {type(coord)}")  # ✅ 디버깅 출력
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                coords.append([float(coord[0]), float(coord[1])])
            else:
                print(f"⚠️ 경고: {n}의 좌표가 비정상입니다. fallback 적용.")
                coords.append([0.0, 0.0])  # fallback

        return jsonify({
            "start": start_id,
            "end": end_id,
            "length": path_length,
            "path": path_nodes,
            "coordinates": coords
        })

    except Exception as e:
        print(f"❌ 서버 내부 오류: {e}")  # 콘솔 로그에 출력
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

