# from flask import Flask, render_template, request, jsonify
# import pickle
# import networkx as nx
# import json
# import os
# import traceback  # 🔍 예외 추적용

# app = Flask(__name__, template_folder="templates")

# # ===================================
# # 위치 좌표 로딩
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
# @app.route('/route', methods=['POST'])
# def get_route():
#     data = request.get_json()
#     start_id = str(data.get("start")).strip()
#     end_id = str(data.get("end")).strip()

#     if start_id not in G.nodes or end_id not in G.nodes:
#         return jsonify({"error": f"입력한 콘존명이 그래프에 없습니다: {start_id} 또는 {end_id}"}), 400

#     try:
#         path_nodes, path_length = cch.query(start_id, end_id)

#         # ✅ 대안 2: 출발지와 도착지 좌표만 사용
#         start_coord = location_map.get(start_id)
#         end_coord = location_map.get(end_id)

#         coords = []

#         if isinstance(start_coord, (list, tuple)) and len(start_coord) == 2:
#             coords.append([float(start_coord[0]), float(start_coord[1])])
#         else:
#             coords.append([0.0, 0.0])  # fallback

#         if isinstance(end_coord, (list, tuple)) and len(end_coord) == 2:
#             coords.append([float(end_coord[0]), float(end_coord[1])])
#         else:
#             coords.append([0.0, 0.0])  # fallback

#         return jsonify({
#             "start": start_id,
#             "end": end_id,
#             "length": path_length,
#             "path": path_nodes,  # 그대로 보냄
#             "coordinates": coords
#         })

#     except Exception as e:
#         import traceback
#         print("❌ 예외 발생:\n", traceback.format_exc())
#         return jsonify({"error": f"서버 오류: {str(e)}"}), 500


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port)


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port)

from flask import Flask, render_template, request, jsonify
import pickle
import networkx as nx
import os

app = Flask(__name__, template_folder="templates")

# ===================================
# 그래프 및 위치 정보 포함된 pkl 로딩
# ===================================
def load_graph_with_location(path="graph_with_location.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

G = load_graph_with_location()

# ===================================
# shortcuts 및 order 불러오기
# ===================================
def load_shortcuts_and_order():
    with open("shortcuts.pkl", "rb") as f: shortcuts = pickle.load(f)
    with open("order.pkl", "rb") as f: order = pickle.load(f)
    return shortcuts, order

shortcuts, order = load_shortcuts_and_order()

# ===================================
# CCH 구조
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
@app.route('/route', methods=['POST'])
def get_route():
    data = request.get_json()
    start_id = str(data.get("start")).strip()
    end_id = str(data.get("end")).strip()

    if start_id not in G.nodes or end_id not in G.nodes:
        return jsonify({"error": f"입력한 콘존명이 그래프에 없습니다: {start_id} 또는 {end_id}"}), 400

    try:
        # 경로 계산
        path_nodes, path_length = cch.query(start_id, end_id)

        # 시작/도착지 좌표만 추출
        coords = []
        for node in [start_id, end_id]:
            coord = location_map.get(node)
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                coords.append([float(coord[0]), float(coord[1])])
            else:
                print(f"[❗경고] '{node}'의 좌표 없음 또는 형식 오류 → 기본값 사용")
                coords.append([0.0, 0.0])

        return jsonify({
            "start": start_id,
            "end": end_id,
            "length": path_length,
            "path": [start_id, end_id],  # 실제 경로는 생략
            "coordinates": coords        # 지도에는 시작-도착만 그림
        })

    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
