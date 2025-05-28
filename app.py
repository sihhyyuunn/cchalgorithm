/from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
import numpy as np
import random
import os
import pickle

app = Flask(__name__, template_folder="templates")

# =========================
# 엑셀에서 그래프 생성
# =========================
def load_graph_from_excel(path="input.xlsx"):
    df = pd.read_excel(path)
    df = df[['콘존ID', '교통량', '콘존길이', '시작노드ID', '종료노드ID']].dropna()
    df['시작노드ID'] = df['시작노드ID'].astype(str)
    df['종료노드ID'] = df['종료노드ID'].astype(str)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        u, v = row['시작노드ID'], row['종료노드ID']
        weight = row['교통량']
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
            G[u][v]['count'] += 1
        else:
            G.add_edge(u, v, weight=weight, count=1)

    for u, v, data in G.edges(data=True):
        data['weight'] = data['weight'] / data['count']

    return G

# =========================
# pickle 저장 및 불러오기
# =========================
def save_preprocessed(graph, order, shortcuts):
    with open("graph.pkl", "wb") as f: pickle.dump(graph, f)
    with open("order.pkl", "wb") as f: pickle.dump(order, f)
    with open("shortcuts.pkl", "wb") as f: pickle.dump(shortcuts, f)

def load_preprocessed():
    try:
        with open("graph.pkl", "rb") as f: graph = pickle.load(f)
        with open("order.pkl", "rb") as f: order = pickle.load(f)
        with open("shortcuts.pkl", "rb") as f: shortcuts = pickle.load(f)
        return graph, order, shortcuts
    except:
        return None, None, None

def is_input_updated():
    try:
        return os.path.getmtime("input.xlsx") > os.path.getmtime("graph.pkl")
    except:
        return True

# =========================
# ACO 최적화
# =========================
class ACO_CCH:
    def __init__(self, graph, num_ants=5, alpha=1, beta=2, evaporation=0.5, iterations=5):
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.iterations = iterations
        self.node_list = list(self.graph.nodes)
        self.index_map = {node: idx for idx, node in enumerate(self.node_list)}
        self.pheromone = np.ones((len(self.node_list), len(self.node_list)))

    def heuristic(self, u, v):
        if self.graph.has_edge(u, v):
            return 1.0 / (self.graph[u][v]['weight'] + 1e-5)
        return 1e-5

    def update_pheromone(self, solutions):
        self.pheromone *= (1 - self.evaporation)
        for solution, cost in solutions:
            deposit = 1.0 / (cost + 1e-5)
            for i in range(len(solution) - 1):
                u_idx = self.index_map[solution[i]]
                v_idx = self.index_map[solution[i + 1]]
                self.pheromone[u_idx][v_idx] += deposit

    def construct_solution(self):
        nodes = self.node_list.copy()
        random.shuffle(nodes)
        solution = [nodes.pop(0)]
        while nodes:
            current = solution[-1]
            probs = []
            total = 0
            for node in nodes:
                u_idx = self.index_map[current]
                v_idx = self.index_map[node]
                tau = self.pheromone[u_idx][v_idx] ** self.alpha
                eta = self.heuristic(current, node) ** self.beta
                w = tau * eta
                probs.append(w)
                total += w
            probs = np.array(probs)
            probs = probs / total if total > 0 else np.ones(len(nodes)) / len(nodes)
            next_node = np.random.choice(nodes, p=probs)
            solution.append(next_node)
            nodes.remove(next_node)
        return solution

    def evaluate_solution(self, solution):
        total = 0
        for i in range(len(solution) - 1):
            u, v = solution[i], solution[i + 1]
            if self.graph.has_edge(u, v):
                total += self.graph[u][v]['weight']
            else:
                total += 1e6
        return total

    def optimize_order(self):
        best_order = None
        best_cost = float('inf')
        for _ in range(self.iterations):
            solutions = [(self.construct_solution(), 0) for _ in range(self.num_ants)]
            solutions = [(sol, self.evaluate_solution(sol)) for sol, _ in solutions]
            best_ant = min(solutions, key=lambda x: x[1])
            if best_ant[1] < best_cost:
                best_order, best_cost = best_ant
            self.update_pheromone(solutions)
        return best_order

# =========================
# CCH 구조 최적화
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

# =========================
# 초기화 실행
# =========================
if is_input_updated():
    G = load_graph_from_excel()
    aco = ACO_CCH(G)
    order = aco.optimize_order()
    cch = CCH(G)
    cch.set_order(order)
    cch.contract_nodes()
    cch.customize()
    save_preprocessed(G, order, cch.shortcuts)
else:
    G, order, shortcuts = load_preprocessed()
    cch = CCH(G)
    cch.set_order(order)
    cch.shortcuts = shortcuts
    cch.customize()

# =========================
# Flask API
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
        return jsonify({
            "start": start_id,
            "end": end_id,
            "length": path_length,
            "path": path_nodes
        })
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
