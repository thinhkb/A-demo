import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

# Define the graph
Graph_nodes = {
    'A': [('B', 2), ('E', 3), ('F', 5)],
    'B': [('C', 1), ('G', 9), ('D', 4)],
    'C': [('H', 7)],
    'D': [('G', 1), ('H', 2), ('I', 6)],
    'E': [('D', 6), ('J', 8)],
    'F': [('J', 4), ('K', 7)],
    'G': [('L', 3)],
    'H': [('L', 5), ('M', 6)],
    'I': [('M', 2), ('N', 3)],
    'J': [('N', 4)],
    'K': [('N', 5), ('O', 6)],
    'L': [('P', 8)],
    'M': [('P', 4), ('Q', 7)],
    'N': [('Q', 5), ('R', 9)],
    'O': [('R', 3)],
    'P': [('S', 6)],
    'Q': [('S', 4), ('T', 8)],
    'R': [('T', 7)],
    'S': [('G', 3), ('U', 2)],
    'T': [('U', 4)],
    'U': []
}

# Heuristic function (estimated distance to goal)
def heuristic(n):
    H_dist = {
        'A': 15, 'B': 12, 'C': 20, 'D': 10, 'E': 13, 'F': 17, 'G': 9, 'H': 8, 'I': 7,
        'J': 10, 'K': 14, 'L': 6, 'M': 5, 'N': 4, 'O': 11, 'P': 3, 'Q': 2, 'R': 1,
        'S': 2, 'T': 1, 'U': 0
    }
    return H_dist.get(n, float('inf'))
def aStarAlgo(start_node, stop_node):

    open_set = set(start_node)
    closed_set = set()
    g = {}  # store distance from starting node
    parents = {}  # parents contains an adjacency map of all nodes

    # distance of starting node from itself is zero
    g[start_node] = 0
    # start_node is root node i.e it has no parent nodes
    # so start_node is set to its own parent node
    parents[start_node] = start_node
    while len(open_set) > 0:
        n = None

        # node with Lowest f() is found
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbors(n):
                # nodes 'm' not in first and last set are added to first
                # n is set its parent
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                #for each node m, compare its distance from start i.e g(m) to the
                #from start through n node
                else:
                    if g[m] > g[n] + weight:
                        #update g(m)
                        g[m] = g[n] + weight
                        #change parent of m to n
                        parents[m] = n

                        #if m in closed set, remove and add to open
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n == None:
            print('Path does not exist!')
            return None

        # if the current node is the stop_node
        # then we begin reconstructing the path from it to the start_node
        if n == stop_node:
            path = []

            while parents[n] != n:
                path.append(n)
                n = parents[n]

            path.append(start_node)

            path.reverse()

            print('Path found: {}'.format(path))
            print('Total Cost: {}'.format(g[stop_node]))
            return path, g[stop_node]

        # remove n from the open_List, and add it to closed_List
        # because all of his neighbors were inspected
        open_set.remove(n)
        closed_set.add(n)

    print('Path does not exist!')
    return None
def draw_graph(path=None):
    G = nx.DiGraph()

    # Add edges
    for node, edges in Graph_nodes.items():
        for neighbor, weight in edges:
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)  # Layout for positioning nodes
    plt.figure(figsize=(6, 4))

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=200, font_size=10, font_weight="bold")
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight path if exists
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)

    st.pyplot(plt)

# Streamlit UI
st.title("A* Pathfinding Algorithm")

# Dropdown to select start and stop nodes
nodes = list(Graph_nodes.keys())
start_node = st.selectbox("Start Node", nodes, index=0)
stop_node = st.selectbox("Stop Node", nodes, index=len(nodes) - 1)

# Show default graph on startup
draw_graph()

# Button to find path
if st.button("Find Path"):
    path, cost = aStarAlgo(start_node, stop_node)
    if path:
        st.success(f"Path found: {' -> '.join(path)}")
        st.info(f"Total Cost: {cost}")
        draw_graph(path)
    else:
        st.error("No path found!")