#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <tuple>
#include <cstdint>
#include <array>

namespace py = pybind11;

constexpr double PI = 3.14159265358979323846;

struct Node {
    int y;
    int x;

    bool operator==(const Node& other) const noexcept {
        return y == other.y && x == other.x;
    }
};

struct NodeHash {
    std::size_t operator()(const Node& n) const noexcept {
        return (static_cast<std::size_t>(static_cast<uint32_t>(n.y)) << 32)
             ^ static_cast<std::size_t>(static_cast<uint32_t>(n.x));
    }
};

struct Edge {
    Node to;
    double weight;
    double biased_weight;
};

using AdjList = std::unordered_map<Node, std::vector<Edge>, NodeHash>;
using NodeSet = std::unordered_set<Node, NodeHash>;

static inline double euclidean(const Node& a, const Node& b) {
    double dy = static_cast<double>(a.y - b.y);
    double dx = static_cast<double>(a.x - b.x);
    return std::sqrt(dy * dy + dx * dx);
}


static inline double distance_point_to_path(const Node& p, const std::vector<Node>& path) {
    if (path.empty()) {
        return 0.0;
    }

    double best = std::numeric_limits<double>::infinity();
    for (const auto& q : path) {
        double d = euclidean(p, q);
        if (d < best) {
            best = d;
        }
    }
    return best;
}

static Node nearest_node_to_point(const AdjList& g, const Node& ref) {
    bool first = true;
    Node best{0, 0};
    double best_dist = 0.0;

    for (const auto& kv : g) {
        const Node& n = kv.first;
        double d = euclidean(n, ref);
        if (first || d < best_dist) {
            first = false;
            best = n;
            best_dist = d;
        }
    }

    return best;
}

static inline std::uint64_t pack_node(const Node& n) {
    return (static_cast<std::uint64_t>(static_cast<uint32_t>(n.y)) << 32)
         | static_cast<uint32_t>(n.x);
}

static inline std::uint64_t edge_key(const Node& a, const Node& b) {
    std::uint64_t pa = pack_node(a);
    std::uint64_t pb = pack_node(b);
    if (pa > pb) std::swap(pa, pb);
    return pa ^ (pb + 0x9e3779b97f4a7c15ULL + (pa << 6) + (pa >> 2));
}

static py::array_t<int> nodes_to_numpy(const std::vector<Node>& nodes) {
    py::array_t<int> out(py::array::ShapeContainer{
        static_cast<py::ssize_t>(nodes.size()),
        static_cast<py::ssize_t>(2)
    });

    auto buf = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nodes.size()); ++i) {
        buf(i, 0) = nodes[static_cast<std::size_t>(i)].y;
        buf(i, 1) = nodes[static_cast<std::size_t>(i)].x;
    }
    return out;
}

static AdjList build_graph_from_points(
    py::array_t<int, py::array::c_style | py::array::forcecast> points
) {
    auto buf = points.unchecked<2>();
    if (buf.ndim() != 2 || buf.shape(1) != 2) {
        throw std::runtime_error("skeleton_points must have shape (N, 2)");
    }

    AdjList graph;
    NodeSet point_set;
    std::vector<Node> nodes;
    nodes.reserve(static_cast<std::size_t>(buf.shape(0)));
    point_set.reserve(static_cast<std::size_t>(buf.shape(0)) * 2);

    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        Node n{buf(i, 0), buf(i, 1)};
        nodes.push_back(n);
        point_set.insert(n);
        graph[n];  // ensure node exists
    }

    const double SQRT2 = std::sqrt(2.0);
    const std::vector<std::tuple<int, int, double>> steps = {
        {0, 1, 1.0},
        {1, 0, 1.0},
        {1, 1, SQRT2},
        {1, -1, SQRT2}
    };

    for (const auto& n : nodes) {
        for (const auto& step : steps) {
            int dy = std::get<0>(step);
            int dx = std::get<1>(step);
            double w = std::get<2>(step);

            Node nb{n.y + dy, n.x + dx};
            if (point_set.find(nb) != point_set.end()) {
                graph[n].push_back({nb, w, w});
                graph[nb].push_back({n, w, w});
            }
        }
    }

    return graph;
}

static int degree(const AdjList& g, const Node& n) {
    auto it = g.find(n);
    if (it == g.end()) return 0;
    return static_cast<int>(it->second.size());
}


static std::vector<Node> neighbors(const AdjList& g, const Node& n) {
    std::vector<Node> out;
    auto it = g.find(n);
    if (it == g.end()) return out;
    out.reserve(it->second.size());
    for (const auto& e : it->second) {
        out.push_back(e.to);
    }
    return out;
}

static std::vector<std::vector<Node>> connected_components(const AdjList& g) {
    std::vector<std::vector<Node>> components;
    NodeSet visited;
    visited.reserve(g.size() * 2);

    for (const auto& kv : g) {
        const Node& start = kv.first;
        if (visited.find(start) != visited.end()) {
            continue;
        }

        std::vector<Node> comp;
        std::queue<Node> q;
        q.push(start);
        visited.insert(start);

        while (!q.empty()) {
            Node cur = q.front();
            q.pop();
            comp.push_back(cur);

            auto it = g.find(cur);
            if (it == g.end()) continue;

            for (const auto& e : it->second) {
                if (visited.find(e.to) == visited.end()) {
                    visited.insert(e.to);
                    q.push(e.to);
                }
            }
        }

        components.push_back(std::move(comp));
    }

    return components;
}

static bool has_path(const AdjList& g, const Node& src, const Node& dst) {
    if (g.find(src) == g.end() || g.find(dst) == g.end()) return false;
    if (src == dst) return true;

    NodeSet visited;
    std::queue<Node> q;
    q.push(src);
    visited.insert(src);

    while (!q.empty()) {
        Node cur = q.front();
        q.pop();

        auto it = g.find(cur);
        if (it == g.end()) continue;

        for (const auto& e : it->second) {
            if (e.to == dst) return true;
            if (visited.find(e.to) == visited.end()) {
                visited.insert(e.to);
                q.push(e.to);
            }
        }
    }

    return false;
}


static std::vector<Node> shortest_path(
    const AdjList& g,
    const Node& src,
    const Node& dst,
    bool use_biased_weight = false
) {
    struct Item {
        double dist;
        Node node;
        bool operator>(const Item& other) const {
            return dist > other.dist;
        }
    };

    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    std::unordered_map<Node, double, NodeHash> dist;
    std::unordered_map<Node, Node, NodeHash> parent;

    dist.reserve(g.size() * 2);
    parent.reserve(g.size() * 2);

    for (const auto& kv : g) {
        dist[kv.first] = std::numeric_limits<double>::infinity();
    }

    auto it_src = g.find(src);
    auto it_dst = g.find(dst);
    if (it_src == g.end() || it_dst == g.end()) {
        return {};
    }

    dist[src] = 0.0;
    pq.push({0.0, src});

    while (!pq.empty()) {
        Item item = pq.top();
        pq.pop();

        if (item.node == dst) {
            break;
        }
        if (item.dist > dist[item.node]) {
            continue;
        }

        auto it = g.find(item.node);
        if (it == g.end()) {
            continue;
        }

        for (const auto& e : it->second) {
            double w = use_biased_weight ? e.biased_weight : e.weight;
            double nd = item.dist + w;

            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                parent[e.to] = item.node;
                pq.push({nd, e.to});
            }
        }
    }

    if (!std::isfinite(dist[dst])) {
        return {};
    }

    std::vector<Node> path;
    Node cur = dst;
    path.push_back(cur);

    while (!(cur == src)) {
        auto itp = parent.find(cur);
        if (itp == parent.end()) {
            return {};
        }
        cur = itp->second;
        path.push_back(cur);
    }

    std::reverse(path.begin(), path.end());
    return path;
}


static std::vector<Node> extend_path_to_endpoints(const AdjList& g, const std::vector<Node>& path) {
    if (path.size() < 2) {
        return path;
    }

    auto find_extreme_endpoint_forward =
        [&](const Node& anchor, const NodeSet& forbidden, bool want_min_y) -> std::vector<Node>
    {
        NodeSet visited;
        std::unordered_map<Node, Node, NodeHash> parent;
        std::queue<Node> q;

        visited.reserve(g.size() * 2);
        parent.reserve(g.size() * 2);

        visited.insert(anchor);
        q.push(anchor);

        while (!q.empty()) {
            Node cur = q.front();
            q.pop();

            auto it = g.find(cur);
            if (it == g.end()) continue;

            for (const auto& e : it->second) {
                const Node& nb = e.to;
                if (visited.find(nb) == visited.end() && forbidden.find(nb) == forbidden.end()) {
                    visited.insert(nb);
                    parent[nb] = cur;
                    q.push(nb);
                }
            }
        }

        std::vector<Node> endpoints_reached;
        for (const auto& n : visited) {
            if (!(n == anchor) && degree(g, n) == 1) {
                endpoints_reached.push_back(n);
            }
        }

        if (endpoints_reached.empty()) {
            for (const auto& n : visited) {
                if (!(n == anchor)) {
                    endpoints_reached.push_back(n);
                }
            }
        }

        if (endpoints_reached.empty()) {
            return {};
        }

        Node target = endpoints_reached[0];
        for (const auto& n : endpoints_reached) {
            if (want_min_y) {
                if (n.y < target.y) target = n;
            } else {
                if (n.y > target.y) target = n;
            }
        }

        std::vector<Node> ext_path;
        Node cur = target;
        ext_path.push_back(cur);

        while (!(cur == anchor)) {
            cur = parent[cur];
            ext_path.push_back(cur);
        }

        std::reverse(ext_path.begin(), ext_path.end());
        return ext_path;
    };

    std::vector<Node> path_list = path;
    NodeSet path_set(path_list.begin(), path_list.end());

    // Prolongation vers le haut
    Node top_anchor = path_list.front();
    NodeSet forbidden_top = path_set;
    forbidden_top.erase(top_anchor);

    std::vector<Node> ext_top = find_extreme_endpoint_forward(top_anchor, forbidden_top, true);
    if (ext_top.size() > 1) {
        std::vector<Node> prefix(ext_top.begin() + 1, ext_top.end());
        std::reverse(prefix.begin(), prefix.end());
        prefix.insert(prefix.end(), path_list.begin(), path_list.end());
        path_list = std::move(prefix);
    }

    path_set = NodeSet(path_list.begin(), path_list.end());

    // Prolongation vers le bas
    Node bottom_anchor = path_list.back();
    NodeSet forbidden_bottom = path_set;
    forbidden_bottom.erase(bottom_anchor);

    std::vector<Node> ext_bottom = find_extreme_endpoint_forward(bottom_anchor, forbidden_bottom, false);
    if (ext_bottom.size() > 1) {
        path_list.insert(path_list.end(), ext_bottom.begin() + 1, ext_bottom.end());
    }

    return path_list;
}

static void remove_nodes(AdjList& g, const std::vector<Node>& to_remove) {
    NodeSet removed(to_remove.begin(), to_remove.end());

    for (const auto& n : to_remove) {
        g.erase(n);
    }

    for (auto& kv : g) {
        auto& edges = kv.second;
        edges.erase(
            std::remove_if(
                edges.begin(),
                edges.end(),
                [&](const Edge& e) {
                    return removed.find(e.to) != removed.end();
                }
            ),
            edges.end()
        );
    }
}

static std::vector<std::vector<Node>> extract_branches_from_graph_cpp(const AdjList& g) {
    std::vector<std::vector<Node>> branches;
    std::unordered_set<std::uint64_t> visited_edges;

    auto comps = connected_components(g);

    for (const auto& comp_nodes : comps) {
        NodeSet comp_set(comp_nodes.begin(), comp_nodes.end());

        auto in_comp = [&](const Node& n) {
            return comp_set.find(n) != comp_set.end();
        };

        auto comp_neighbors = [&](const Node& n) {
            std::vector<Node> out;
            auto it = g.find(n);
            if (it == g.end()) return out;
            for (const auto& e : it->second) {
                if (in_comp(e.to)) {
                    out.push_back(e.to);
                }
            }
            return out;
        };

        auto comp_degree = [&](const Node& n) {
            return static_cast<int>(comp_neighbors(n).size());
        };

        // Étape 1 : partir des extrémités
        for (const auto& node : comp_nodes) {
            if (comp_degree(node) != 1) {
                continue;
            }

            Node current = node;
            std::vector<Node> path{current};

            while (true) {
                auto neighs = comp_neighbors(current);
                std::vector<Node> next_candidates;

                for (const auto& nb : neighs) {
                    auto ek = edge_key(current, nb);
                    if (visited_edges.find(ek) == visited_edges.end()) {
                        next_candidates.push_back(nb);
                    }
                }

                if (next_candidates.empty()) {
                    break;
                }

                Node nxt = next_candidates[0];
                visited_edges.insert(edge_key(current, nxt));
                path.push_back(nxt);
                current = nxt;

                if (comp_degree(current) != 2) {
                    break;
                }
            }

            if (path.size() > 1) {
                branches.push_back(std::move(path));
            }
        }

        // Étape 2 : cycles / restes
        for (const auto& u : comp_nodes) {
            auto neighs_u = comp_neighbors(u);
            for (const auto& v : neighs_u) {
                auto ek = edge_key(u, v);
                if (visited_edges.find(ek) != visited_edges.end()) {
                    continue;
                }

                Node current = u;
                std::vector<Node> path{current};

                while (true) {
                    auto neighs = comp_neighbors(current);
                    std::vector<Node> next_candidates;

                    for (const auto& nb : neighs) {
                        auto ek2 = edge_key(current, nb);
                        if (visited_edges.find(ek2) == visited_edges.end()) {
                            next_candidates.push_back(nb);
                        }
                    }

                    if (next_candidates.empty()) {
                        break;
                    }

                    Node nxt = next_candidates[0];
                    visited_edges.insert(edge_key(current, nxt));
                    path.push_back(nxt);
                    current = nxt;
                }

                if (path.size() > 1) {
                    branches.push_back(std::move(path));
                }
            }
        }
    }

    return branches;
}

static double path_length(const std::vector<Node>& path) {
    if (path.size() < 2) return 0.0;

    double total = 0.0;
    for (std::size_t i = 1; i < path.size(); ++i) {
        total += euclidean(path[i - 1], path[i]);
    }
    return total;
}

static py::dict analyze_skeleton_graph(
    py::array_t<int, py::array::c_style | py::array::forcecast> skeleton_points,
    py::object main_ref_path_obj = py::none(),
    double main_path_bias = 20.0
) {
    AdjList g = build_graph_from_points(skeleton_points);

    std::vector<Node> ref_path;
    bool has_ref_path = false;

    if (!main_ref_path_obj.is_none()) {
        auto ref_arr = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(main_ref_path_obj);
        if (ref_arr) {
            auto ref_buf = ref_arr.unchecked<2>();
            if (ref_buf.ndim() == 2 && ref_buf.shape(1) == 2 && ref_buf.shape(0) >= 2) {
                ref_path.reserve(static_cast<std::size_t>(ref_buf.shape(0)));
                for (py::ssize_t i = 0; i < ref_buf.shape(0); ++i) {
                    ref_path.push_back(Node{ref_buf(i, 0), ref_buf(i, 1)});
                }
                has_ref_path = true;
            }
        }
    }

    py::dict out;
    auto comps = connected_components(g);

    // Choisir la composante à plus grande amplitude verticale
    double best_span = -1.0;
    std::vector<Node> best_comp;

    for (const auto& comp : comps) {
        if (comp.size() < 2) {
            continue;
        }

        int ymin = comp[0].y;
        int ymax = comp[0].y;
        for (const auto& n : comp) {
            ymin = std::min(ymin, n.y);
            ymax = std::max(ymax, n.y);
        }

        double span = static_cast<double>(ymax - ymin);
        if (span > best_span) {
            best_span = span;
            best_comp = comp;
        }
    }

    if (has_ref_path && g.size() >= 2) {
        std::unordered_map<Node, double, NodeHash> node_dist;
        node_dist.reserve(g.size() * 2);

        double max_d = 0.0;
        for (const auto& kv : g) {
            const Node& n = kv.first;
            double d = distance_point_to_path(n, ref_path);
            node_dist[n] = d;
            if (d > max_d) {
                max_d = d;
            }
        }

        if (max_d <= 0.0) {
            max_d = 1.0;
        }

        double alpha = static_cast<double>(main_path_bias) * 2.0;

        for (auto& kv : g) {
            const Node& u = kv.first;
            for (auto& e : kv.second) {
                const Node& v = e.to;
                double d = (node_dist[u] + node_dist[v]) / 2.0;
                double penalty = 1.0 + alpha * (d / max_d) * (d / max_d);
                e.biased_weight = e.weight * penalty;
            }
        }
    }

    std::vector<Node> main_path;
    double main_len = 0.0;
    double secondary_len = 0.0;

    if (has_ref_path && g.size() >= 2) {
        Node ref_top = ref_path.front();
        Node ref_bottom = ref_path.back();

        Node start_node = nearest_node_to_point(g, ref_top);
        Node end_node = nearest_node_to_point(g, ref_bottom);

        if (has_path(g, start_node, end_node)) {
            main_path = shortest_path(g, start_node, end_node, true);
            main_path = extend_path_to_endpoints(g, main_path);
            main_len = path_length(main_path);
            remove_nodes(g, main_path);
        }
    }

    if (main_path.empty() && !best_comp.empty()) {
        std::vector<Node> endpoints;
        endpoints.reserve(best_comp.size());

        for (const auto& n : best_comp) {
            if (degree(g, n) == 1) {
                endpoints.push_back(n);
            }
        }

        if (endpoints.size() >= 2) {
            Node top = *std::min_element(
                endpoints.begin(), endpoints.end(),
                [](const Node& a, const Node& b) { return a.y < b.y; }
            );

            Node bottom = *std::max_element(
                endpoints.begin(), endpoints.end(),
                [](const Node& a, const Node& b) { return a.y < b.y; }
            );

            if (has_path(g, top, bottom)) {
                main_path = shortest_path(g, top, bottom, false);
                main_path = extend_path_to_endpoints(g, main_path);
                main_len = path_length(main_path);
                remove_nodes(g, main_path);
            }
        }
    }

    auto branches = extract_branches_from_graph_cpp(g);
    for (const auto& br : branches) {
        secondary_len += path_length(br);
    }

    py::list py_branches;
    for (const auto& br : branches) {
        py_branches.append(nodes_to_numpy(br));
    }

    out["main_path"] = nodes_to_numpy(main_path);
    out["secondary_branches"] = py_branches;
    out["main_path_length"] = main_len;
    out["secondary_length"] = secondary_len;
    out["branch_count"] = static_cast<int>(branches.size());
    out["component_count"] = static_cast<int>(comps.size());

    return out;
}


static inline bool in_bounds(int y, int x, int h, int w) {
    return (y >= 0 && y < h && x >= 0 && x < w);
}

static py::array_t<uint8_t> prune_terminal_spurs_cpp(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> skel_arr,
    int min_len_px,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> protect_arr,
    int max_iter = 50
) {
    auto skel_in = skel_arr.unchecked<2>();
    auto protect_in = protect_arr.unchecked<2>();

    if (skel_in.ndim() != 2 || protect_in.ndim() != 2) {
        throw std::runtime_error("skel and protect_mask must be 2D arrays");
    }
    if (skel_in.shape(0) != protect_in.shape(0) || skel_in.shape(1) != protect_in.shape(1)) {
        throw std::runtime_error("skel and protect_mask must have the same shape");
    }

    const int H = static_cast<int>(skel_in.shape(0));
    const int W = static_cast<int>(skel_in.shape(1));

    py::array_t<uint8_t> out(py::array::ShapeContainer{
        static_cast<py::ssize_t>(H),
        static_cast<py::ssize_t>(W)
    });
    auto skel = out.mutable_unchecked<2>();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            skel(y, x) = skel_in(y, x) ? 1 : 0;
        }
    }

    static const std::array<std::pair<int, int>, 8> neigh = {{
        {-1,-1}, {-1,0}, {-1,1},
        {0,-1},          {0,1},
        {1,-1},  {1,0},  {1,1}
    }};

    auto degree_at = [&](int y, int x) -> int {
        int d = 0;
        for (const auto& off : neigh) {
            int yy = y + off.first;
            int xx = x + off.second;
            if (in_bounds(yy, xx, H, W) && skel(yy, xx)) {
                ++d;
            }
        }
        return d;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<uint8_t> degree_map(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
        std::vector<std::pair<int, int>> endpoints;
        endpoints.reserve((H * W) / 20 + 8);

        auto idx = [&](int y, int x) -> size_t {
            return static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
        };

        // Recompute degrees
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (!skel(y, x)) continue;
                int d = degree_at(y, x);
                degree_map[idx(y, x)] = static_cast<uint8_t>(d);
                if (d == 1) {
                    endpoints.push_back({y, x});
                }
            }
        }

        bool removed_any = false;

        for (const auto& ep : endpoints) {
            int y0 = ep.first;
            int x0 = ep.second;

            if (protect_in(y0, x0)) {
                continue;
            }
            if (!skel(y0, x0)) {
                continue;
            }
            if (degree_map[idx(y0, x0)] != 1) {
                continue;
            }

            std::vector<std::pair<int, int>> path;
            path.reserve(static_cast<size_t>(std::max(2, min_len_px + 2)));

            int cur_y = y0;
            int cur_x = x0;
            int prev_y = -999999;
            int prev_x = -999999;

            path.push_back({cur_y, cur_x});

            bool blocked_by_protect = false;

            while (true) {
                int cur_deg = degree_map[idx(cur_y, cur_x)];
                if (cur_deg >= 3) {
                    break;
                }

                std::vector<std::pair<int, int>> nxts;
                nxts.reserve(2);

                for (const auto& off : neigh) {
                    int yy = cur_y + off.first;
                    int xx = cur_x + off.second;
                    if (!in_bounds(yy, xx, H, W)) continue;
                    if (!skel(yy, xx)) continue;
                    if (yy == prev_y && xx == prev_x) continue;
                    nxts.push_back({yy, xx});
                }

                if (nxts.empty()) {
                    break;
                }
                if (nxts.size() > 1) {
                    break;
                }

                prev_y = cur_y;
                prev_x = cur_x;
                cur_y = nxts[0].first;
                cur_x = nxts[0].second;
                path.push_back({cur_y, cur_x});

                if (protect_in(cur_y, cur_x)) {
                    blocked_by_protect = true;
                    break;
                }
            }

            if (blocked_by_protect) {
                continue;
            }

            if (static_cast<int>(path.size()) < std::max(1, min_len_px)) {
                for (const auto& p : path) {
                    int yy = p.first;
                    int xx = p.second;
                    if (!protect_in(yy, xx)) {
                        skel(yy, xx) = 0;
                    }
                }
                removed_any = true;
            }
        }

        if (!removed_any) {
            break;
        }
    }

    return out;
}


static py::tuple compute_secondary_angles_cpp(
    py::array_t<int, py::array::c_style | py::array::forcecast> main_path_arr,
    py::list secondary_branches_list
) {
    auto main_buf = main_path_arr.unchecked<2>();

    std::vector<double> abs_angles_deg;
    std::vector<double> angles_deg;

    if (main_buf.ndim() != 2 || main_buf.shape(1) != 2 || main_buf.shape(0) < 2) {
        return py::make_tuple(abs_angles_deg, angles_deg);
    }

    const py::ssize_t main_n = main_buf.shape(0);

    // angle de la racine principale
    double main_vec_y = static_cast<double>(main_buf(main_n - 1, 0) - main_buf(0, 0));
    double main_vec_x = static_cast<double>(main_buf(main_n - 1, 1) - main_buf(0, 1));
    double main_angle = std::atan2(main_vec_y, main_vec_x);

    for (auto item : secondary_branches_list) {
        auto br_arr = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(item);
        if (!br_arr) {
            angles_deg.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }

        auto br_buf = br_arr.unchecked<2>();
        if (br_buf.ndim() != 2 || br_buf.shape(1) != 2 || br_buf.shape(0) < 2) {
            angles_deg.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }

        const py::ssize_t br_n = br_buf.shape(0);

        // 1) base = point de la branche le plus proche de main_path
        py::ssize_t base_idx = 0;
        double best_dist2 = std::numeric_limits<double>::infinity();

        for (py::ssize_t i = 0; i < br_n; ++i) {
            double by = static_cast<double>(br_buf(i, 0));
            double bx = static_cast<double>(br_buf(i, 1));

            for (py::ssize_t j = 0; j < main_n; ++j) {
                double my = static_cast<double>(main_buf(j, 0));
                double mx = static_cast<double>(main_buf(j, 1));
                double dy = by - my;
                double dx = bx - mx;
                double d2 = dy * dy + dx * dx;

                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    base_idx = i;
                }
            }
        }

        double base_y = static_cast<double>(br_buf(base_idx, 0));
        double base_x = static_cast<double>(br_buf(base_idx, 1));

        // 2) extrémité = point le plus éloigné de la base
        py::ssize_t end_idx = 0;
        double best_norm2 = -1.0;

        for (py::ssize_t i = 0; i < br_n; ++i) {
            double dy = static_cast<double>(br_buf(i, 0)) - base_y;
            double dx = static_cast<double>(br_buf(i, 1)) - base_x;
            double norm2 = dy * dy + dx * dx;

            if (norm2 > best_norm2) {
                best_norm2 = norm2;
                end_idx = i;
            }
        }

        double end_y = static_cast<double>(br_buf(end_idx, 0));
        double end_x = static_cast<double>(br_buf(end_idx, 1));

        double vec_y = end_y - base_y;
        double vec_x = end_x - base_x;

        if (std::abs(vec_y) < 1e-12 && std::abs(vec_x) < 1e-12) {
            angles_deg.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }

        double branch_angle = std::atan2(vec_y, vec_x);

        double rel = (branch_angle - main_angle) * 180.0 / PI;
        rel = std::fmod(rel + 180.0, 360.0);
        if (rel < 0.0) rel += 360.0;
        rel -= 180.0;

        angles_deg.push_back(rel);
        abs_angles_deg.push_back(std::abs(rel));
    }

    return py::make_tuple(abs_angles_deg, angles_deg);
}


static inline int clamp_int(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static std::vector<std::vector<std::pair<float, float>>> parse_region_contours(py::list region_contours_list) {
    std::vector<std::vector<std::pair<float, float>>> contours;
    contours.reserve(py::len(region_contours_list));

    for (auto item : region_contours_list) {
        auto arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(item);
        if (!arr) {
            contours.emplace_back();
            continue;
        }

        auto buf = arr.unchecked<2>();
        if (buf.ndim() != 2 || buf.shape(1) != 2) {
            contours.emplace_back();
            continue;
        }

        std::vector<std::pair<float, float>> pts;
        pts.reserve(static_cast<std::size_t>(buf.shape(0)));
        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
            pts.push_back({
                static_cast<float>(buf(i, 0)),
                static_cast<float>(buf(i, 1))
            });
        }
        contours.push_back(std::move(pts));
    }

    return contours;
}

static inline double dist2_pts(const std::pair<float, float>& a, const std::pair<float, float>& b) {
    double dy = static_cast<double>(a.first - b.first);
    double dx = static_cast<double>(a.second - b.second);
    return dy * dy + dx * dx;
}

static void draw_filled_disk_on_mask(
    py::detail::unchecked_mutable_reference<uint8_t, 2>& mask,
    int cy, int cx, int radius,
    int H, int W
) {
    int r2 = radius * radius;
    int y0 = clamp_int(cy - radius, 0, H - 1);
    int y1 = clamp_int(cy + radius, 0, H - 1);
    int x0 = clamp_int(cx - radius, 0, W - 1);
    int x1 = clamp_int(cx + radius, 0, W - 1);

    for (int y = y0; y <= y1; ++y) {
        int dy = y - cy;
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            if (dy * dy + dx * dx <= r2) {
                mask(y, x) = 1;
            }
        }
    }
}

static void draw_thick_line_on_mask(
    py::detail::unchecked_mutable_reference<uint8_t, 2>& mask,
    int y0, int x0, int y1, int x1,
    int radius,
    int H, int W
) {
    int dy = std::abs(y1 - y0);
    int dx = std::abs(x1 - x0);
    int sy = (y0 < y1) ? 1 : -1;
    int sx = (x0 < x1) ? 1 : -1;
    int err = dx - dy;

    int y = y0;
    int x = x0;

    while (true) {
        draw_filled_disk_on_mask(mask, y, x, radius, H, W);

        if (y == y1 && x == x1) {
            break;
        }

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

static py::array_t<uint8_t> connect_regions_cpp(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> current_mask_arr,
    py::list region_contours_list,
    int line_thickness,
    double max_connection_distance
) {
    auto mask_in = current_mask_arr.unchecked<2>();
    if (mask_in.ndim() != 2) {
        throw std::runtime_error("current_mask must be a 2D array");
    }

    const int H = static_cast<int>(mask_in.shape(0));
    const int W = static_cast<int>(mask_in.shape(1));

    py::array_t<uint8_t> out(py::array::ShapeContainer{
        static_cast<py::ssize_t>(H),
        static_cast<py::ssize_t>(W)
    });
    auto mask = out.mutable_unchecked<2>();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            mask(y, x) = mask_in(y, x) ? 1 : 0;
        }
    }

    auto contours = parse_region_contours(region_contours_list);
    const int n = static_cast<int>(contours.size());
    const double max_dist2 = max_connection_distance * max_connection_distance;
    const int radius = std::max(1, line_thickness);

    struct Connection {
        int i;
        int j;
        int y0;
        int x0;
        int y1;
        int x1;
        double dist2;
    };

    std::vector<Connection> connections;
    connections.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        if (contours[i].empty()) continue;

        double best_dist2 = std::numeric_limits<double>::infinity();
        Connection best_conn{};
        bool found = false;

        for (int j = i + 1; j < n; ++j) {
            if (contours[j].empty()) continue;

            for (const auto& pi : contours[i]) {
                for (const auto& pj : contours[j]) {
                    double d2 = dist2_pts(pi, pj);
                    if (d2 < best_dist2) {
                        best_dist2 = d2;
                        best_conn = Connection{
                            i, j,
                            static_cast<int>(std::lround(pi.first)),
                            static_cast<int>(std::lround(pi.second)),
                            static_cast<int>(std::lround(pj.first)),
                            static_cast<int>(std::lround(pj.second)),
                            d2
                        };
                        found = true;
                    }
                }
            }
        }

        if (found && best_dist2 <= max_dist2) {
            connections.push_back(best_conn);
        }
    }

    for (const auto& c : connections) {
        draw_thick_line_on_mask(mask, c.y0, c.x0, c.y1, c.x1, radius, H, W);
    }

    return out;
}


static py::dict analyze_skeleton_pixels_cpp(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> skeleton_arr,
    int grid_rows,
    int grid_cols
) {
    auto sk = skeleton_arr.unchecked<2>();

    if (sk.ndim() != 2) {
        throw std::runtime_error("skeleton must be a 2D array");
    }

    const int H = static_cast<int>(sk.shape(0));
    const int W = static_cast<int>(sk.shape(1));

    py::dict out;

    if (H <= 0 || W <= 0) {
        out["exact_skeleton_length"] = 0.0;
        out["grid_lengths"] = py::none();
        return out;
    }

    const bool use_grid = (grid_rows > 0 && grid_cols > 0);
    constexpr double SQRT2 = 1.4142135623730950488;
    double exact_length = 0.0;

    if (!use_grid) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (!sk(y, x)) {
                    continue;
                }

                if (x + 1 < W && sk(y, x + 1)) {
                    exact_length += 1.0;
                }
                if (y + 1 < H && sk(y + 1, x)) {
                    exact_length += 1.0;
                }
                if (y + 1 < H && x + 1 < W && sk(y + 1, x + 1)) {
                    exact_length += SQRT2;
                }
                if (y + 1 < H && x - 1 >= 0 && sk(y + 1, x - 1)) {
                    exact_length += SQRT2;
                }
            }
        }

        out["exact_skeleton_length"] = exact_length;
        out["grid_lengths"] = py::none();
        return out;
    }

    py::array_t<double> grid_out(py::array::ShapeContainer{
        static_cast<py::ssize_t>(grid_rows),
        static_cast<py::ssize_t>(grid_cols)
    });

    auto grid = grid_out.mutable_unchecked<2>();

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            grid(r, c) = 0.0;
        }
    }

    const double cell_h = static_cast<double>(H) / static_cast<double>(grid_rows);
    const double cell_w = static_cast<double>(W) / static_cast<double>(grid_cols);

    auto accumulate_edge = [&](int y, int x, int dy, int dx, double seg_len) {
        exact_length += seg_len;

        double my = static_cast<double>(y) + 0.5 * static_cast<double>(dy);
        double mx = static_cast<double>(x) + 0.5 * static_cast<double>(dx);

        int rr = static_cast<int>(std::floor(my / cell_h));
        int cc = static_cast<int>(std::floor(mx / cell_w));

        rr = clamp_int(rr, 0, grid_rows - 1);
        cc = clamp_int(cc, 0, grid_cols - 1);

        grid(rr, cc) += seg_len;
    };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (!sk(y, x)) {
                continue;
            }

            if (x + 1 < W && sk(y, x + 1)) {
                accumulate_edge(y, x, 0, 1, 1.0);
            }
            if (y + 1 < H && sk(y + 1, x)) {
                accumulate_edge(y, x, 1, 0, 1.0);
            }
            if (y + 1 < H && x + 1 < W && sk(y + 1, x + 1)) {
                accumulate_edge(y, x, 1, 1, SQRT2);
            }
            if (y + 1 < H && x - 1 >= 0 && sk(y + 1, x - 1)) {
                accumulate_edge(y, x, 1, -1, SQRT2);
            }
        }
    }

    out["exact_skeleton_length"] = exact_length;
    out["grid_lengths"] = grid_out;
    return out;
}



PYBIND11_MODULE(root_graph_cpp, m) {
    m.doc() = "C++ graph helpers for root skeleton analysis";

    m.def(
        "analyze_skeleton_graph",
        &analyze_skeleton_graph,
        py::arg("skeleton_points"),
        py::arg("main_ref_path") = py::none(),
        py::arg("main_path_bias") = 20.0,
        "Analyze graph built from skeleton points"
    );

    m.def(
        "prune_terminal_spurs",
        &prune_terminal_spurs_cpp,
        py::arg("skel"),
        py::arg("min_len_px"),
        py::arg("protect_mask"),
        py::arg("max_iter") = 50,
        "Remove terminal branches shorter than min_len_px from a skeleton"
    );

    m.def(
        "compute_secondary_angles",
        &compute_secondary_angles_cpp,
        py::arg("main_path"),
        py::arg("secondary_branches"),
        "Compute relative and absolute angles for secondary branches"
    );

    m.def(
        "connect_regions",
        &connect_regions_cpp,
        py::arg("current_mask"),
        py::arg("region_contours"),
        py::arg("line_thickness"),
        py::arg("max_connection_distance"),
        "Connect sampled region contours and draw thick lines on the mask"
    );

    m.def(
        "analyze_skeleton_pixels",
        &analyze_skeleton_pixels_cpp,
        py::arg("skeleton"),
        py::arg("grid_rows"),
        py::arg("grid_cols"),
        "Compute exact skeleton length and grid lengths in one pass"
    );
}