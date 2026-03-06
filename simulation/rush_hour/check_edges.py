import xml.etree.ElementTree as ET
from collections import Counter

def analyze_routes():
    """
    Analyze which edges are used vs unused
    in the existing route file
    """

    route_file = "simulation/maps/tahrir.rou.xml"
    net_file   = "simulation/maps/tahrirupdated.net.xml"

    # ─── Count edge usage in routes ───────────────────
    tree     = ET.parse(route_file)
    root     = tree.getroot()
    vehicles = root.findall("vehicle")

    edge_counter = Counter()
    all_routes   = []

    for v in vehicles:
        route_elem = v.find("route")
        if route_elem is not None:
            edges = route_elem.get("edges", "")
            edge_list = edges.split()
            all_routes.append(edge_list)
            for edge in edge_list:
                edge_counter[edge] += 1

    # ─── Get all edges from network ───────────────────
    net_tree  = ET.parse(net_file)
    net_root  = net_tree.getroot()
    all_edges = set()

    for edge in net_root.findall("edge"):
        edge_id = edge.get("id", "")
        # Skip internal edges (start with :)
        if not edge_id.startswith(":"):
            all_edges.add(edge_id)

    # ─── Find unused edges ────────────────────────────
    used_edges   = set(edge_counter.keys())
    unused_edges = all_edges - used_edges

    # ─── Print Report ─────────────────────────────────

    print("  EDGE USAGE ANALYSIS")
    print(f"\nTotal edges in network:  {len(all_edges)}")
    print(f"Edges used in routes:    {len(used_edges)}")
    print(f"Edges NEVER used:        {len(unused_edges)}")
    print(f"Usage rate:              "
          f"{len(used_edges)/len(all_edges)*100:.1f}%")

    print(f"\nRoute statistics:")
    print(f"Total vehicles:          {len(vehicles)}")
    print(f"Unique routes:           {len(set([str(r) for r in all_routes]))}")

    # Route length distribution
    lengths = [len(r) for r in all_routes]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"Avg route length:        {avg_len:.1f} edges")
    print(f"Shortest route:          {min(lengths)} edges")
    print(f"Longest route:           {max(lengths)} edges")

    print(f"\nTop 10 most used edges:")
    for edge, count in edge_counter.most_common(10):
        print(f"  {edge:40s} → {count:3d} vehicles")

    print(f"\nSample unused edges (first 10):")
    for edge in list(unused_edges)[:10]:
        print(f"  {edge}")

    # ─── Short Route Warning ──────────────────────────
    short_routes = [r for r in all_routes if len(r) <= 2]
    print(f"\n Short routes (≤2 edges): "
          f"{len(short_routes)} vehicles")
    print(f"   These cause edges to look empty")
    print(f"   Vehicles enter and exit too fast")


def fix_short_routes():
    """
    Find vehicles with short routes and
    extend them using connected edges
    """
    route_file = "simulation/maps/tahrir.rou.xml"
    net_file   = "simulation/maps/tahrirupdated.net.xml"

    # Build edge connections from network
    net_tree  = ET.parse(net_file)
    net_root  = net_tree.getroot()

    # Map: edge_id → list of edges it connects to
    connections = {}
    for conn in net_root.findall("connection"):
        from_edge = conn.get("from", "")
        to_edge   = conn.get("to", "")
        if from_edge and to_edge:
            if from_edge not in connections:
                connections[from_edge] = []
            if to_edge not in connections[from_edge]:
                connections[from_edge].append(to_edge)

    # Read routes
    tree     = ET.parse(route_file)
    root     = tree.getroot()
    vehicles = root.findall("vehicle")

    short_count = 0
    fixed_count = 0

    for v in vehicles:
        route_elem = v.find("route")
        if route_elem is None:
            continue

        edges     = route_elem.get("edges", "").split()

        if len(edges) <= 2:
            short_count += 1
            # Try to extend route
            last_edge = edges[-1]
            extended  = edges.copy()

            # Add up to 3 more connected edges
            for _ in range(3):
                next_edges = connections.get(last_edge, [])
                if next_edges:
                    next_edge = next_edges[0]
                    extended.append(next_edge)
                    last_edge = next_edge
                else:
                    break

            if len(extended) > len(edges):
                route_elem.set("edges", " ".join(extended))
                fixed_count += 1

    # Save fixed routes
    output = "simulation/maps/tahrir_fixed.rou.xml"
    tree.write(output, encoding="unicode", xml_declaration=True)

    print(f"\nShort routes found:  {short_count}")
    print(f"Routes extended:     {fixed_count}")
    print(f"Fixed file saved:    {output} ")

if __name__ == "__main__":
    analyze_routes()
    fix_short_routes()