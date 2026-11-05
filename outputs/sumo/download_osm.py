#!/usr/bin/env python3
"""Download the OSM road network for the Q-BS study area in small tiles (avoids Overpass 504 timeouts) and merge it.
Output: ktm_bbox.osm.xml   (same name the SUMO pipeline expects).  Needs only the Python standard library."""
import sys, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

W, S, E, N = 85.2439, 27.6522, 85.4592, 27.7696      # Q-BS bounding box + 0.01 deg margin
NX, NY = 4, 3                                         # tiles (increase if a tile still times out)
MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.openstreetmap.fr/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
SKIP = "footway|path|steps|pedestrian|cycleway|corridor|bridleway|elevator|proposed|construction|track|raceway|bus_guideway"

def query(s, w, n, e):
    return (f'[out:xml][timeout:180];(way["highway"]["highway"!~"{SKIP}"]({s},{w},{n},{e}););'
            f'(._;>;);out body qt;')

def fetch(q, tries=6):
    data = urllib.parse.urlencode({"data": q}).encode()
    for k in range(tries):
        url = MIRRORS[k % len(MIRRORS)]
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "qbts-sumo-study/1.0 (research)"})
            with urllib.request.urlopen(req, timeout=240) as r:
                return r.read()
        except Exception as ex:
            print(f"   {url.split('/')[2]}: {ex} -> retry", flush=True); time.sleep(5 * (k + 1))
    raise SystemExit("All mirrors failed for a tile. Increase NX/NY, try later, or use the Geofabrik fallback (README).")

def main(out="ktm_bbox.osm.xml", fetcher=fetch):
    nodes, ways = {}, {}
    dw, ds = (E - W) / NX, (N - S) / NY
    for i in range(NX):
        for j in range(NY):
            w, e = W + i * dw, W + (i + 1) * dw; s, n = S + j * ds, S + (j + 1) * ds
            print(f"tile {i * NY + j + 1}/{NX * NY}: {s:.4f},{w:.4f},{n:.4f},{e:.4f}", flush=True)
            root = ET.fromstring(fetcher(query(round(s, 5), round(w, 5), round(n, 5), round(e, 5))))
            for el in root:
                if el.tag == "node": nodes[el.get("id")] = el
                elif el.tag == "way": ways[el.get("id")] = el
    osm = ET.Element("osm", version="0.6", generator="download_osm.py")
    ET.SubElement(osm, "bounds", minlat=str(S), minlon=str(W), maxlat=str(N), maxlon=str(E))
    for el in nodes.values(): osm.append(el)
    for el in ways.values(): osm.append(el)
    ET.ElementTree(osm).write(out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {out}: {len(nodes)} nodes, {len(ways)} ways")

if __name__ == "__main__":
    main()
