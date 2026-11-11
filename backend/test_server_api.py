import json
from fastapi.testclient import TestClient
from main import app, startup_event

# Trigger startup event to preload model and grids
startup_event()

client = TestClient(app)

response = client.get("/risk-grid?region=south_bengal&limit=3")
print(f"Status Code: {response.status_code}")
data = response.json()
print("Header fields:", {k: data[k] for k in ["region", "timezone", "total_cells", "returned_cells", "scoring_parameters"]})
print("\nSample Cell Output (First 2 cells):")
for cell in data.get("cells", [])[:2]:
    print(json.dumps(cell, indent=2))

