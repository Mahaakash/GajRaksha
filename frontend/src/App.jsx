import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';

// Ensure global L is assigned for Leaflet.heat
window.L = L;

const REGION_CONFIGS = {
  south_bengal: {
    name: 'South Bengal',
    center: [22.8, 87.0],
    zoom: 9,
    bounds: [[21.7, 85.8], [23.9, 88.0]]
  },
  kruger: {
    name: 'Kruger',
    center: [-24.5, 31.4],
    zoom: 9,
    bounds: [[-25.6, 30.5], [-22.5, 32.2]]
  }
};

const getRecommendedAction = (riskLevel, topReasons = []) => {
  const reasonsText = (topReasons || []).join(' ').toLowerCase();

  if (riskLevel === 'high') {
    if (reasonsText.includes('water') || reasonsText.includes('river')) {
      return 'Increase night patrols along river drainage corridors, deploy spotlights, and activate acoustic early warnings.';
    } else if (reasonsText.includes('crop') || reasonsText.includes('agricultural')) {
      return 'Increase night patrols across peripheral farmlands, mobilize community watch squads, and broadcast SMS alerts.';
    } else {
      return 'Increase night patrols along forest boundary trails, maintain perimeter illumination, and alert response teams.';
    }
  } else if (riskLevel === 'medium') {
    if (reasonsText.includes('crop') || reasonsText.includes('agricultural')) {
      return 'Increase evening patrols near crop fields, verify solar fence voltage, and advise farmers to secure grain stores.';
    } else if (reasonsText.includes('forest')) {
      return 'Increase evening patrols along woodland edge, coordinate village vigil squads, and avoid unlit trail crossings after dusk.';
    } else {
      return 'Increase evening patrols, inspect community acoustic alarms, and maintain radio contact with local range officers.';
    }
  } else {
    return 'Maintain routine village monitoring; elephant presence risk is currently low along this perimeter.';
  }
};

export default function App() {
  const searchParams = new URLSearchParams(window.location.search);
  const initialHour = searchParams.has('hour') ? Math.min(23, Math.max(0, parseInt(searchParams.get('hour'), 10))) : 2;
  const initialRain = searchParams.has('rainfall') ? Math.max(0, parseFloat(searchParams.get('rainfall'))) : 5.0;
  const initialSelected = searchParams.get('selected');
  const initialTracks = searchParams.get('tracks') === 'true';
  const initialModal = searchParams.get('modal') === 'true';

  const [region, setRegion] = useState(initialTracks ? 'kruger' : 'south_bengal');
  const [hour, setHour] = useState(initialHour);
  const [rainfall, setRainfall] = useState(initialRain);
  const [debouncedHour, setDebouncedHour] = useState(initialHour);
  const [debouncedRainfall, setDebouncedRainfall] = useState(initialRain);
  const [isPlaying, setIsPlaying] = useState(false);

  const [loading, setLoading] = useState(true);
  const [gridCellCount, setGridCellCount] = useState(0);
  const [villageList, setVillageList] = useState([]);
  const [selectedVillage, setSelectedVillage] = useState(null);

  // New: Transparency Modal and Elephant Track Layer states
  const [isAboutOpen, setIsAboutOpen] = useState(initialModal);
  const [showTracks, setShowTracks] = useState(initialTracks);
  const [tracksData, setTracksData] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const tileLayerRef = useRef(null);
  const heatLayerRef = useRef(null);
  const markersLayerRef = useRef(null);
  const tracksLayerRef = useRef(null);
  const markersMapRef = useRef({});
  const debounceTimerRef = useRef(null);

  const isNight = hour >= 18 || hour < 6;

  // Load Model Info once on mount for transparency modal
  useEffect(() => {
    async function fetchModelInfo() {
      try {
        const res = await fetch('/model-info');
        const data = await res.json();
        setModelInfo(data);
      } catch (err) {
        console.error('Failed to load model info:', err);
      }
    }
    fetchModelInfo();
  }, []);

  // Debounce slider inputs (wait 350ms after sliding stops)
  useEffect(() => {
    if (isPlaying) {
      setDebouncedHour(hour);
      setDebouncedRainfall(rainfall);
      return;
    }

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      setDebouncedHour(hour);
      setDebouncedRainfall(rainfall);
    }, 350);

    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, [hour, rainfall, isPlaying]);

  // Handle "Play 24h" animation cycle
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setHour((prev) => (prev + 1) % 24);
      }, 700);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying]);

  // Initialize Map Once
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const initialConfig = REGION_CONFIGS[region];
    const map = L.map(mapRef.current, {
      center: initialConfig.center,
      zoom: initialConfig.zoom,
      zoomControl: true,
      preferCanvas: true
    });

    // Clean Dark basemap using OpenStreetMap with CSS dark filter
    const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      className: 'dark-tiles night-mode',
      maxZoom: 19
    }).addTo(map);

    tileLayerRef.current = tiles;
    markersLayerRef.current = L.layerGroup().addTo(map);
    tracksLayerRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;

    setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Tile Mode (Day / Night ambient shift)
  useEffect(() => {
    const tileContainer = document.querySelector('.dark-tiles');
    if (tileContainer) {
      if (isNight) {
        tileContainer.classList.add('night-mode');
        tileContainer.classList.remove('day-mode');
      } else {
        tileContainer.classList.add('day-mode');
        tileContainer.classList.remove('night-mode');
      }
    }
  }, [isNight]);

  // Handle Real Elephant Tracks Layer (Kruger)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !tracksLayerRef.current) return;

    if (!showTracks) {
      tracksLayerRef.current.clearLayers();
      return;
    }

    // Switch region to Kruger if not already
    if (region !== 'kruger') {
      setRegion('kruger');
    }
    const kConfig = REGION_CONFIGS.kruger;
    map.flyTo([-24.75, 31.6], 10, { duration: 1.0 });

    // Ensure dedicated tracks pane exists above overlay pane
    if (!map.getPane('tracksPane')) {
      const tp = map.createPane('tracksPane');
      tp.style.zIndex = '650';
    }

    async function loadTracks() {
      try {
        const res = await fetch('/tracks-sample?limit_per_animal=200');
        const data = await res.json();
        setTracksData(data);

        tracksLayerRef.current.clearLayers();

        if (data.animals) {
          Object.values(data.animals).forEach((animal) => {
            const latlngs = animal.points.map((p) => [p.lat, p.lon]);

            // Draw elephant tracking path polyline
            const pathLine = L.polyline(latlngs, {
              color: animal.color,
              weight: 4.0,
              opacity: 0.95,
              lineCap: 'round',
              lineJoin: 'round',
              pane: 'tracksPane'
            });

            pathLine.bindPopup(`
              <div class="popup-card">
                <div class="popup-title">Elephant Collar: ${animal.animal_id}</div>
                <div class="popup-subtitle">Movebank Study 736029750 (Kruger National Park)</div>
                <div style="font-size: 11px; margin-top: 6px;">Total fixes in study: <strong>${animal.total_fixes.toLocaleString()}</strong></div>
                <div style="font-size: 11px;">Sampled coordinates: <strong>${animal.sampled_fixes}</strong></div>
              </div>
            `);

            tracksLayerRef.current.addLayer(pathLine);

            // Start & End markers
            if (animal.points.length > 0) {
              const startPt = animal.points[0];
              const endPt = animal.points[animal.points.length - 1];

              const startMarker = L.circleMarker([startPt.lat, startPt.lon], {
                radius: 6,
                fillColor: animal.color,
                color: '#ffffff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.95,
                pane: 'tracksPane'
              }).bindPopup(`<b>${animal.animal_id}</b> Start Fix: ${startPt.timestamp}`);

              const endMarker = L.circleMarker([endPt.lat, endPt.lon], {
                radius: 6,
                fillColor: '#ffffff',
                color: animal.color,
                weight: 3,
                opacity: 1,
                fillOpacity: 0.95,
                pane: 'tracksPane'
              }).bindPopup(`<b>${animal.animal_id}</b> Latest Fix: ${endPt.timestamp}`);

              tracksLayerRef.current.addLayer(startMarker);
              tracksLayerRef.current.addLayer(endMarker);
            }
          });
        }
      } catch (err) {
        console.error('Failed to load tracks sample:', err);
      }
    }

    loadTracks();
  }, [showTracks, region]);

  // Fetch Risk Grid and Villages whenever region or debounced values change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    let isMounted = true;
    setLoading(true);

    const config = REGION_CONFIGS[region];
    const currentCenter = map.getCenter();
    const targetCenter = L.latLng(config.center[0], config.center[1]);
    if (currentCenter.distanceTo(targetCenter) > 200000) {
      map.setView(config.center, config.zoom);
    }

    async function loadData() {
      try {
        if (!L.heatLayer && !window.L.heatLayer) {
          await import('leaflet.heat');
        }

        const heatFactory = L.heatLayer || window.L.heatLayer;

        // 1. Fetch Risk Grid Points
        const gridUrl = `/risk-grid?region=${region}&hour=${debouncedHour}&rainfall=${debouncedRainfall}&limit=20000`;
        const gridRes = await fetch(gridUrl);
        const gridData = await gridRes.json();

        // 2. Fetch Villages
        const villageUrl = `/village-risk?region=${region}&hour=${debouncedHour}&rainfall=${debouncedRainfall}`;
        const villageRes = await fetch(villageUrl);
        const villageData = await villageRes.json();

        if (!isMounted) return;

        setGridCellCount(gridData.returned_cells || 0);
        const rawVillages = villageData.villages || [];

        const sorted = [...rawVillages].sort(
          (a, b) => (b.elephant_presence_risk ?? 0) - (a.elephant_presence_risk ?? 0)
        );
        setVillageList(sorted);

        // Auto-select initial village
        if (sorted.length > 0) {
          if (initialSelected) {
            const found = sorted.find((v) => v.village.toLowerCase() === initialSelected.toLowerCase());
            setSelectedVillage(found || sorted[0]);
          } else {
            setSelectedVillage((prev) => {
              if (prev) {
                const refreshed = sorted.find((v) => v.village === prev.village);
                return refreshed || sorted[0];
              }
              return sorted[0];
            });
          }
        }

        // Draw Heatmap
        if (heatLayerRef.current) {
          map.removeLayer(heatLayerRef.current);
          heatLayerRef.current = null;
        }

        const currentIsNight = debouncedHour >= 18 || debouncedHour < 6;
        const diurnalMultiplier = currentIsNight ? 1.30 : 0.65;
        const heatRadius = currentIsNight ? 22 : 16;
        const heatBlur = currentIsNight ? 18 : 12;
        const heatMax = currentIsNight ? 0.85 : 1.15;

        if (gridData.points && gridData.points.length > 0 && heatFactory) {
          const heatPoints = gridData.points.map((p) => {
            const intensity = Math.min(1.0, p.risk * diurnalMultiplier);
            return [p.lat, p.lon, intensity];
          });

          const heat = heatFactory(heatPoints, {
            radius: heatRadius,
            blur: heatBlur,
            maxZoom: 13,
            max: heatMax,
            minOpacity: currentIsNight ? 0.20 : 0.10,
            gradient: {
              0.1: '#00ff88',
              0.3: '#a3e635',
              0.5: '#facc15',
              0.7: '#fb923c',
              0.85: '#ef4444'
            }
          });

          heat.addTo(map);
          heatLayerRef.current = heat;
        }

        // Clear and draw Village Markers
        if (markersLayerRef.current) {
          markersLayerRef.current.clearLayers();
          markersMapRef.current = {};

          sorted.forEach((vil) => {
            const riskLevel = vil.risk_level || 'low';
            const riskScore = vil.elephant_presence_risk ?? 0;
            const reasons = vil.top_reasons || [];
            const action = getRecommendedAction(riskLevel, reasons);

            const customIcon = L.divIcon({
              className: 'custom-village-icon-wrapper',
              iconSize: [28, 48],
              iconAnchor: [14, 24],
              html: `
                <div class="village-custom-marker">
                  <div class="marker-ring ${riskLevel}">
                    <div class="marker-center-dot"></div>
                  </div>
                  <div class="marker-name-tag">${vil.village}</div>
                </div>
              `
            });

            const marker = L.marker([vil.lat, vil.lon], { icon: customIcon });

            const reasonsHtml = reasons.map((r) => `<li>🔹 ${r}</li>`).join('');
            const popupContent = `
              <div class="popup-card">
                <div class="popup-header">
                  <div class="popup-title">${vil.village}</div>
                  <div class="popup-subtitle">${region === 'south_bengal' ? 'South Bengal, India' : 'Kruger Interface, South Africa'}</div>
                  <span class="popup-risk-badge ${riskLevel}">${riskLevel} Risk</span>
                </div>
                <div class="popup-score-row">
                  <span>Elephant Presence Risk:</span>
                  <strong>${(riskScore * 100).toFixed(1)}%</strong>
                </div>
                <div class="popup-reasons-title">Top Contributing Factors (SHAP)</div>
                <ul class="popup-reasons-list">
                  ${reasonsHtml}
                </ul>
                <div class="action-card" style="margin-top: 8px;">
                  <div class="action-card-header">🛡️ Recommended Action</div>
                  <div class="action-card-text">${action}</div>
                </div>
              </div>
            `;

            marker.bindPopup(popupContent, { maxWidth: 320 });
            marker.on('click', () => {
              handleSelectVillage(vil);
            });

            markersLayerRef.current.addLayer(marker);
            markersMapRef.current[vil.village] = marker;
          });
        }
      } catch (err) {
        console.error('Failed to load risk map data:', err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadData();

    return () => {
      isMounted = false;
    };
  }, [region, debouncedHour, debouncedRainfall]);

  const handleSelectVillage = (vil) => {
    setSelectedVillage(vil);
    const map = mapInstanceRef.current;
    if (map) {
      map.flyTo([vil.lat, vil.lon], 12, { duration: 1.0 });
      const marker = markersMapRef.current[vil.village];
      if (marker) {
        setTimeout(() => {
          marker.openPopup();
        }, 500);
      }
    }
  };

  const formatHour = (h) => {
    const period = h >= 12 ? 'PM' : 'AM';
    const displayH = h % 12 === 0 ? 12 : h % 12;
    return `${String(h).padStart(2, '0')}:00 (${displayH}:00 ${period})`;
  };

  const sortedVillages = [...villageList].sort(
    (a, b) => (b.elephant_presence_risk ?? 0) - (a.elephant_presence_risk ?? 0)
  );

  return (
    <div className="app-container">
      {/* Header Bar */}
      <header className="header-bar">
        <div className="brand-section">
          <div className="brand-icon">🐘</div>
          <div>
            <h1 className="brand-title">GajRaksha</h1>
            <div className="brand-subtitle">Elephant Presence Risk Early Warning System</div>
          </div>
        </div>

        <div className="header-controls">
          {/* About this data button */}
          <button className="about-btn" onClick={() => setIsAboutOpen(true)}>
            ℹ️ About this data
          </button>

          <div className="status-badge">
            <span className="status-dot"></span>
            <span>
              {gridCellCount > 0 ? `${gridCellCount.toLocaleString()} Grid Cells Scored` : 'Loading Grids...'}
            </span>
          </div>

          <div className="status-badge">
            <span>Hour: {String(hour).padStart(2, '0')}:00 {isNight ? '(Night)' : '(Day)'}</span>
          </div>

          <div className="region-selector">
            <label htmlFor="region-select" className="region-label">Region:</label>
            <select
              id="region-select"
              className="region-dropdown"
              value={region}
              onChange={(e) => {
                const newRegion = e.target.value;
                setRegion(newRegion);
                setSelectedVillage(null);
                if (newRegion !== 'kruger') {
                  setShowTracks(false);
                }
                const config = REGION_CONFIGS[newRegion];
                if (mapInstanceRef.current && config) {
                  mapInstanceRef.current.setView(config.center, config.zoom);
                }
              }}
            >
              <option value="south_bengal">South Bengal</option>
              <option value="kruger">Kruger</option>
            </select>
          </div>
        </div>
      </header>

      {/* Left Control Panel */}
      <aside className="control-panel">
        <div className="panel-header">
          <h2 className="panel-title">
            <span>⚙️</span> Interactive Controls
          </h2>
          <p className="panel-subtitle">Simulate diurnal elephant presence risk shifts</p>
        </div>

        {/* Real Elephant Tracks Toggle (Kruger) */}
        <div className="track-toggle-container">
          <div className="track-toggle-label">
            <span>📍</span> Show real elephant tracks (Kruger)
          </div>
          <label className="switch">
            <input
              type="checkbox"
              checked={showTracks}
              onChange={(e) => setShowTracks(e.target.checked)}
            />
            <span className="slider-switch"></span>
          </label>
        </div>

        {/* Hour of Day Slider */}
        <div className="control-group">
          <div className="control-header-row">
            <label htmlFor="hour-slider" className="control-label">Hour of Day</label>
            <span className={`time-badge ${isNight ? 'night' : 'day'}`}>
              {isNight ? '🌙' : '☀️'} {formatHour(hour)}
            </span>
          </div>
          <input
            id="hour-slider"
            type="range"
            min="0"
            max="23"
            step="1"
            value={hour}
            onChange={(e) => setHour(Number(e.target.value))}
            className="slider-input"
          />
          <div className="slider-ticks">
            <span>00:00 (Midnight)</span>
            <span>12:00 (Noon)</span>
            <span>23:00</span>
          </div>
        </div>

        {/* Play 24h Cycle Button */}
        <button
          className={`play-button ${isPlaying ? 'active' : ''}`}
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? '⏸ Pause 24h Cycle' : '▶ Play 24h Cycle'}
        </button>

        {/* Rainfall Slider */}
        <div className="control-group">
          <div className="control-header-row">
            <label htmlFor="rain-slider" className="control-label">Recent Rainfall (7-Day)</label>
            <span className="value-badge">{rainfall.toFixed(1)} mm</span>
          </div>
          <input
            id="rain-slider"
            type="range"
            min="0"
            max="50"
            step="1"
            value={rainfall}
            onChange={(e) => setRainfall(Number(e.target.value))}
            className="slider-input"
          />
          <div className="slider-ticks">
            <span>0 mm (Dry)</span>
            <span>25 mm</span>
            <span>50 mm (Monsoon)</span>
          </div>
        </div>

        {/* Temporal Window Information */}
        <div className="panel-metrics-grid">
          <div className="metric-box">
            <span className="metric-label">Temporal Window</span>
            <span className="metric-value" style={{ color: isNight ? '#a5b4fc' : '#fde047' }}>
              {isNight ? 'Active Nocturnal' : 'Diurnal Resting'}
            </span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Monitored Villages</span>
            <span className="metric-value">
              {sortedVillages.length} Settlements
            </span>
          </div>
        </div>
      </aside>

      {/* Kruger Tracks Legend Card (Appears when tracks are enabled) */}
      {showTracks && tracksData && (
        <div className="track-legend-card">
          <div className="track-legend-title">
            <span>📡</span> Training Tracks (Movebank)
          </div>
          <div className="track-legend-list">
            {Object.values(tracksData.animals || {}).map((a) => (
              <div key={a.animal_id} className="track-legend-row">
                <div>
                  <span className="track-color-pill" style={{ backgroundColor: a.color }}></span>
                  <strong>{a.animal_id}</strong>
                </div>
                <span>{a.total_fixes.toLocaleString()} fixes</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Right Panel: Ranked Villages & Open Detail Card */}
      <aside className="villages-panel">
        <div className="villages-panel-header">
          <h2 className="villages-panel-title">
            <span>📍</span> Villages at Risk
          </h2>
          <span className="villages-count-badge">
            {sortedVillages.length} Villages
          </span>
        </div>

        {/* Selected Village Open Detail Card */}
        {selectedVillage && (
          <div className="detail-card">
            <div className="detail-card-top">
              <div>
                <h3 className="detail-village-name">{selectedVillage.village}</h3>
                <div className="detail-district-name">
                  {region === 'south_bengal' ? 'South Bengal Interface' : 'Kruger Border Interface'}
                </div>
              </div>
              <span className={`village-badge ${selectedVillage.risk_level || 'low'}`}>
                {(selectedVillage.risk_level || 'low').toUpperCase()}
              </span>
            </div>

            <div className="detail-score-banner">
              <span className="detail-score-label">Elephant Presence Risk</span>
              <span
                className="detail-score-val"
                style={{
                  color:
                    selectedVillage.risk_level === 'high'
                      ? '#f87171'
                      : selectedVillage.risk_level === 'medium'
                      ? '#fbbf24'
                      : '#34d399'
                }}
              >
                {((selectedVillage.elephant_presence_risk ?? 0) * 100).toFixed(1)}%
              </span>
            </div>

            <div className="detail-confidence-row">
              <span>Model Confidence:</span>
              <span className="detail-confidence-badge">
                {selectedVillage.confidence === 'high' ? 'High' : 'Moderate (Extrapolated)'}
              </span>
            </div>

            <div className="detail-reasons-title">Top 3 Contributing Factors (SHAP)</div>
            <ul className="detail-reasons-list">
              {(selectedVillage.top_reasons || []).map((reason, idx) => (
                <li key={idx}>🔹 {reason}</li>
              ))}
            </ul>

            {/* Recommended Action */}
            <div className="action-card">
              <div className="action-card-header">🛡️ Recommended Action</div>
              <div className="action-card-text">
                {getRecommendedAction(selectedVillage.risk_level, selectedVillage.top_reasons)}
              </div>
            </div>
          </div>
        )}

        {/* Ranked Villages List */}
        <div className="ranked-villages-section">
          <div className="ranked-section-label">
            Ranked by Elephant Presence Risk ({sortedVillages.length})
          </div>
          <div className="ranked-list-scroll">
            {sortedVillages.map((vil, index) => {
              const isSelected = selectedVillage && selectedVillage.village === vil.village;
              const riskPct = ((vil.elephant_presence_risk ?? 0) * 100).toFixed(1);
              return (
                <button
                  key={vil.village}
                  className={`village-item-btn ${isSelected ? 'active' : ''}`}
                  onClick={() => handleSelectVillage(vil)}
                >
                  <div className="village-item-left">
                    <span className="village-rank-num">#{index + 1}</span>
                    <div>
                      <div className="village-name-primary">{vil.village}</div>
                      <div className="village-district-secondary">
                        Risk: {riskPct}%
                      </div>
                    </div>
                  </div>
                  <span className={`village-badge ${vil.risk_level || 'low'}`}>
                    {riskPct}%
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </aside>

      {/* Loading Banner */}
      {loading && (
        <div className="loading-indicator">
          Updating elephant presence risk heatmap and village telemetry...
        </div>
      )}

      {/* Full-Screen Leaflet Map */}
      <div id="map" ref={mapRef}></div>

      {/* Legend Card */}
      <div className="legend-card">
        <div className="legend-title">Elephant Presence Risk</div>

        {/* Heatmap Section */}
        <div className="legend-section">
          <div className="legend-section-title">Risk Heatmap (Grid Density)</div>
          <div className="gradient-bar"></div>
          <div className="gradient-labels">
            <span>Low (Green)</span>
            <span>Medium</span>
            <span>High (Red)</span>
          </div>
        </div>

        {/* Village Markers Section */}
        <div className="legend-section">
          <div className="legend-section-title">Village Interface Markers</div>
          <div className="village-legend-list">
            <div className="village-legend-item">
              <div className="legend-ring high"></div>
              <span>High Risk Village (≥ 65%)</span>
            </div>
            <div className="village-legend-item">
              <div className="legend-ring medium"></div>
              <span>Medium Risk Village (30% - 65%)</span>
            </div>
            <div className="village-legend-item">
              <div className="legend-ring low"></div>
              <span>Low Risk Village (&lt; 30%)</span>
            </div>
          </div>
        </div>

        <div className="legend-subtext">
          Coloured rings reflect predicted elephant presence risk at the 500m community perimeter.
        </div>
      </div>

      {/* "About this data" Transparency Modal */}
      {isAboutOpen && (
        <div className="modal-backdrop" onClick={() => setIsAboutOpen(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-wrap">
                <span className="modal-icon">ℹ️</span>
                <div>
                  <h2 className="modal-title">About This Data & Methodology</h2>
                  <div className="modal-subtitle">GajRaksha Elephant Presence Risk Early Warning Architecture</div>
                </div>
              </div>
              <button className="modal-close-btn" onClick={() => setIsAboutOpen(false)}>✕</button>
            </div>

            <div className="modal-body">
              {/* Critical Transparency Statement */}
              <div className="disclaimer-banner">
                <span className="disclaimer-icon">⚠️</span>
                <div>
                  <div className="disclaimer-title">Critical Data & Prediction Disclaimer</div>
                  <div className="disclaimer-text">
                    The model has not yet been trained or validated on Indian elephant collar data and predicts elephant presence near villages, not confirmed incidents.
                  </div>
                </div>
              </div>

              <div className="modal-content-split">
                <div className="modal-left-col" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {/* 1. Training Telemetry Data */}
                  <div className="modal-section">
                    <h3 className="modal-section-title">📡 1. Training GPS Collar Telemetry (Kruger Region)</h3>
                    <table className="info-table">
                      <tbody>
                        <tr>
                          <th>Movebank Study</th>
                          <td>Elephant Population (Wall et al. 2013 / Slotow et al. 2005 - Kruger National Park, South Africa)</td>
                        </tr>
                        <tr>
                          <th>Movebank Study ID</th>
                          <td><code>736029750</code></td>
                        </tr>
                        <tr>
                          <th>Citation</th>
                          <td>Wall, J., Wittemyer, G., LeMay, V., Douglas-Hamilton, I., Slotow, R. (2013). Tracking African Elephants: Movebank Study 736029750.</td>
                        </tr>
                        <tr>
                          <th>License</th>
                          <td>Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)</td>
                        </tr>
                        <tr>
                          <th>Animals Tracked</th>
                          <td>4 African Elephants (<code>AM105</code>, <code>AM107</code>, <code>AM108</code>, <code>AM110</code>)</td>
                        </tr>
                        <tr>
                          <th>Tracking Period</th>
                          <td>2007-08-13 to 2009-08-12 (24 months, 34,483 hourly tracking fixes)</td>
                        </tr>
                        <tr>
                          <th>Training Framework</th>
                          <td>1:5 "Used vs. Available" resource selection design (34,483 positive GPS fixes vs. 172,415 random background points)</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* 2. Target Transfer */}
                  <div className="modal-section">
                    <h3 className="modal-section-title">🗺️ 2. Landscape Transfer (South Bengal, India)</h3>
                    <table className="info-table">
                      <tbody>
                        <tr>
                          <th>Target Districts</th>
                          <td>Jhargram, Paschim Medinipur, Bankura, Purulia (22,570.9 km² total target area)</td>
                        </tr>
                        <tr>
                          <th>Landscape Features</th>
                          <td>OpenStreetMap (OSM) settlements, water features, waterways; ESA WorldCover 10m tree cover & cropland</td>
                        </tr>
                        <tr>
                          <th>Resolution</th>
                          <td>~1 km regular spatial grid (19,820 evaluated cells inside target district boundary polygons)</td>
                        </tr>
                        <tr>
                          <th>WII Report Check</th>
                          <td>
                            Checked against published conflict hotspots from: <em>"Human-elephant conflict in the State of West Bengal, India (2000-2025)"</em> (Mathur et al., Wildlife Institute of India & Project Elephant TR-0039-2026).
                            <br />
                            <strong>Sanity-Check Result:</strong> Top 10% highest-risk cells in South Bengal capture published conflict hotspots with <strong>2.45x enrichment</strong> over random chance expectation (5 km radius).
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* 3. Model Architecture & Metrics */}
                  <div className="modal-section">
                    <h3 className="modal-section-title">📊 3. Model Performance & Validation</h3>
                    <div className="metrics-cards-grid">
                      <div className="metric-card">
                        <div className="metric-card-header">GroupKFold Cross-Validation (by Animal)</div>
                        <div className="metric-stat-row">
                          <span>Evaluation Protocol:</span>
                          <span>Leaves out completely unseen individual animals</span>
                        </div>
                        <div className="metric-stat-row">
                          <span>Mean ROC-AUC:</span>
                          <strong>0.4803</strong>
                        </div>
                        <div className="metric-stat-row">
                          <span>Std ROC-AUC:</span>
                          <span>±0.2244</span>
                        </div>
                        <div className="metric-stat-row">
                          <span>Mean PR-AUC:</span>
                          <strong>0.2038</strong>
                        </div>
                      </div>

                      <div className="metric-card">
                        <div className="metric-card-header">Temporal Hold-Out (Final 20% Unseen Time)</div>
                        <div className="metric-stat-row">
                          <span>Evaluation Protocol:</span>
                          <span>Strict chronological future test</span>
                        </div>
                        <div className="metric-stat-row">
                          <span>ROC-AUC:</span>
                          <strong>0.7934</strong>
                        </div>
                        <div className="metric-stat-row">
                          <span>PR-AUC:</span>
                          <strong>0.3775</strong>
                        </div>
                        <div className="metric-stat-row">
                          <span>Precision / Recall:</span>
                          <strong>37.2% / 52.6%</strong>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="modal-right-col" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  {/* 4. SHAP Feature Importance */}
                  <div className="modal-section">
                    <h3 className="modal-section-title">📈 4. SHAP Feature Importance Ranking</h3>
                    <div className="modal-chart-box">
                      <img
                        src="/feature_importance.png"
                        alt="SHAP Feature Importance Ranking"
                        className="modal-chart-img"
                      />
                    </div>
                    <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: '1.4', background: 'rgba(30, 41, 59, 0.4)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <strong>Key Insight:</strong> Distance to forest edge (0.9256) and water proximity (0.9214) dominate elephant habitat selection. Community interface buffer distance (0.5601) and agricultural crop density (0.3284) drive peripheral presence risk near villages.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="modal-footer">
              <button className="modal-dismiss-btn" onClick={() => setIsAboutOpen(false)}>
                Close Transparency Info
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
