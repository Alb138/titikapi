import type { FeatureCollection } from "geojson";
import * as maplibregl from "maplibre-gl";
// MapLibre 6 runs its tile parsing in a module worker; let Vite bundle it and pass the URL.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "./style.css";

maplibregl.setWorkerUrl(workerUrl);

const DATA = "https://huggingface.co/datasets/Alb138/titikapi-live/resolve/main/latest";
const BASEMAP = "https://tiles.openfreemap.org/styles/positron";
const PROVINCES_SHOWN = 6;

type Province = { name: string; expected_fire_cells: number };
type Lead = { lead: number; date: string; expected_fire_cells: number; provinces: Province[] };
type Forecast = {
  issue_date: string;
  fire_data_through: string;
  png_bounds: [number, number][];
  leads: Lead[];
};
type LeadScore = {
  lead: number;
  days: number;
  fires: number;
  final_top5: number | null;
  persistence_top5: number | null;
};
type Scorecard = {
  n_scored: number;
  first_issue_date?: string;
  windows: { since_launch?: LeadScore[] };
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// Dates in the data are local calendar days in Indonesia; format them without time zones.
const asDate = (iso: string) => new Date(`${iso}T12:00:00Z`);
const dayLong = (iso: string) =>
  asDate(iso).toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", timeZone: "UTC" });
const dayShort = (iso: string) =>
  asDate(iso).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", timeZone: "UTC" });
const fmtCells = (x: number) => (x < 10 ? x.toFixed(1) : Math.round(x).toLocaleString("en-GB"));
const pct = (x: number) => `${Math.round(x * 100)}%`;

async function getJSON<T>(name: string): Promise<T> {
  const r = await fetch(`${DATA}/${name}`, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json() as Promise<T>;
}

// ------------------------------------------------------------------ map

const map = new maplibregl.Map({
  container: "map",
  style: BASEMAP,
  bounds: [
    [95, -6.5],
    [119.5, 6.5],
  ],
  fitBoundsOptions: { padding: fitPadding() },
  attributionControl: { compact: true },
  dragRotate: false,
  pitchWithRotate: false,
});
map.touchZoomRotate.disableRotation();
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

function fitPadding() {
  const wide = window.innerWidth > 760;
  return wide ? { top: 24, bottom: 150, left: 380, right: 40 } : { top: 16, bottom: 16, left: 16, right: 16 };
}

const mapReady = new Promise<void>((resolve) => map.on("load", () => resolve()));

function firstLabelLayer(): string | undefined {
  return map.getStyle().layers.find((l) => l.type === "symbol")?.id;
}

function addForecastLayers(f: Forecast, observed: FeatureCollection) {
  map.addSource("risk", {
    type: "image",
    url: `${DATA}/risk_d1.png`,
    coordinates: f.png_bounds as [[number, number], [number, number], [number, number], [number, number]],
  });
  const beforeLabels = firstLabelLayer();
  map.addLayer(
    {
      id: "risk",
      type: "raster",
      source: "risk",
      paint: { "raster-opacity": 0.85, "raster-resampling": "nearest", "raster-fade-duration": 0 },
    },
    beforeLabels,
  );
  map.addSource("fires", { type: "geojson", data: observed });
  map.addLayer(
    {
      id: "fires",
      type: "circle",
      source: "fires",
      paint: {
        "circle-color": "#2563c9",
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 2.2, 8, 5],
        "circle-stroke-color": "#f7f8f7",
        "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 4, 0.6, 8, 1.2],
        "circle-opacity": ["match", ["get", "days_ago"], 0, 1, 1, 0.7, 0.45],
      },
    },
    beforeLabels,
  );

  const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
  map.on("mousemove", "fires", (e) => {
    const p = e.features?.[0]?.properties as { date: string; detections: number; province: string };
    if (!p) return;
    map.getCanvas().style.cursor = "pointer";
    const n = Number(p.detections);
    popup
      .setLngLat(e.lngLat)
      .setHTML(`Fire detected ${dayLong(p.date)}<br>${n} detection${n === 1 ? "" : "s"}, ${p.province}`)
      .addTo(map);
  });
  map.on("mouseleave", "fires", () => {
    map.getCanvas().style.cursor = "";
    popup.remove();
  });
}

// ------------------------------------------------------------------ panel + day strip

let showAllProvinces = false;

function renderDay(f: Forecast, lead: number) {
  const L = f.leads[lead - 1];
  $("day-title").textContent = dayLong(L.date);
  $("day-sub").textContent =
    `Day ${lead} of 7. Forecast made ${dayLong(f.issue_date)} ` +
    `from fires and weather up to that day.`;
  $("expected").innerHTML =
    `About <strong>${fmtCells(L.expected_fire_cells)}</strong> of the region's 9,275 land cells ` +
    `are expected to have a fire.`;

  const list = $("provinces");
  const max = Math.max(...L.provinces.map((p) => p.expected_fire_cells), 1e-9);
  const shown = showAllProvinces ? L.provinces : L.provinces.slice(0, PROVINCES_SHOWN);
  list.replaceChildren(
    ...shown.map((p) => {
      const li = document.createElement("li");
      li.innerHTML =
        `<span>${p.name}</span><span class="value">${fmtCells(p.expected_fire_cells)}</span>` +
        `<span class="bar"><span style="width:${(100 * p.expected_fire_cells) / max}%"></span></span>`;
      return li;
    }),
  );
  $("more-provinces").hidden = showAllProvinces;

  document.querySelectorAll<HTMLButtonElement>(".day").forEach((b) => {
    b.setAttribute("aria-pressed", String(Number(b.dataset.lead) === lead));
  });
  const src = map.getSource("risk") as maplibregl.ImageSource | undefined;
  src?.updateImage({ url: `${DATA}/risk_d${lead}.png` });
}

function buildStrip(f: Forecast, onPick: (lead: number) => void) {
  const max = Math.max(...f.leads.map((l) => l.expected_fire_cells), 1e-9);
  const strip = $("day-strip");
  strip.replaceChildren(
    ...f.leads.map((L) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "day";
      b.dataset.lead = String(L.lead);
      b.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-label", `${dayLong(L.date)}: about ${fmtCells(L.expected_fire_cells)} fire cells expected`);
      b.style.setProperty("--h", `${Math.max(4, (54 * L.expected_fire_cells) / max)}px`);
      b.innerHTML =
        `<span class="col" aria-hidden="true"></span>` +
        `<span class="when">${dayShort(L.date)}</span>` +
        `<span class="count">${fmtCells(L.expected_fire_cells)} cells</span>`;
      b.addEventListener("click", () => onPick(L.lead));
      return b;
    }),
  );
  // arrow keys move between days
  strip.addEventListener("keydown", (e) => {
    const cur = Number((document.activeElement as HTMLElement)?.dataset?.lead);
    if (!cur || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const next = Math.min(7, Math.max(1, cur + (e.key === "ArrowRight" ? 1 : -1)));
    (strip.querySelector(`[data-lead="${next}"]`) as HTMLButtonElement).focus();
    onPick(next);
    e.preventDefault();
  });
}

// ------------------------------------------------------------------ accuracy

function renderTested() {
  // Test period Jan 2023 - Oct 2026 (reports/comparison.csv in the repository)
  const rows: [string, number, boolean][] = [
    ["TitikApi", 0.425, true],
    ["Assume today's fires continue", 0.37, false],
    ["Usual fire season for the place", 0.253, false],
    ["Random 5% of the map", 0.05, false],
  ];
  $("tested").innerHTML = rows
    .map(
      ([label, v, ours]) =>
        `<dt>${label}</dt><div class="track${ours ? " ours" : ""}" aria-hidden="true">` +
        `<span style="width:${v * 100}%"></span></div><dd>${pct(v)}</dd>`,
    )
    .join("");
}

function renderLive(s: Scorecard | null) {
  const box = $("live-score");
  const lead1 = s?.windows?.since_launch?.find((r) => r.lead === 1);
  if (!s || !lead1 || lead1.final_top5 == null || lead1.persistence_top5 == null) {
    box.innerHTML =
      `<p class="measure-note">Each forecast is graded once its day has passed and the satellite ` +
      `data for it is final, about three days later. The first grades appear here soon after the ` +
      `forecasts that started on 16 November 2026.</p>`;
    return;
  }
  box.innerHTML =
    `<p class="measure-note">Since ${dayLong(s.first_issue_date ?? "")}: ${lead1.days} days graded, ` +
    `${lead1.fires.toLocaleString("en-GB")} fire cells. Share of next-day fires in the riskiest 5%:</p>` +
    `<dl class="compare">` +
    `<dt>TitikApi</dt><div class="track ours" aria-hidden="true"><span style="width:${lead1.final_top5 * 100}%"></span></div><dd>${pct(lead1.final_top5)}</dd>` +
    `<dt>Assume today's fires continue</dt><div class="track" aria-hidden="true"><span style="width:${lead1.persistence_top5 * 100}%"></span></div><dd>${pct(lead1.persistence_top5)}</dd>` +
    `</dl>` +
    `<p class="measure-note">Few fires burn in the rainy season (November to April), so these ` +
    `numbers move around until the dry season.</p>`;
}

// ------------------------------------------------------------------ start

async function main() {
  renderTested();
  try {
    const [forecast, observed] = await Promise.all([
      getJSON<Forecast>("forecast.json"),
      getJSON<FeatureCollection>("observed.geojson"),
    ]);
    await mapReady;
    addForecastLayers(forecast, observed);
    let current = 1;
    const pick = (lead: number) => {
      current = lead;
      renderDay(forecast, lead);
    };
    buildStrip(forecast, pick);
    $("more-provinces").addEventListener("click", () => {
      showAllProvinces = true;
      renderDay(forecast, current);
    });
    $<HTMLInputElement>("show-fires").addEventListener("change", (e) => {
      const on = (e.target as HTMLInputElement).checked;
      map.setLayoutProperty("fires", "visibility", on ? "visible" : "none");
    });
    pick(1);
    $("status").textContent = `Updated daily at 19:00 WIB`;
  } catch (err) {
    const s = $("status");
    s.classList.add("error");
    s.textContent = `Today's forecast could not be loaded (${(err as Error).message}). Reload the page in a minute.`;
  }
  try {
    renderLive(await getJSON<Scorecard>("scorecard.json"));
  } catch {
    renderLive(null);
  }
}

main();
