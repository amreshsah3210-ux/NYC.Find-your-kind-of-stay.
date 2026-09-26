const numberFormat = new Intl.NumberFormat("en-US");
const moneyFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const roomColors = {
  "Private room": "private",
  "Entire home/apt": "entire",
  "Shared room": "shared",
};

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function formatRoomTag(roomType) {
  const className = roomColors[roomType] || "private";
  return `<span class="room-tag ${className}-room">${escapeHtml(roomType)}</span>`;
}

function renderSummary(data) {
  const status = document.getElementById("dataset-status");
  const dot = document.getElementById("status-dot");
  if (!data.available) {
    status.textContent = "Dataset unavailable";
    dot.classList.add("error");
    document.getElementById("dataset-count").textContent = "Classifier remains available";
    document.getElementById("room-bars").innerHTML = `<p class="empty-cell">${data.message}</p>`;
    document.getElementById("listing-rows").innerHTML = `<tr><td colspan="4" class="empty-cell">Dataset could not be loaded.</td></tr>`;
    return;
  }

  status.textContent = "Dataset connected";
  dot.classList.add("ready");
  document.getElementById("dataset-count").textContent = `${numberFormat.format(data.listing_count)} listings loaded`;
  document.getElementById("metric-listings").textContent = numberFormat.format(data.listing_count);
  document.getElementById("metric-boroughs").textContent = data.boroughs.length;
  document.getElementById("metric-price").textContent = data.median_price === null ? "--" : `$${moneyFormat.format(data.median_price)}`;

  const leadingRoom = data.room_types[0];
  document.getElementById("metric-room").textContent = leadingRoom?.name || "--";
  document.getElementById("metric-room-share").textContent = leadingRoom ? `${leadingRoom.share}% of listings` : "Share of listings";
  const roomBars = document.getElementById("room-bars");
  roomBars.innerHTML = data.room_types.map((room, index) => {
    const color = roomColors[room.name] || ["private", "entire", "shared"][index % 3];
    return `<div class="room-bar-row"><span class="room-bar-label">${escapeHtml(room.name)}</span><div class="bar-track"><div class="bar-fill ${color}" style="width:${room.share}%"></div></div><span class="bar-value">${room.share}%</span></div>`;
  }).join("");

  document.getElementById("borough-strip").innerHTML = data.boroughs.map((borough) =>
    `<span class="borough-chip">${escapeHtml(borough.name)}<b>${numberFormat.format(borough.count)}</b></span>`
  ).join("");
  document.getElementById("neighbourhood-options").innerHTML = data.neighbourhoods.map((name) =>
    `<option value="${escapeHtml(name)}">`
  ).join("");
  document.getElementById("listing-rows").innerHTML = data.listings.map((listing) =>
    `<tr><td title="${escapeHtml(listing.name)}">${escapeHtml(listing.name || "Untitled listing")}</td><td>${escapeHtml(listing.neighbourhood)}, ${escapeHtml(listing.neighbourhood_group)}</td><td>${formatRoomTag(listing.room_type)}</td><td>$${moneyFormat.format(listing.price)}</td></tr>`
  ).join("");
}

async function loadSummary() {
  try {
    const response = await fetch("/api/summary");
    if (!response.ok) throw new Error("Could not load the dataset summary.");
    renderSummary(await response.json());
  } catch (error) {
    document.getElementById("dataset-status").textContent = "API unavailable";
    document.getElementById("status-dot").classList.add("error");
    document.getElementById("dataset-count").textContent = "Start the FastAPI server to connect";
  }
}

function showPrediction(data) {
  const result = document.getElementById("prediction-result");
  const probabilities = Object.entries(data.probabilities || {}).sort((a, b) => b[1] - a[1]);
  result.classList.remove("error");
  result.innerHTML = `<div class="result-top"><div><div class="result-label">PREDICTED ROOM TYPE</div><div class="result-name">${escapeHtml(data.room_type)}</div></div><span class="result-confidence">${data.confidence === null ? "" : `${data.confidence}% confidence`}</span></div>${probabilities.length ? `<div class="probability-list">${probabilities.map(([name, chance]) => `<div class="probability-row"><span>${escapeHtml(name)}</span><div class="probability-track"><div class="probability-fill" style="width:${chance}%"></div></div><span>${chance}%</span></div>`).join("")}</div>` : ""}`;
  result.hidden = false;
}

document.getElementById("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type=submit]");
  const result = document.getElementById("prediction-result");
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  ["latitude", "longitude", "price", "reviews_per_month"].forEach((key) => { payload[key] = Number(payload[key]); });
  ["minimum_nights", "number_of_reviews", "calculated_host_listings_count", "availability_365"].forEach((key) => { payload[key] = Number.parseInt(payload[key], 10); });
  button.disabled = true;
  button.querySelector("span").textContent = "Classifying...";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Prediction failed.");
    showPrediction(data);
  } catch (error) {
    result.classList.add("error");
    result.textContent = error.message;
    result.hidden = false;
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Classify room type";
  }
});

if (window.lucide) window.lucide.createIcons();
loadSummary();