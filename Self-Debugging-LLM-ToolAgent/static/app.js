async function loadTools() {
  const response = await fetch("/api/tools");
  const data = await response.json();
  document.getElementById("tools").innerHTML = data.tools.map(tool => `
    <div class="tool">
      <strong>${tool.name} · ${tool.category}</strong>
      <span>${tool.description}</span>
    </div>
  `).join("");
}

async function loadProviderStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    const label = status.available ? "Available" : "Fallback mode";
    document.getElementById("provider-status").textContent =
      `${label}: ${status.provider} / ${status.model} — ${status.detail}`;
  } catch (error) {
    document.getElementById("provider-status").textContent =
      "Provider status unavailable";
  }
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = String(value);
  return element.innerHTML;
}

let uploadedImagePath = "";

async function loadHistory() {
  try {
    const response = await fetch("/api/history?limit=20");
    const data = await response.json();
    const container = document.getElementById("history-list");
    container.innerHTML = data.executions.length
      ? data.executions.map(item => `
          <div class="history-item">
            <strong>#${item.id} · ${escapeHtml(item.query)}</strong>
            <span>Tools: ${escapeHtml(item.selected_tools.join(", ") || "None")}
              · ${item.recovered ? "Recovered" : "No recovery"}
              · ${escapeHtml(item.created_at)}</span>
          </div>
        `).join("")
      : "<p>No executions stored yet.</p>";
  } catch (error) {
    document.getElementById("history-list").textContent =
      "Could not load execution history.";
  }
}

document.getElementById("image").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  const status = document.getElementById("upload-status");
  status.textContent = "Uploading image...";
  const formData = new FormData();
  formData.append("image", file);
  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Image upload failed");
    uploadedImagePath = data.uploaded[0].path;
    status.textContent = `Uploaded: ${uploadedImagePath}`;
  } catch (error) {
    uploadedImagePath = "";
    status.textContent = error.message;
  }
});

document.getElementById("request-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  button.textContent = "Running...";

  try {
    let query = document.getElementById("query").value;
    if (!query && uploadedImagePath) {
      query = `Inspect image ${uploadedImagePath}`;
      document.getElementById("query").value = query;
    } else if (uploadedImagePath && !query.includes(uploadedImagePath)) {
      query = `${query} ${uploadedImagePath}`;
    }
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        query,
        text: document.getElementById("text").value,
      }),
    });
    const report = await response.json();
    if (!response.ok) throw new Error(report.error || "Request failed");

    document.getElementById("report").classList.remove("hidden");
    document.getElementById("use-tool").textContent = report.use_tool ? "Yes" : "No";
    document.getElementById("selected-tools").textContent =
      report.selected_tools.length ? report.selected_tools.join(", ") : "None";
    document.getElementById("recovered").textContent = report.recovered ? "Yes" : "No";

    document.getElementById("plan").innerHTML = report.plan
      .map(step => `<li>${escapeHtml(step)}</li>`).join("");

    document.getElementById("calls").innerHTML = report.tool_calls.length
      ? report.tool_calls.map(call => `
          <div class="call ${call.success ? "" : "failed"}">
            <strong>Step ${call.step}: ${escapeHtml(call.name)}</strong>
            <div><code>${escapeHtml(JSON.stringify(call.arguments))}</code></div>
            <div>${call.success
              ? `Result: ${escapeHtml(call.result)}`
              : `Error: ${escapeHtml(call.error)}`}</div>
          </div>
        `).join("")
      : "<p>No tool calls were made.</p>";

    document.getElementById("answer").textContent = report.answer;
    document.getElementById("diagnostics").textContent =
      JSON.stringify({
        errors: report.errors,
        critical_error: report.critical_error,
      }, null, 2);
    await loadHistory();
  } catch (error) {
    document.getElementById("report").classList.remove("hidden");
    document.getElementById("answer").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Run Agent";
  }
});

loadTools();
loadProviderStatus();
loadHistory();
