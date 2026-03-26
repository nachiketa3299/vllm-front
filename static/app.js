const fileInput = document.getElementById("file-input");
const browseButton = document.getElementById("browse-button");
const generateButton = document.getElementById("generate-button");
const dropzone = document.getElementById("dropzone");
const previewImage = document.getElementById("preview-image");
const userRequestInput = document.getElementById("user-request-input");
const configTableBody = document.getElementById("config-table-body");
const textPreview = document.getElementById("text-preview");
const status = document.getElementById("status");
const logOutput = document.getElementById("log-output");
const modelStatusBadge = document.getElementById("model-status-badge");
const modelRuntimeModel = document.getElementById("model-runtime-model");
const modelRuntimeMaxLen = document.getElementById("model-runtime-max-len");
const modelRuntimeOwnership = document.getElementById("model-runtime-ownership");
const modelRuntimePid = document.getElementById("model-runtime-pid");
const modelRuntimeTheoreticalMax = document.getElementById("model-runtime-theoretical-max");
const modelRuntimeRecommendedMax = document.getElementById("model-runtime-recommended-max");
const modelRuntimeDetail = document.getElementById("model-runtime-detail");
const modelRuntimeCapabilityNote = document.getElementById("model-runtime-capability-note");
const maxModelLenInput = document.getElementById("max-model-len-input");
const modelStartButton = document.getElementById("model-start-button");
const modelStopButton = document.getElementById("model-stop-button");
const modelRefreshButton = document.getElementById("model-refresh-button");
const tokenBudgetPanel = document.querySelector(".budget-panel");
const tokenBudgetCaption = document.getElementById("token-budget-caption");
const tokenBudgetUsed = document.getElementById("token-budget-used");
const tokenBudgetOutput = document.getElementById("token-budget-output");
const tokenBudgetInput = document.getElementById("token-budget-input");
const tokenBudgetOutputValue = document.getElementById("token-budget-output-value");
const tokenBudgetTotal = document.getElementById("token-budget-total");
const tokenBudgetLimit = document.getElementById("token-budget-limit");

const READY_STATUSES = new Set(["running_managed", "running_unmanaged"]);
const PENDING_STATUSES = new Set(["starting", "stopping"]);

let selectedFile = null;
let generationStartedAt = null;
let generationTimerId = null;
let runtimePollId = null;
let budgetEstimateTimerId = null;
let budgetRequestSeq = 0;
let generationInFlight = false;
let modelRuntime = null;
const configInputs = new Map();

function timestamp() {
  return new Date().toLocaleTimeString("ko-KR", { hour12: false });
}

function appendLog(message) {
  const line = `[${timestamp()}] ${message}`;
  if (logOutput.textContent.trim() === "대기 중입니다.") {
    logOutput.textContent = line;
  } else {
    logOutput.textContent += `\n${line}`;
  }
  logOutput.scrollTop = logOutput.scrollHeight;
}

function appendLogs(messages) {
  if (!Array.isArray(messages)) {
    return;
  }
  messages.forEach((message) => appendLog(message));
}

function clearLog() {
  logOutput.textContent = "대기 중입니다.";
}

function startGenerationTimer() {
  stopGenerationTimer();
  generationStartedAt = Date.now();
  generationTimerId = window.setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - generationStartedAt) / 1000);
    setStatus(`vLLM으로 텍스트 생성 중... ${elapsedSeconds}초 경과`);
  }, 1000);
}

function stopGenerationTimer() {
  if (generationTimerId !== null) {
    window.clearInterval(generationTimerId);
    generationTimerId = null;
  }
}

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function setPreview(target, data) {
  target.textContent = data;
}

function resetGeneratedState() {
  setPreview(textPreview, "");
}

function hasPromptText() {
  return userRequestInput.value.trim().length > 0;
}

function hasGenerationInput() {
  return Boolean(selectedFile) || hasPromptText();
}

function isModelReady() {
  return modelRuntime !== null && READY_STATUSES.has(modelRuntime.status);
}

function updateGenerateButtonState() {
  generateButton.disabled = generationInFlight || !hasGenerationInput() || !isModelReady();
}

function resolveBudgetMaxModelLen() {
  const planned = Number.parseInt(maxModelLenInput.value.trim(), 10);
  if (Number.isInteger(planned) && planned > 0) {
    return planned;
  }
  if (modelRuntime && Number.isInteger(modelRuntime.current_max_model_len)) {
    return modelRuntime.current_max_model_len;
  }
  if (modelRuntime && Number.isInteger(modelRuntime.default_max_model_len)) {
    return modelRuntime.default_max_model_len;
  }
  return null;
}

function refreshIdleStatus() {
  if (generationInFlight) {
    return;
  }
  if (!isModelReady()) {
    setStatus("모델이 준비되지 않았습니다. 먼저 모델을 켜세요.");
    return;
  }
  if (selectedFile) {
    setStatus(`선택됨: ${selectedFile.name}`);
    return;
  }
  if (hasPromptText()) {
    setStatus("프롬프트만으로 생성할 수 있습니다.");
    return;
  }
  setStatus("");
}

function updateImagePreview(file) {
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  dropzone.classList.add("has-image");
}

function setSelectedFile(file) {
  selectedFile = file;
  updateGenerateButtonState();
  resetGeneratedState();

  if (!file) {
    previewImage.removeAttribute("src");
    dropzone.classList.remove("has-image");
    refreshIdleStatus();
    scheduleBudgetEstimate();
    return;
  }

  updateImagePreview(file);
  refreshIdleStatus();
  appendLog(`이미지 선택: ${file.name}`);
  scheduleBudgetEstimate();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const rawText = await response.text();
  let payload = {};
  if (rawText) {
    try {
      payload = JSON.parse(rawText);
    } catch {
      throw new Error(rawText || `서버가 JSON이 아닌 응답을 반환했습니다. HTTP ${response.status}`);
    }
  }
  if (!response.ok) {
    throw new Error(payload.detail || `요청에 실패했습니다. HTTP ${response.status}`);
  }
  return payload;
}

function runtimeStatusLabel(statusValue) {
  const labels = {
    stopped: "꺼짐",
    starting: "부팅 중",
    running_managed: "실행 중",
    running_unmanaged: "외부 실행 중",
    stopping: "종료 중",
    error: "오류",
  };
  return labels[statusValue] || statusValue || "알 수 없음";
}

function ownershipLabel(ownershipValue) {
  const labels = {
    managed: "앱 관리",
    unmanaged: "외부 실행",
    none: "없음",
  };
  return labels[ownershipValue] || ownershipValue || "-";
}

function updateRuntimePolling() {
  const shouldPoll = modelRuntime !== null && PENDING_STATUSES.has(modelRuntime.status);
  if (shouldPoll && runtimePollId === null) {
    runtimePollId = window.setInterval(() => {
      loadModelRuntime({ quiet: true });
    }, 2000);
    return;
  }
  if (!shouldPoll && runtimePollId !== null) {
    window.clearInterval(runtimePollId);
    runtimePollId = null;
  }
}

function renderBudgetUnavailable(message) {
  tokenBudgetPanel.classList.remove("over-limit");
  tokenBudgetUsed.style.width = "0%";
  tokenBudgetOutput.style.left = "0%";
  tokenBudgetOutput.style.width = "0%";
  tokenBudgetInput.textContent = "-";
  tokenBudgetOutputValue.textContent = "-";
  tokenBudgetTotal.textContent = "-";
  tokenBudgetLimit.textContent = "-";
  tokenBudgetCaption.textContent = message;
}

function renderTokenBudget(payload) {
  const limit = Number(payload.max_model_len) || 0;
  const inputTokens = Number(payload.input_tokens) || 0;
  const outputTokens = Number(payload.output_tokens) || 0;
  const totalTokens = Number(payload.total_tokens) || 0;
  const inputWidth = limit > 0 ? Math.min((inputTokens / limit) * 100, 100) : 0;
  const totalWidth = limit > 0 ? Math.min((totalTokens / limit) * 100, 100) : 0;

  tokenBudgetPanel.classList.toggle("over-limit", Boolean(payload.exceeds_limit));
  tokenBudgetUsed.style.width = `${inputWidth}%`;
  tokenBudgetOutput.style.left = `${inputWidth}%`;
  tokenBudgetOutput.style.width = `${Math.max(totalWidth - inputWidth, 0)}%`;
  tokenBudgetInput.textContent = `${inputTokens}`;
  tokenBudgetOutputValue.textContent = `${outputTokens}`;
  tokenBudgetTotal.textContent = `${totalTokens}`;
  tokenBudgetLimit.textContent = `${limit}`;

  if (!payload.input_present) {
    tokenBudgetCaption.textContent = "입력이 없어서 예산 계산을 대기 중입니다.";
    return;
  }

  const remaining = Number(payload.remaining_tokens);
  const imageTokens = Number(payload.image_tokens) || 0;
  if (payload.exceeds_limit) {
    tokenBudgetCaption.textContent =
      `한도를 ${Math.abs(remaining)} 토큰 초과합니다. 입력${imageTokens > 0 ? ` (이미지 ${imageTokens} 포함)` : ""} 또는 출력 예약을 줄이세요.`;
    return;
  }

  tokenBudgetCaption.textContent =
    `남은 여유는 ${remaining} 토큰입니다.${imageTokens > 0 ? ` 이미지 입력이 약 ${imageTokens} 토큰을 사용합니다.` : ""}`;
}

async function estimateTokenBudget() {
  const currentRequestSeq = ++budgetRequestSeq;
  if (!hasGenerationInput()) {
    renderBudgetUnavailable("입력이 없어서 예산 계산을 대기 중입니다.");
    return;
  }

  const maxModelLen = resolveBudgetMaxModelLen();
  if (!Number.isInteger(maxModelLen) || maxModelLen <= 0) {
    renderBudgetUnavailable("유효한 max_model_len 값을 지정하면 토큰 예산을 계산합니다.");
    return;
  }

  const formData = new FormData();
  if (selectedFile) {
    formData.append("image", selectedFile);
  }
  formData.append("user_request", userRequestInput.value.trim());
  formData.append(
    "max_completion_tokens",
    configInputs.get("max_completion_tokens")?.value?.trim() || "0"
  );
  formData.append(
    "max_image_bytes",
    configInputs.get("max_image_bytes")?.value?.trim() || "0"
  );
  formData.append("max_model_len", `${maxModelLen}`);

  try {
    const payload = await fetchJson("/api/token-budget", {
      method: "POST",
      body: formData,
    });
    if (currentRequestSeq !== budgetRequestSeq) {
      return;
    }
    renderTokenBudget(payload);
  } catch (error) {
    if (currentRequestSeq !== budgetRequestSeq) {
      return;
    }
    const message = error instanceof Error ? error.message : "토큰 예산을 계산하지 못했습니다.";
    renderBudgetUnavailable(message);
  }
}

function scheduleBudgetEstimate() {
  if (budgetEstimateTimerId !== null) {
    window.clearTimeout(budgetEstimateTimerId);
  }
  budgetEstimateTimerId = window.setTimeout(() => {
    budgetEstimateTimerId = null;
    estimateTokenBudget();
  }, 250);
}

function renderModelRuntime(runtime) {
  modelRuntime = runtime;

  modelStatusBadge.textContent = runtimeStatusLabel(runtime.status);
  modelStatusBadge.classList.remove("running", "pending", "error");
  if (READY_STATUSES.has(runtime.status)) {
    modelStatusBadge.classList.add("running");
  } else if (PENDING_STATUSES.has(runtime.status)) {
    modelStatusBadge.classList.add("pending");
  } else if (runtime.status === "error") {
    modelStatusBadge.classList.add("error");
  }

  modelRuntimeModel.textContent = runtime.model || "없음";
  modelRuntimeMaxLen.textContent =
    runtime.current_max_model_len !== null && runtime.current_max_model_len !== undefined
      ? `${runtime.current_max_model_len}`
      : `${runtime.default_max_model_len} (기본)`;
  modelRuntimeOwnership.textContent = ownershipLabel(runtime.ownership);
  modelRuntimePid.textContent = runtime.pid !== null && runtime.pid !== undefined ? `${runtime.pid}` : "-";
  modelRuntimeTheoreticalMax.textContent =
    runtime.theoretical_max_model_len !== null && runtime.theoretical_max_model_len !== undefined
      ? runtime.theoretical_max_model_len.toLocaleString()
      : "-";
  modelRuntimeRecommendedMax.textContent =
    runtime.recommended_max_model_len !== null && runtime.recommended_max_model_len !== undefined
      ? runtime.recommended_max_model_len.toLocaleString()
      : "-";
  modelRuntimeDetail.textContent = runtime.detail || "추가 정보 없음";
  if (runtime.recommended_max_model_len_reason) {
    modelRuntimeCapabilityNote.textContent = runtime.recommended_max_model_len_reason;
  } else {
    modelRuntimeCapabilityNote.textContent = "추천값 근거를 아직 확보하지 못했습니다.";
  }

  modelStartButton.disabled = !runtime.can_start;
  modelStopButton.disabled = !runtime.can_stop;
  modelRefreshButton.disabled = false;

  if (!maxModelLenInput.value.trim()) {
    maxModelLenInput.value = `${runtime.default_max_model_len}`;
  }

  updateGenerateButtonState();
  updateRuntimePolling();
  refreshIdleStatus();
  scheduleBudgetEstimate();
}

async function loadModelRuntime({ quiet = false } = {}) {
  try {
    const payload = await fetchJson("/api/model/runtime");
    renderModelRuntime(payload);
    return payload;
  } catch (error) {
    const message = error instanceof Error ? error.message : "모델 상태를 불러오지 못했습니다.";
    modelStatusBadge.textContent = "오류";
    modelStatusBadge.classList.remove("running", "pending");
    modelStatusBadge.classList.add("error");
    modelRuntimeDetail.textContent = message;
    modelRuntimeCapabilityNote.textContent = "상한 정보를 읽지 못했습니다.";
    renderBudgetUnavailable("모델 상태를 읽지 못해 토큰 예산을 계산할 수 없습니다.");
    if (!quiet) {
      appendLog(message);
    }
    return null;
  }
}

async function startModel() {
  const raw = maxModelLenInput.value.trim();
  const maxModelLen = Number.parseInt(raw, 10);
  if (!Number.isInteger(maxModelLen) || maxModelLen <= 0) {
    modelRuntimeDetail.textContent = "유효한 max_model_len 값을 입력하세요.";
    modelStatusBadge.textContent = "오류";
    modelStatusBadge.classList.remove("running", "pending");
    modelStatusBadge.classList.add("error");
    scheduleBudgetEstimate();
    return;
  }

  try {
    appendLog(`모델 시작 요청: max_model_len=${maxModelLen}`);
    const payload = await fetchJson("/api/model/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_model_len: maxModelLen }),
    });
    renderModelRuntime(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "모델 시작에 실패했습니다.";
    appendLog(message);
    modelRuntimeDetail.textContent = message;
    modelStatusBadge.textContent = "오류";
    modelStatusBadge.classList.remove("running", "pending");
    modelStatusBadge.classList.add("error");
  }
}

async function stopModel() {
  try {
    appendLog("모델 종료 요청");
    const payload = await fetchJson("/api/model/stop", {
      method: "POST",
    });
    renderModelRuntime(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "모델 종료에 실패했습니다.";
    appendLog(message);
    modelRuntimeDetail.textContent = message;
  }
}

async function generate() {
  if (!hasGenerationInput()) {
    setStatus("이미지 또는 프롬프트를 입력하세요.", true);
    return;
  }
  if (!isModelReady()) {
    setStatus("모델이 준비되지 않았습니다. 먼저 모델을 켜세요.", true);
    return;
  }

  const userRequest = userRequestInput.value.trim();
  const formData = new FormData();
  if (selectedFile) {
    formData.append("image", selectedFile);
  }
  formData.append("user_request", userRequest);
  for (const key of ["max_completion_tokens", "timeout_seconds", "max_image_bytes"]) {
    const input = configInputs.get(key);
    if (input && input.value.trim()) {
      formData.append(key, input.value.trim());
    }
  }
  const jsonOutputInput = configInputs.get("json_output");
  formData.append("json_output", jsonOutputInput && jsonOutputInput.checked ? "true" : "false");

  generationInFlight = true;
  updateGenerateButtonState();
  resetGeneratedState();
  clearLog();
  setStatus("vLLM으로 텍스트 생성 중...");
  appendLog("생성 요청 시작");
  appendLog(selectedFile ? `이미지 전송: ${selectedFile.name}` : "이미지 없이 전송");
  appendLog(`프롬프트 전송: ${userRequest.length}자`);
  appendLog(`JSON 출력: ${jsonOutputInput && jsonOutputInput.checked ? "on" : "off"}`);
  appendLog("생성 중입니다. 큰 이미지와 긴 출력 때문에 수 분 걸릴 수 있습니다.");
  startGenerationTimer();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: formData,
    });

    const rawText = await response.text();
    let payload;
    try {
      payload = JSON.parse(rawText);
    } catch {
      throw new Error(rawText || `서버가 JSON이 아닌 응답을 반환했습니다. HTTP ${response.status}`);
    }

    appendLogs(payload.logs);
    if (!response.ok) {
      throw new Error(payload.detail || `생성에 실패했습니다. HTTP ${response.status}`);
    }

    setPreview(textPreview, typeof payload.output_text === "string" ? payload.output_text : "");
    setStatus("생성이 완료되었습니다. 출력 텍스트를 확인하세요.");
    appendLog("생성 완료");
  } catch (error) {
    const message = error instanceof Error ? error.message : "생성에 실패했습니다.";
    setStatus(message, true);
    appendLog(message);
  } finally {
    stopGenerationTimer();
    generationInFlight = false;
    await loadModelRuntime({ quiet: true });
    updateGenerateButtonState();
    scheduleBudgetEstimate();
  }
}

function renderConfigEntries(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    configTableBody.innerHTML = '<tr><td colspan="2">표시할 설정이 없습니다.</td></tr>';
    return;
  }

  configInputs.clear();
  configTableBody.innerHTML = "";

  entries.forEach((entry) => {
    const row = document.createElement("tr");
    const keyCell = document.createElement("td");
    const valueCell = document.createElement("td");

    keyCell.textContent = typeof entry.label === "string" ? entry.label : entry.key;

    if (entry.editable) {
      if (entry.control === "checkbox") {
        const input = document.createElement("input");
        input.type = "checkbox";
        input.className = "config-checkbox";
        input.checked = Boolean(entry.value);
        input.dataset.configKey = entry.key;
        valueCell.appendChild(input);
        configInputs.set(entry.key, input);
      } else {
        const input = document.createElement("input");
        input.type = "number";
        input.min = "1";
        input.step = "1";
        input.className = "config-input";
        input.value = `${entry.value ?? ""}`;
        input.dataset.configKey = entry.key;
        valueCell.appendChild(input);
        configInputs.set(entry.key, input);
      }
    } else {
      const code = document.createElement("code");
      code.textContent = `${entry.value ?? ""}`;
      valueCell.appendChild(code);
    }

    row.appendChild(keyCell);
    row.appendChild(valueCell);
    configTableBody.appendChild(row);
  });

  ["max_completion_tokens", "max_image_bytes"].forEach((key) => {
    const input = configInputs.get(key);
    if (!input) {
      return;
    }
    input.addEventListener("input", scheduleBudgetEstimate);
    input.addEventListener("change", scheduleBudgetEstimate);
  });
}

async function loadConfig() {
  try {
    const payload = await fetchJson("/api/config");
    if (!Array.isArray(payload.entries)) {
      throw new Error("설정 정보를 불러오지 못했습니다.");
    }
    renderConfigEntries(payload.entries);
    scheduleBudgetEstimate();
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "설정 정보를 불러오지 못했습니다.";
    configTableBody.innerHTML = `<tr><td colspan="2">${message}</td></tr>`;
    appendLog(message);
  }
}

browseButton.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setSelectedFile(file || null);
});

userRequestInput.addEventListener("input", () => {
  updateGenerateButtonState();
  refreshIdleStatus();
  scheduleBudgetEstimate();
});

maxModelLenInput.addEventListener("input", scheduleBudgetEstimate);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file && file.type.startsWith("image/")) {
    fileInput.files = event.dataTransfer.files;
    setSelectedFile(file);
  } else {
    setStatus("이미지 파일만 업로드할 수 있습니다.", true);
    appendLog("이미지가 아닌 파일은 거부됨");
  }
});

generateButton.addEventListener("click", generate);
modelStartButton.addEventListener("click", startModel);
modelStopButton.addEventListener("click", stopModel);
modelRefreshButton.addEventListener("click", () => loadModelRuntime());

loadConfig();
loadModelRuntime();
updateGenerateButtonState();
renderBudgetUnavailable("입력 또는 설정을 바꾸면 토큰 사용량을 계산합니다.");
