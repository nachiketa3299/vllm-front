const fileInput = document.getElementById("file-input");
const browseButton = document.getElementById("browse-button");
const resetImageButton = document.getElementById("reset-image-button");
const generateButton = document.getElementById("generate-button");
const dropzone = document.getElementById("dropzone");
const previewImage = document.getElementById("preview-image");
const userRequestInput = document.getElementById("user-request-input");
const configTableBody = document.getElementById("config-table-body");
const configInfoTableBody = document.getElementById("config-info-table-body");
const textPreview = document.getElementById("text-preview");
const reasoningPanel = document.getElementById("reasoning-panel");
const reasoningPreview = document.getElementById("reasoning-preview");
const copyOutputButton = document.getElementById("copy-output-button");
const logOutput = document.getElementById("log-output");
const modelStatusBadge = document.getElementById("model-status-badge");
const modelRuntimeModel = document.getElementById("model-runtime-model");
const modelRuntimeMaxLen = document.getElementById("model-runtime-max-len");
const modelRuntimeBaseUrl = document.getElementById("model-runtime-base-url");
const tokenBudgetPanel = document.querySelector(".budget-panel");
const imageMetaEl = document.getElementById("image-meta");
const textMetaEl = document.getElementById("text-meta");
const tokenBudgetUsed = document.getElementById("token-budget-used");
const tokenBudgetOutput = document.getElementById("token-budget-output");
const tokenBudgetInput = document.getElementById("token-budget-input");
const tokenBudgetOutputValue = document.getElementById("token-budget-output-value");
const tokenBudgetTotal = document.getElementById("token-budget-total");
const tokenBudgetLimit = document.getElementById("token-budget-limit");

let selectedFile = null;
let generationStartedAt = null;
let generationTimerId = null;
let budgetEstimateTimerId = null;
let budgetRequestSeq = 0;
let generationInFlight = false;
let modelInfo = null;
const configInputs = new Map();

function timestamp() {
  return new Date().toLocaleTimeString("ko-KR", { hour12: false });
}

const LOG_IMPORTANT_PATTERNS = [
  /\[ERROR\]/,
  /\[WARN\]/,
  /실패/,
  /에러/,
  /오류/,
  /거부/,
  /초과/,
  /중단/,
  /취소/,
  /error/i,
  /fail/i,
  /disconnect/i,
  /cancel/i,
  /timeout/i,
  /unhandled/i,
  /warning/i,
  /invalid/i,
  /empty response/i,
  /cannot identify/i,
  /HTTP [45]\d\d/,
];

function isImportantLog(message) {
  const text = String(message || "");
  return LOG_IMPORTANT_PATTERNS.some((pattern) => pattern.test(text));
}

function appendLog(message, options = {}) {
  const { force = false } = options;
  if (!force && !isImportantLog(message)) return;
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

const GENERATE_BUTTON_IDLE_LABEL = "생성";

function updateGenerateButtonTimerLabel() {
  const elapsedSeconds = Math.floor((Date.now() - generationStartedAt) / 1000);
  generateButton.textContent = `생성 중 (${elapsedSeconds}초 경과)`;
}

function startGenerationTimer() {
  stopGenerationTimer();
  generationStartedAt = Date.now();
  updateGenerateButtonTimerLabel();
  generationTimerId = window.setInterval(updateGenerateButtonTimerLabel, 1000);
}

function stopGenerationTimer() {
  if (generationTimerId !== null) {
    window.clearInterval(generationTimerId);
    generationTimerId = null;
  }
  generateButton.textContent = GENERATE_BUTTON_IDLE_LABEL;
}

function setStatus(message, isError = false) {
  // status div 제거됨 — 메시지는 로그 영역으로만 흐름
  if (!message) return;
  appendLog(isError ? `[ERROR] ${message}` : message);
}

function setPreview(target, data) {
  target.textContent = data;
  if (target === textPreview) {
    updateCopyButtonState();
  }
}

function updateCopyButtonState() {
  const hasContent = textPreview.textContent.length > 0;
  copyOutputButton.disabled = !hasContent;
}

function setReasoning(text) {
  const value = typeof text === "string" ? text : "";
  reasoningPreview.textContent = value;
  reasoningPanel.hidden = value.length === 0;
}

function resetGeneratedState() {
  setPreview(textPreview, "");
  setReasoning("");
}

function hasPromptText() {
  return userRequestInput.value.trim().length > 0;
}

function hasGenerationInput() {
  return Boolean(selectedFile) || hasPromptText();
}

function isModelReady() {
  return modelInfo !== null && modelInfo.online === true;
}

function updateGenerateButtonState() {
  generateButton.disabled = generationInFlight || !hasGenerationInput() || !isModelReady();
}

function refreshIdleStatus() {
  if (generationInFlight) {
    return;
  }
  if (!isModelReady()) {
    setStatus("vLLM 서버에 연결되지 않았습니다.");
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

function clearInputMeta() {
  if (imageMetaEl) imageMetaEl.textContent = "";
  if (textMetaEl) textMetaEl.textContent = "";
}

function renderInputMeta(payload) {
  if (imageMetaEl) {
    if (selectedFile) {
      const kb = Math.max(1, Math.round(selectedFile.size / 1024));
      const tokens = Number(payload?.image_tokens) || 0;
      imageMetaEl.textContent = `${kb} KB · ${tokens} tokens`;
    } else {
      imageMetaEl.textContent = "";
    }
  }
  if (textMetaEl) {
    const chars = userRequestInput.value.length;
    if (chars > 0) {
      const tokens = Number(payload?.text_tokens) || 0;
      textMetaEl.textContent = `${chars}자 · ${tokens} tokens`;
    } else {
      textMetaEl.textContent = "";
    }
  }
}

function renderBudgetUnavailable() {
  tokenBudgetPanel.classList.remove("over-limit");
  tokenBudgetUsed.style.width = "0%";
  tokenBudgetOutput.style.left = "0%";
  tokenBudgetOutput.style.width = "0%";
  tokenBudgetInput.textContent = "-";
  tokenBudgetOutputValue.textContent = "-";
  tokenBudgetTotal.textContent = "-";
  tokenBudgetLimit.textContent = "-";
  clearInputMeta();
}

function renderTokenBudget(payload) {
  const limit = Number(payload.max_model_len) || 0;
  const inputTokens = Number(payload.input_tokens) || 0;
  const outputTokens = Number(payload.output_tokens) || 0;
  const totalTokens = Number(payload.total_tokens) || 0;
  const inputWidth = limit > 0 ? Math.min((inputTokens / limit) * 100, 100) : 0;
  const totalWidth = limit > 0 ? Math.min((totalTokens / limit) * 100, 100) : 0;

  tokenBudgetPanel.classList.toggle("over-limit", Boolean(payload.exceeds_limit));
  const outputSpan = Math.max(totalWidth - inputWidth, 0);
  tokenBudgetUsed.style.width = inputTokens > 0 ? `max(2px, ${inputWidth}%)` : "0";
  tokenBudgetOutput.style.left = `${inputWidth}%`;
  tokenBudgetOutput.style.width = outputTokens > 0 ? `max(2px, ${outputSpan}%)` : "0";
  tokenBudgetInput.textContent = `${inputTokens}`;
  tokenBudgetOutputValue.textContent = `${outputTokens}`;
  tokenBudgetTotal.textContent = `${totalTokens}`;
  tokenBudgetLimit.textContent = `${limit}`;

  renderInputMeta(payload);
}

async function estimateTokenBudget() {
  const currentRequestSeq = ++budgetRequestSeq;
  if (!hasGenerationInput()) {
    renderBudgetUnavailable("입력이 없어서 예산 계산을 대기 중입니다.");
    return;
  }
  if (!isModelReady()) {
    renderBudgetUnavailable("vLLM 서버에 연결되지 않아 토큰 예산을 계산할 수 없습니다.");
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

function renderModelInfo(info) {
  modelInfo = info;

  modelStatusBadge.classList.remove("running", "pending", "error");
  if (info.online) {
    modelStatusBadge.textContent = "연결됨";
    modelStatusBadge.classList.add("running");
    modelRuntimeModel.textContent = info.model || "-";
    modelRuntimeMaxLen.textContent =
      info.max_model_len !== null && info.max_model_len !== undefined
        ? info.max_model_len.toLocaleString()
        : "-";
  } else {
    modelStatusBadge.textContent = "오프라인";
    modelStatusBadge.classList.add("error");
    modelRuntimeModel.textContent = "-";
    modelRuntimeMaxLen.textContent = "-";
  }

  updateGenerateButtonState();
  refreshIdleStatus();
  scheduleBudgetEstimate();
}

async function loadRuntimeInfo() {
  // 시작 시 설정된 vLLM 서버 주소를 정적으로 표시 (probe 아님).
  try {
    const payload = await fetchJson("/api/runtime");
    modelRuntimeBaseUrl.textContent = payload.vllm_base_url || "-";
  } catch (error) {
    modelRuntimeBaseUrl.textContent = "-";
  }
}

async function loadModelInfo({ quiet = false } = {}) {
  try {
    const payload = await fetchJson("/api/model/info");
    renderModelInfo(payload);
    return payload;
  } catch (error) {
    const message = error instanceof Error ? error.message : "모델 정보를 불러오지 못했습니다.";
    modelInfo = null;
    modelStatusBadge.textContent = "오류";
    modelStatusBadge.classList.remove("running", "pending");
    modelStatusBadge.classList.add("error");
    renderBudgetUnavailable("모델 정보를 읽지 못해 토큰 예산을 계산할 수 없습니다.");
    updateGenerateButtonState();
    if (!quiet) {
      appendLog(message);
    }
    return null;
  }
}

async function generate() {
  if (!hasGenerationInput()) {
    setStatus("이미지 또는 프롬프트를 입력하세요.", true);
    return;
  }
  if (!isModelReady()) {
    setStatus("vLLM 서버에 연결되지 않았습니다.", true);
    return;
  }

  const userRequest = userRequestInput.value.trim();
  const formData = new FormData();
  if (selectedFile) {
    formData.append("image", selectedFile);
  }
  formData.append("user_request", userRequest);
  for (const key of ["max_completion_tokens", "temperature"]) {
    const input = configInputs.get(key);
    if (input && input.value.trim()) {
      formData.append(key, input.value.trim());
    }
  }
  const jsonOutputInput = configInputs.get("json_output");
  formData.append("json_output", jsonOutputInput && jsonOutputInput.checked ? "true" : "false");
  const thinkingInput = configInputs.get("enable_thinking");
  formData.append("enable_thinking", thinkingInput && thinkingInput.checked ? "true" : "false");

  generationInFlight = true;
  updateGenerateButtonState();
  resetGeneratedState();
  clearLog();
  setStatus("");
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
    setReasoning(typeof payload.reasoning_text === "string" ? payload.reasoning_text : "");
    setStatus("생성이 완료되었습니다. 출력 텍스트를 확인하세요.");
    const elapsedSec = generationStartedAt !== null
      ? (Date.now() - generationStartedAt) / 1000
      : 0;
    appendLog(`✓ 생성 완료 (${elapsedSec.toFixed(1)}s)`, { force: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "생성에 실패했습니다.";
    setStatus(message, true);
    appendLog(message);
  } finally {
    stopGenerationTimer();
    generationInFlight = false;
    await loadModelInfo({ quiet: true });
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
    if (typeof entry.key === "string" && entry.key) {
      const tipLines = [entry.key];
      if (typeof entry.description === "string" && entry.description) {
        tipLines.push(entry.description);
      }
      keyCell.dataset.tip = tipLines.join("\n");
      keyCell.classList.add("config-key-cell");
    }

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
        input.min = typeof entry.min === "string" ? entry.min : "1";
        input.step = typeof entry.step === "string" ? entry.step : "1";
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

  ["max_completion_tokens"].forEach((key) => {
    const input = configInputs.get(key);
    if (!input) {
      return;
    }
    input.addEventListener("input", scheduleBudgetEstimate);
    input.addEventListener("change", scheduleBudgetEstimate);
  });
}

function renderConfigInfoEntries(entries) {
  if (!configInfoTableBody) {
    return;
  }
  configInfoTableBody.innerHTML = "";
  if (!Array.isArray(entries) || entries.length === 0) {
    return;
  }
  entries.forEach((entry) => {
    const row = document.createElement("tr");
    const keyCell = document.createElement("td");
    const valueCell = document.createElement("td");

    keyCell.textContent = typeof entry.label === "string" ? entry.label : entry.key;
    if (typeof entry.key === "string" && entry.key) {
      const tipLines = [entry.key];
      if (typeof entry.description === "string" && entry.description) {
        tipLines.push(entry.description);
      }
      keyCell.dataset.tip = tipLines.join("\n");
      keyCell.classList.add("config-key-cell");
    }

    const code = document.createElement("code");
    code.textContent = `${entry.value ?? ""}`;
    valueCell.appendChild(code);

    row.appendChild(keyCell);
    row.appendChild(valueCell);
    configInfoTableBody.appendChild(row);
  });
}

async function loadConfig() {
  try {
    const payload = await fetchJson("/api/config");
    if (!Array.isArray(payload.entries)) {
      throw new Error("설정 정보를 불러오지 못했습니다.");
    }
    renderConfigEntries(payload.entries);
    renderConfigInfoEntries(payload.info_entries);
    scheduleBudgetEstimate();
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "설정 정보를 불러오지 못했습니다.";
    configTableBody.innerHTML = `<tr><td colspan="2">${message}</td></tr>`;
    appendLog(message);
  }
}

browseButton.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setSelectedFile(file || null);
});

function updatePromptContentClass() {
  userRequestInput.classList.toggle("has-content", hasPromptText());
}

userRequestInput.addEventListener("input", () => {
  updatePromptContentClass();
  updateGenerateButtonState();
  refreshIdleStatus();
  scheduleBudgetEstimate();
});

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
resetImageButton.addEventListener("click", () => {
  fileInput.value = "";
  setSelectedFile(null);
});

async function copyToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  const selection = document.getSelection();
  const previousRange = selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;
  textarea.select();
  try {
    const ok = document.execCommand("copy");
    if (!ok) throw new Error("execCommand copy returned false");
  } finally {
    document.body.removeChild(textarea);
    if (previousRange && selection) {
      selection.removeAllRanges();
      selection.addRange(previousRange);
    }
  }
}

let copyResetTimerId = null;
copyOutputButton.addEventListener("click", async () => {
  const text = textPreview.textContent;
  if (!text) return;
  try {
    await copyToClipboard(text);
    copyOutputButton.textContent = "복사됨";
    copyOutputButton.classList.add("copied");
    if (copyResetTimerId !== null) window.clearTimeout(copyResetTimerId);
    copyResetTimerId = window.setTimeout(() => {
      copyOutputButton.textContent = "복사";
      copyOutputButton.classList.remove("copied");
      copyResetTimerId = null;
    }, 1200);
  } catch (error) {
    const message = error instanceof Error ? error.message : "클립보드 복사 실패";
    appendLog(`복사 실패: ${message}`);
  }
});

loadConfig();
loadRuntimeInfo();
loadModelInfo();
updateGenerateButtonState();

const MODEL_INFO_POLL_MS = 20000;
window.setInterval(() => {
  if (generationInFlight) return;
  loadModelInfo({ quiet: true });
}, MODEL_INFO_POLL_MS);
renderBudgetUnavailable("입력 또는 설정을 바꾸면 토큰 사용량을 계산합니다.");
