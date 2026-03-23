const fileInput = document.getElementById("file-input");
const browseButton = document.getElementById("browse-button");
const generateButton = document.getElementById("generate-button");
const downloadButton = document.getElementById("download-button");
const dropzone = document.getElementById("dropzone");
const previewImage = document.getElementById("preview-image");
const promptInput = document.getElementById("prompt-input");
const structPreview = document.getElementById("struct-preview");
const floatsPreview = document.getElementById("floats-preview");
const status = document.getElementById("status");
const logOutput = document.getElementById("log-output");

let selectedFile = null;
let zipPayload = null;
let zipFilename = "chatgarment_result.zip";
let generationStartedAt = null;
let generationTimerId = null;

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
    setStatus(`vLLM으로 JSON 생성 중... ${elapsedSeconds}초 경과`);
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
  target.textContent = JSON.stringify(data, null, 2);
}

function resetGeneratedState() {
  zipPayload = null;
  zipFilename = "chatgarment_result.zip";
  downloadButton.disabled = true;
  setPreview(structPreview, {});
  setPreview(floatsPreview, {});
}

function updateImagePreview(file) {
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  dropzone.classList.add("has-image");
}

function setSelectedFile(file) {
  selectedFile = file;
  generateButton.disabled = !file;
  resetGeneratedState();

  if (!file) {
    previewImage.removeAttribute("src");
    dropzone.classList.remove("has-image");
    setStatus("");
    return;
  }

  updateImagePreview(file);
  setStatus(`선택됨: ${file.name}`);
  appendLog(`이미지 선택: ${file.name}`);
}

function decodeBase64ToBlob(base64String) {
  const binary = atob(base64String);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: "application/zip" });
}

async function generate() {
  if (!selectedFile) {
    setStatus("먼저 이미지를 선택하세요.", true);
    return;
  }

  const promptText = promptInput.value.trim();
  if (!promptText) {
    setStatus("프롬프트를 입력하세요.", true);
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("prompt_text", promptText);

  generateButton.disabled = true;
  downloadButton.disabled = true;
  resetGeneratedState();
  clearLog();
  setStatus("vLLM으로 JSON 생성 중...");
  appendLog("생성 요청 시작");
  appendLog(`프롬프트 전송: ${promptText.length}자`);
  appendLog("생성 중입니다. 큰 이미지와 긴 JSON 출력 때문에 수 분 걸릴 수 있습니다.");
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

    setPreview(structPreview, payload.struct);
    setPreview(floatsPreview, payload.floats);
    zipPayload = payload.zip_base64;
    zipFilename = payload.filename || "chatgarment_result.zip";
    downloadButton.disabled = false;
    setStatus("생성이 완료되었습니다. JSON을 확인하고 ZIP을 다운로드하세요.");
    appendLog("생성 완료");
  } catch (error) {
    const message = error instanceof Error ? error.message : "생성에 실패했습니다.";
    setStatus(message, true);
    appendLog(message);
  } finally {
    stopGenerationTimer();
    generateButton.disabled = false;
  }
}

function downloadZip() {
  if (!zipPayload) {
    return;
  }

  const blob = decodeBase64ToBlob(zipPayload);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = zipFilename;
  link.click();
  URL.revokeObjectURL(url);
  appendLog(`다운로드 완료: ${zipFilename}`);
}

async function loadPrompt() {
  try {
    const response = await fetch("/api/prompt");
    const payload = await response.json();
    if (!response.ok || typeof payload.prompt_text !== "string") {
      throw new Error("기본 프롬프트를 불러오지 못했습니다.");
    }
    promptInput.value = payload.prompt_text;
    promptInput.placeholder = "";
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "기본 프롬프트를 불러오지 못했습니다.";
    promptInput.placeholder = message;
    setStatus(message, true);
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
downloadButton.addEventListener("click", downloadZip);

loadPrompt();
