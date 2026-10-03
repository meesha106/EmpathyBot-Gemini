// Frontend chat UI
// Sends text to /api/chat, images to /api/classify, with confirmation step.

const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const messages = document.getElementById("messages");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const imageInput = document.getElementById("image-input");
const imagePreviewArea = document.getElementById("image-preview-area");
const imagePreview = document.getElementById("image-preview");
const imageNameEl = document.getElementById("image-name");
const removeImageBtn = document.getElementById("remove-image-btn");

let history = [];
let currentBotMessageEl = null;
let fullBotReply = "";
let selectedImageFile = null;
let pendingFoodName = null;
let pendingExtraText = null;
let awaitingCorrection = false;

// ── Image picker ──────────────────────────────────────────────────────────────
uploadBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  selectedImageFile = file;
  imagePreview.src = URL.createObjectURL(file);
  imageNameEl.textContent = file.name;
  imagePreviewArea.style.display = "flex";
  input.placeholder = "Add a question about this food, or just press Send...";
  input.removeAttribute("required");
});

removeImageBtn.addEventListener("click", clearImageSelection);

function clearImageSelection() {
  selectedImageFile = null;
  imageInput.value = "";
  imagePreview.src = "";
  imageNameEl.textContent = "";
  imagePreviewArea.style.display = "none";
  input.placeholder = "Ask a food question or upload a photo...";
  input.setAttribute("required", "");
}

// ── Message helpers ───────────────────────────────────────────────────────────
function addMessage(text, who = "bot", isTyping = false) {
  const el = document.createElement("div");
  el.className = "msg " + (who === "user" ? "user" : "bot");
  if (isTyping) {
    el.classList.add("typing");
    el.textContent = "…";
  } else {
    el.textContent = text;
  }
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addImageMessage(file) {
  const el = document.createElement("div");
  el.className = "msg user image-msg";
  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  img.alt = file.name;
  img.className = "msg-image";
  const label = document.createElement("span");
  label.textContent = file.name;
  label.className = "msg-image-label";
  el.appendChild(img);
  el.appendChild(label);
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addConfirmationBubble(foodName, confidence, others) {
  const el = document.createElement("div");
  el.className = "msg bot confirmation-bubble";

  const isLowConfidence = confidence < 85;
  const header = document.createElement("div");
  header.className = "confirm-header";
  header.innerHTML = isLowConfidence
    ? `🤔 I'm not very sure, but this looks like <strong>${foodName}</strong> <span class="confidence">(${confidence.toFixed(1)}%)</span>`
    : `🍽️ I detected <strong>${foodName}</strong> <span class="confidence">(${confidence.toFixed(1)}%)</span>`;
  el.appendChild(header);

  if (others && others.length > 0) {
    const otherEl = document.createElement("div");
    otherEl.className = "other-predictions";
    otherEl.textContent = "Other possibilities: " +
      others.filter(p => p.class !== "unknown")
            .map(p => `${p.class} (${p.confidence.toFixed(1)}%)`)
            .join(", ");
    el.appendChild(otherEl);
  }

  const question = document.createElement("div");
  question.className = "confirm-question";
  question.textContent = "Is this correct?";
  el.appendChild(question);

  const btnRow = document.createElement("div");
  btnRow.className = "confirm-btns";

  const yesBtn = document.createElement("button");
  yesBtn.className = "confirm-btn yes";
  yesBtn.textContent = "✓ Yes, that's right";
  yesBtn.addEventListener("click", () => handleConfirmYes(el, foodName));

  const noBtn = document.createElement("button");
  noBtn.className = "confirm-btn no";
  noBtn.textContent = "✗ No, it's something else";
  noBtn.addEventListener("click", () => handleConfirmNo(el));

  btnRow.appendChild(yesBtn);
  btnRow.appendChild(noBtn);
  el.appendChild(btnRow);

  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addCorrectionPrompt() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.textContent = "No problem! Please type the correct food name below and I'll give you GD advice for it.";
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;

  awaitingCorrection = true;
  input.placeholder = "Type the correct food name...";
  input.removeAttribute("required");
  input.disabled = false;
  sendBtn.disabled = false;
  uploadBtn.disabled = false;
  input.focus();
}

// ── Confirmation handlers ─────────────────────────────────────────────────────
function handleConfirmYes(bubbleEl, foodName) {
  bubbleEl.querySelectorAll(".confirm-btn").forEach(b => b.disabled = true);
  addMessage(`Yes, it's ${foodName}`, "user");
  history.push({ role: "user", text: `Yes, it's ${foodName}` });

  const question = pendingExtraText
    ? `I have ${foodName}. ${pendingExtraText} Is it safe for gestational diabetes?`
    : `I have ${foodName}. Is it safe for gestational diabetes? How much can I eat and what precautions should I take?`;

  pendingFoodName = null;
  pendingExtraText = null;
  streamBotResponse(question);
}

function handleConfirmNo(bubbleEl) {
  bubbleEl.querySelectorAll(".confirm-btn").forEach(b => b.disabled = true);
  pendingFoodName = null;
  addCorrectionPrompt();
}

function setTyping(enabled) {
  if (enabled) {
    if (!messages.querySelector(".typing")) addMessage("…", "bot", true);
  } else {
    const t = messages.querySelector(".typing");
    if (t) t.remove();
  }
}

// ── Stream text response ──────────────────────────────────────────────────────
async function streamFromServer(text) {
  fullBotReply = "";
  currentBotMessageEl = addMessage("", "bot");

  const payload = { message: text, history };
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) throw new Error((await res.text()) || "Server error");
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    currentBotMessageEl.textContent += chunk;
    fullBotReply += chunk;
    messages.scrollTop = messages.scrollHeight;
  }

  if (fullBotReply.includes("---STREAMING ERROR---")) {
    currentBotMessageEl.classList.add("error");
    throw new Error("Server-side streaming error.");
  }

  return fullBotReply;
}

async function streamBotResponse(question) {
  setTyping(true);
  sendBtn.disabled = true;
  input.disabled = true;
  uploadBtn.disabled = true;

  try {
    const reply = await streamFromServer(question);
    history.push({ role: "model", text: reply });
  } catch (err) {
    setTyping(false);
    if (currentBotMessageEl) {
      currentBotMessageEl.textContent = "⚠️ Error: " + (err.message || "Unknown");
      currentBotMessageEl.classList.add("error");
    } else {
      addMessage("⚠️ Error: " + (err.message || "Unknown"), "bot");
    }
  } finally {
    setTyping(false);
    sendBtn.disabled = false;
    input.disabled = false;
    uploadBtn.disabled = false;
    input.placeholder = "Ask a food question or upload a photo...";
    input.setAttribute("required", "");
    awaitingCorrection = false;
    input.focus();
    currentBotMessageEl = null;
    fullBotReply = "";
  }
}

// ── Classify image then show confirmation ─────────────────────────────────────
async function classifyAndConfirm(file, extraText) {
  const formData = new FormData();
  formData.append("image", file);

  const res = await fetch("/api/classify", { method: "POST", body: formData });

  if (!res.ok) throw new Error("Image classification failed");

  const data = await res.json();
  if (!data.success) throw new Error(data.error || "Could not classify image");

  const top = data.top_prediction;
  const others = data.top_predictions.slice(1);

  pendingFoodName = top.class;
  pendingExtraText = extraText || null;

  addConfirmationBubble(top.class, top.confidence, others);
}

// ── Form submit ───────────────────────────────────────────────────────────────
form.addEventListener("submit", async (ev) => {
  ev.preventDefault();

  const textValue = input.value.trim();
  const hasImage = !!selectedImageFile;

  // Correction flow — user typed correct food name after clicking "No"
  if (awaitingCorrection) {
    if (!textValue) return;
    addMessage(textValue, "user");
    history.push({ role: "user", text: textValue });
    input.value = "";
    awaitingCorrection = false;

    const question = pendingExtraText
      ? `I have ${textValue}. ${pendingExtraText} Is it safe for gestational diabetes?`
      : `I have ${textValue}. Is it safe for gestational diabetes? How much can I eat and what precautions should I take?`;

    pendingExtraText = null;
    streamBotResponse(question);
    return;
  }

  if (!textValue && !hasImage) return;

  // Show user messages in chat
  if (hasImage) addImageMessage(selectedImageFile);
  if (textValue) addMessage(textValue, "user");

  // Only add plain text to history — NOT image uploads
  // (image questions get added to history after confirmation)
  if (!hasImage && textValue) {
    history.push({ role: "user", text: textValue });
  }

  const imageFileCopy = selectedImageFile;
  const extraText = textValue;
  input.value = "";
  sendBtn.disabled = true;
  input.disabled = true;
  uploadBtn.disabled = true;
  clearImageSelection();

  try {
    if (imageFileCopy) {
      setTyping(true);
      await classifyAndConfirm(imageFileCopy, extraText);
      setTyping(false);
      // Re-enable UI so user can click Yes / No
      sendBtn.disabled = false;
      input.disabled = false;
      uploadBtn.disabled = false;
    } else {
      setTyping(true);
      const reply = await streamFromServer(textValue);
      history.push({ role: "model", text: reply });
      setTyping(false);
      sendBtn.disabled = false;
      input.disabled = false;
      uploadBtn.disabled = false;
      input.focus();
      currentBotMessageEl = null;
      fullBotReply = "";
    }
  } catch (err) {
    setTyping(false);
    addMessage("⚠️ Error: " + (err.message || "Unknown"), "bot");
    sendBtn.disabled = false;
    input.disabled = false;
    uploadBtn.disabled = false;
  }
});

// Enter to send (Shift+Enter for newline)
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});