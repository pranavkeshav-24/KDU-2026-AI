const THREAD_ID = "local_session_" + Math.random().toString(36).slice(2, 11);
const USER_ID = "local_user";
const chatBox = document.getElementById("chat-box");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");
const btnClear = document.getElementById("btn-clear");
const personaSelect = document.getElementById("persona-select");
const imageUpload = document.getElementById("image-upload");
const imagePreviewContainer = document.getElementById("image-preview-container");
const imagePreview = document.getElementById("image-preview");
const btnRemoveImage = document.getElementById("btn-remove-image");
const statusDot = document.getElementById("status-dot");
const connectionStatus = document.getElementById("connection-status");

let selectedFile = null;

function escapeHtml(content) {
    return content
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function formatMessage(content) {
    return escapeHtml(content)
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

function setConnectionStatus(state, label) {
    statusDot.classList.remove("online", "offline", "warning");
    statusDot.classList.add(state);
    connectionStatus.textContent = label;
}

function resetSelectedImage() {
    selectedFile = null;
    imageUpload.value = "";
    imagePreview.src = "";
    imagePreviewContainer.classList.add("hidden");
}

function appendMessage(isUser, content, imageSrc = null) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${isUser ? "user" : "ai"}`;

    const avatarSvg = isUser
        ? `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
        : `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none"><path d="M12 2a10 10 0 1 0 10 10H12V2Z"></path><path d="M12 2a10 10 0 1 1-10 10h10V2Z"></path></svg>`;

    let innerContent = "";
    if (imageSrc) {
        innerContent += `<img src="${imageSrc}" class="payload-img" alt="Uploaded preview">`;
    }
    innerContent += `<p>${formatMessage(content)}</p>`;

    msgDiv.innerHTML = `
        <div class="avatar">${avatarSvg}</div>
        <div class="bubble">${innerContent}</div>
    `;

    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showLoader() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message ai loader-msg";
    msgDiv.innerHTML = `
        <div class="avatar"><svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none"><path d="M12 2a10 10 0 1 0 10 10H12V2Z"></path></svg></div>
        <div class="bubble"><div class="loader"></div></div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function hideLoader() {
    const loader = document.querySelector(".loader-msg");
    if (loader) {
        loader.remove();
    }
}

btnRemoveImage.onclick = resetSelectedImage;

imageUpload.onchange = (event) => {
    if (event.target.files && event.target.files[0]) {
        selectedFile = event.target.files[0];
        const reader = new FileReader();
        reader.onload = (loadEvent) => {
            imagePreview.src = loadEvent.target.result;
            imagePreviewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(selectedFile);
    }
};

async function sendMessage() {
    const text = chatInput.value.trim();
    const fileToSend = selectedFile;

    if (!text && !fileToSend) {
        return;
    }

    const imagePreviewSrc = fileToSend ? imagePreview.src : null;
    appendMessage(true, text || "Sent an image", imagePreviewSrc);
    chatInput.value = "";
    resetSelectedImage();

    btnSend.disabled = true;
    chatInput.disabled = true;
    showLoader();

    try {
        let response;
        if (fileToSend) {
            const formData = new FormData();
            formData.append("message", text || "Analyze this image.");
            formData.append("thread_id", THREAD_ID);
            formData.append("file", fileToSend);

            response = await fetch("/chat/image", {
                method: "POST",
                headers: {
                    "X-User-Id": USER_ID,
                    "X-Response-Style": personaSelect.value,
                },
                body: formData,
            });
        } else {
            response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-User-Id": USER_ID,
                    "X-Response-Style": personaSelect.value,
                },
                body: JSON.stringify({ message: text, thread_id: THREAD_ID }),
            });
        }

        const data = await response.json().catch(() => ({}));
        hideLoader();

        if (!response.ok) {
            const errorMessage =
                data.content ||
                data.detail ||
                data.error ||
                `Request failed with status ${response.status}.`;
            throw new Error(errorMessage);
        }

        setConnectionStatus("online", "Server ready");

        let aiReply = data.content || "Sorry, I couldn't process that.";
        if (data.data && data.data.temperature !== undefined) {
            aiReply += `\n\nCondition: ${data.data.temperature} deg ${data.data.unit}, ${data.data.condition}`;
        }

        appendMessage(false, aiReply);
    } catch (error) {
        hideLoader();
        setConnectionStatus("warning", "Backend needs attention");
        appendMessage(
            false,
            error.message || "An error occurred connecting to the backend. Is the server running?"
        );
        console.error(error);
    } finally {
        btnSend.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
    }
}

btnSend.onclick = sendMessage;
chatInput.onkeydown = (event) => {
    if (event.key === "Enter") {
        sendMessage();
    }
};

btnClear.onclick = async () => {
    if (!confirm("Clear conversation memory?")) {
        return;
    }

    try {
        const response = await fetch(`/history/${THREAD_ID}`, { method: "DELETE" });
        if (!response.ok) {
            throw new Error("Unable to clear server history right now.");
        }
    } catch (error) {
        setConnectionStatus("warning", "Backend needs attention");
        console.error(error);
    }

    chatBox.innerHTML = "";
    appendMessage(false, "Conversation completely cleared. How can we start fresh?");
};

async function checkServerHealth() {
    try {
        const response = await fetch("/health");
        if (!response.ok) {
            throw new Error("Health check failed.");
        }
        setConnectionStatus("online", "Server ready");
    } catch (error) {
        setConnectionStatus("offline", "Backend unavailable");
    }
}

checkServerHealth();
chatInput.focus();
