console.log("CommuniSign JS Loaded");

const video        = document.getElementById("webcam");
const canvas       = document.getElementById("output-canvas");
const ctx          = canvas.getContext("2d");
const letterEl     = document.getElementById("letter");
const wordEl       = document.getElementById("word");
const confidenceEl = document.getElementById("confidence");
const confBarEl    = document.getElementById("confidence-bar");
const sentenceEl   = document.getElementById("sentence");
const predTextEl   = document.getElementById("prediction-text");
const startBtn     = document.getElementById("startBtn");
const stopBtn      = document.getElementById("stopBtn");
const clearBtn     = document.getElementById("clearBtn");
const spaceBtn     = document.getElementById("spaceBtn");
const statusDot    = document.getElementById("cam-status-dot");

let currentWord   = "";
let sentence      = "";
let lastLetter    = "";
let lastLetterTime = 0;
const LETTER_COOLDOWN = 1500;
let cameraInstance = null;
let isRunning      = false;

// ── MediaPipe ────────────────────────────────────────────────────────────────
const hands = new Hands({
    locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.75,
    minTrackingConfidence: 0.6,
});

hands.onResults((results) => {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#9b5de5", lineWidth: 2 });
        drawLandmarks(ctx,  landmarks,                   { color: "#f15bb5", lineWidth: 1, radius: 3 });
        sendLandmarks(landmarks);
    } else {
        // ← KEY FIX: clear stale buffer when hand leaves frame
        fetch("/clear_buffer", { method: "POST" }).catch(() => {});
        predTextEl.textContent = "No hand detected";
        updateUI({ letter: "-", confidence: 0 });
    }
});

async function sendLandmarks(landmarks) {
    try {
        const flat = landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
        const res  = await fetch("/predict_landmarks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ landmarks: flat }),
        });
        if (res.ok) updateUI(await res.json());
    } catch (_) {}
}

function updateUI({ letter, confidence }) {
    const conf = parseFloat(confidence) || 0;
    confidenceEl.textContent   = conf + "%";
    confBarEl.style.width      = conf + "%";
    confBarEl.style.background = conf >= 80 ? "#00f5d4" : conf >= 50 ? "#fee440" : "#f15bb5";

    if (!letter || letter === "-" || letter === "Warming up...") {
        predTextEl.textContent = letter === "Warming up..." ? "Warming up…" : "Waiting for sign...";
        return;
    }

    letterEl.textContent   = letter;
    predTextEl.textContent = `Detected: ${letter}  (${conf}%)`;

    const now = Date.now();
    if (letter !== lastLetter || (now - lastLetterTime) > LETTER_COOLDOWN) {
        if (letter !== lastLetter) {
            currentWord   += letter;
            lastLetter     = letter;
            lastLetterTime = now;
            wordEl.textContent = currentWord || "-";
        }
    }
}

// ── Controls ──────────────────────────────────────────────────────────────────
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click",  stopCamera);

clearBtn.addEventListener("click", () => {
    currentWord = ""; sentence = ""; lastLetter = "";
    wordEl.textContent = sentenceEl.textContent = "-";
    letterEl.textContent = "-";
    confidenceEl.textContent = "0%";
    confBarEl.style.width = "0%";
    predTextEl.textContent = "Waiting for sign...";
    fetch("/clear_buffer", { method: "POST" }).catch(() => {});
});

spaceBtn.addEventListener("click", () => {
    if (currentWord.trim()) {
        sentence   += (sentence ? " " : "") + currentWord;
        currentWord = ""; lastLetter = "";
        wordEl.textContent     = "-";
        sentenceEl.textContent = sentence;
    }
});

async function startCamera() {
    if (isRunning) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" }, audio: false,
        });
        video.srcObject = stream;
        await video.play();
        console.log("Camera Started");

        cameraInstance = new Camera(video, {
            onFrame: async () => { await hands.send({ image: video }); },
            width: 640, height: 480,
        });
        await cameraInstance.start();

        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled  = false;
        statusDot.style.background = "#00f5d4";
    } catch (err) {
        console.error("Camera error:", err);
        predTextEl.textContent = "Camera access denied.";
    }
}

function stopCamera() {
    if (cameraInstance) { cameraInstance.stop(); cameraInstance = null; }
    if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled  = true;
    statusDot.style.background = "#9b5de5";
    predTextEl.textContent = "Camera stopped.";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    fetch("/clear_buffer", { method: "POST" }).catch(() => {});
}

window.addEventListener("load", startCamera);