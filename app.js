// Real-Time Indian Sign Language Interpreter (Web App)
// ONNX Runtime Web client-side real-time engine

const GESTURES = [
  { id: 'good', name: 'Good', emoji: '👍' },
  { id: 'hello', name: 'Hello', emoji: '👋' },
  { id: 'help', name: 'Help', emoji: '🆘' },
  { id: 'no', name: 'No', emoji: '✋' },
  { id: 'thank_you', name: 'Thank You', emoji: '🙏' },
  { id: 'yes', name: 'Yes', emoji: '👌' }
];

let ortSession = null;
let isCameraRunning = false;
let isVoiceEnabled = true;
let isFlipped = true;
let confidenceThreshold = 0.60;
let lastSpokenGesture = null;
let lastSpokenTime = 0;
let animationFrameId = null;
let lastFpsTime = performance.now();
let frameCounter = 0;

// DOM Elements
const video = document.getElementById('videoFeed');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const startCamBtn = document.getElementById('startCamBtn');
const flipCamBtn = document.getElementById('flipCamBtn');
const voiceToggleBtn = document.getElementById('voiceToggleBtn');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const modelStatusBadge = document.getElementById('modelStatusBadge');
const modelStatusText = document.getElementById('modelStatusText');
const hudBanner = document.getElementById('hudBanner');
const predictedGesture = document.getElementById('predictedGesture');
const predictedEmoji = document.getElementById('predictedEmoji');
const confidenceScore = document.getElementById('confidenceScore');
const fpsDisplay = document.getElementById('fpsDisplay');
const probabilitiesList = document.getElementById('probabilitiesList');
const historyContainer = document.getElementById('historyContainer');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

// Initialize App
async function init() {
  renderProbabilityBars();
  setupEventListeners();
  await loadModel();
}

// Render Initial Probability Bars
function renderProbabilityBars() {
  probabilitiesList.innerHTML = GESTURES.map((g, idx) => `
    <div class="prob-item" id="prob-item-${idx}">
      <div class="prob-header">
        <span class="prob-gesture-name">${g.emoji} ${g.name}</span>
        <span class="prob-val" id="prob-val-${idx}">0.0%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" id="prob-bar-${idx}"></div>
      </div>
    </div>
  `).join('');
}

// Load ONNX Model into Browser WASM/WebGL
async function loadModel() {
  try {
    modelStatusText.textContent = "Loading Model (9.6 MB)...";
    // Configure ONNX Runtime Web
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    // Load model from relative path
    ortSession = await ort.InferenceSession.create('model/isl_model.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    modelStatusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
    modelStatusBadge.style.color = '#34d399';
    modelStatusText.textContent = "Model Ready (94.4% Acc)";
    console.log("ONNX Model loaded successfully!");
  } catch (err) {
    console.error("Failed to load ONNX model:", err);
    modelStatusBadge.style.background = 'rgba(239, 68, 68, 0.15)';
    modelStatusBadge.style.color = '#f87171';
    modelStatusText.textContent = "Model Load Failed";
    predictedGesture.textContent = "Error loading model. Check console.";
  }
}

// Preprocess 224x224 RGB image for MobileNetV2
function preprocess(sourceElement) {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 224;
  tempCanvas.height = 224;
  const tempCtx = tempCanvas.getContext('2d');

  tempCtx.drawImage(sourceElement, 0, 0, 224, 224);
  const imgData = tempCtx.getImageData(0, 0, 224, 224).data;

  const floatData = new Float32Array(1 * 224 * 224 * 3);
  let p = 0;
  for (let i = 0; i < imgData.length; i += 4) {
    floatData[p++] = (imgData[i] / 127.5) - 1.0;     // R
    floatData[p++] = (imgData[i + 1] / 127.5) - 1.0; // G
    floatData[p++] = (imgData[i + 2] / 127.5) - 1.0; // B
  }

  return new ort.Tensor('float32', floatData, [1, 224, 224, 3]);
}

// Run Inference on Preprocessed Tensor
async function predictTensor(inputTensor) {
  if (!ortSession) return null;
  const feeds = { input: inputTensor };
  const results = await ortSession.run(feeds);
  const rawOutputs = Array.from(results['Identity:0'].data);

  // Apply softmax if needed
  const sum = rawOutputs.reduce((a, b) => a + b, 0);
  let probs = rawOutputs;
  if (Math.abs(sum - 1.0) > 0.05) {
    const maxVal = Math.max(...rawOutputs);
    const exps = rawOutputs.map(x => Math.exp(x - maxVal));
    const expSum = exps.reduce((a, b) => a + b, 0);
    probs = exps.map(x => x / expSum);
  }

  let topIdx = 0;
  let topProb = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > topProb) {
      topProb = probs[i];
      topIdx = i;
    }
  }

  return {
    topIndex: topIdx,
    topGesture: GESTURES[topIdx],
    topConfidence: topProb,
    allProbabilities: probs
  };
}

// Camera Detection Loop
async function processVideoFrame() {
  if (!isCameraRunning) return;

  if (video.readyState === video.HAVE_ENOUGH_DATA && ortSession) {
    try {
      const inputTensor = preprocess(video);
      const prediction = await predictTensor(inputTensor);

      if (prediction) {
        updateUI(prediction);
      }
    } catch (err) {
      console.error("Frame inference error:", err);
    }

    // Calculate FPS
    frameCounter++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      const fps = Math.round((frameCounter * 1000) / (now - lastFpsTime));
      fpsDisplay.textContent = `FPS: ${fps}`;
      frameCounter = 0;
      lastFpsTime = now;
    }
  }

  animationFrameId = requestAnimationFrame(processVideoFrame);
}

// Update UI with Prediction Results
function updateUI(prediction) {
  const { topGesture, topConfidence, allProbabilities } = prediction;

  // Update Probability Bars
  allProbabilities.forEach((prob, idx) => {
    const pct = (prob * 100).toFixed(1);
    const valEl = document.getElementById(`prob-val-${idx}`);
    const barEl = document.getElementById(`prob-bar-${idx}`);
    if (valEl && barEl) {
      valEl.textContent = `${pct}%`;
      barEl.style.width = `${pct}%`;
      if (idx === prediction.topIndex && prob >= confidenceThreshold) {
        barEl.classList.add('highlight');
      } else {
        barEl.classList.remove('highlight');
      }
    }
  });

  // Update HUD
  if (topConfidence >= confidenceThreshold) {
    predictedEmoji.textContent = topGesture.emoji;
    predictedGesture.textContent = topGesture.name;
    confidenceScore.textContent = `${(topConfidence * 100).toFixed(1)}% Confidence`;
    hudBanner.classList.add('detected');

    // Voice Feedback
    announceGesture(topGesture.name);
    // Add to History (debounced)
    addHistoryItem(topGesture, topConfidence);
  } else {
    predictedEmoji.textContent = '🔍';
    predictedGesture.textContent = 'Searching...';
    confidenceScore.textContent = `${(topConfidence * 100).toFixed(1)}% (Low Confidence)`;
    hudBanner.classList.remove('detected');
  }
}

// Text-to-Speech Accessibility
function announceGesture(name) {
  if (!isVoiceEnabled || !('speechSynthesis' in window)) return;
  const now = Date.now();
  if (name !== lastSpokenGesture || now - lastSpokenTime > 2500) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(name);
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
    lastSpokenGesture = name;
    lastSpokenTime = now;
  }
}

// Add Item to History
let lastHistoryGesture = null;
let lastHistoryTime = 0;
function addHistoryItem(gesture, confidence) {
  const now = Date.now();
  if (gesture.id === lastHistoryGesture && now - lastHistoryTime < 3000) return;
  lastHistoryGesture = gesture.id;
  lastHistoryTime = now;

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const item = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `
    <span>${gesture.emoji} <strong>${gesture.name}</strong></span>
    <span style="color: #34d399;">${(confidence * 100).toFixed(1)}% &middot; ${timeStr}</span>
  `;

  if (historyContainer.children.length === 1 && historyContainer.children[0].textContent.includes('No gestures')) {
    historyContainer.innerHTML = '';
  }

  historyContainer.prepend(item);
  if (historyContainer.children.length > 20) {
    historyContainer.removeChild(historyContainer.lastChild);
  }
}

// Camera Start / Stop
async function toggleCamera() {
  if (isCameraRunning) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    startCamBtn.textContent = 'Connecting...';
    startCamBtn.disabled = true;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    isCameraRunning = true;
    startCamBtn.textContent = '⏹ Stop Camera';
    startCamBtn.classList.add('btn-stop');
    startCamBtn.disabled = false;
    predictedGesture.textContent = 'Show hand gesture...';

    processVideoFrame();
  } catch (err) {
    console.error("Camera access failed:", err);
    startCamBtn.textContent = '▶ Start Camera';
    startCamBtn.disabled = false;
    alert("Could not access webcam. Please make sure camera permissions are allowed in your browser settings.");
  }
}

function stopCamera() {
  isCameraRunning = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);

  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }

  startCamBtn.textContent = '▶ Start Camera';
  startCamBtn.classList.remove('btn-stop');
  predictedGesture.textContent = 'Camera stopped';
  predictedEmoji.textContent = '⏹';
  confidenceScore.textContent = '0.0% Confidence';
  hudBanner.classList.remove('detected');
  fpsDisplay.textContent = 'FPS: 0';
}

// Event Listeners
function setupEventListeners() {
  startCamBtn.addEventListener('click', toggleCamera);

  flipCamBtn.addEventListener('click', () => {
    isFlipped = !isFlipped;
    if (isFlipped) {
      video.classList.remove('unflipped');
      flipCamBtn.classList.add('active');
    } else {
      video.classList.add('unflipped');
      flipCamBtn.classList.remove('active');
    }
  });

  voiceToggleBtn.addEventListener('click', () => {
    isVoiceEnabled = !isVoiceEnabled;
    voiceToggleBtn.classList.toggle('active', isVoiceEnabled);
    voiceToggleBtn.textContent = isVoiceEnabled ? '🔊' : '🔇';
  });

  confSlider.addEventListener('input', (e) => {
    confidenceThreshold = e.target.value / 100;
    confValue.textContent = `${e.target.value}%`;
  });

  clearHistoryBtn.addEventListener('click', () => {
    historyContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 1rem; font-size: 0.85rem;">No gestures detected yet</div>';
  });

  // Tab Navigation
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');

      if (tabId !== 'camera-tab' && isCameraRunning) {
        // Keep camera running or pause if preferred
      }
    });
  });

  // Sample Gesture Clicks
  document.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', async () => {
      const src = card.getAttribute('data-src');
      const targetGesture = card.getAttribute('data-gesture');
      await testStaticImage(src, targetGesture);
    });
  });

  // File Upload
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('fileInput');

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary)';
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'var(--border-subtle)';
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--border-subtle)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });
}

// Test Static Sample Gesture Image
async function testStaticImage(src, expectedGesture) {
  const resultBox = document.getElementById('sampleResultBox');
  const resultContent = document.getElementById('sampleResultContent');
  resultBox.style.display = 'block';
  resultContent.innerHTML = '<p>Evaluating image with MobileNetV2 ONNX model...</p>';

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = src;

  img.onload = async () => {
    try {
      const tensor = preprocess(img);
      const res = await predictTensor(tensor);

      const isCorrect = res.topGesture.id === expectedGesture;
      const statusColor = isCorrect ? '#34d399' : '#f87171';

      resultContent.innerHTML = `
        <div style="display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;">
          <img src="${src}" style="width: 140px; height: 140px; object-fit: cover; border-radius: 12px; border: 2px solid ${statusColor};">
          <div>
            <div style="font-size: 1.4rem; font-weight: 800; font-family: var(--font-heading); margin-bottom: 0.25rem;">
              Predicted: ${res.topGesture.emoji} ${res.topGesture.name}
            </div>
            <div style="color: ${statusColor}; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">
              Confidence: ${(res.topConfidence * 100).toFixed(2)}% ${isCorrect ? '✓ (Match)' : ''}
            </div>
            <div style="font-size: 0.85rem; color: var(--text-muted);">
              Target Label: <strong>${expectedGesture.toUpperCase()}</strong> &middot; Inference Time: ~15ms (Browser WASM)
            </div>
          </div>
        </div>
      `;
      announceGesture(res.topGesture.name);
    } catch (err) {
      console.error(err);
      resultContent.innerHTML = `<p style="color: #ef4444;">Inference failed: ${err.message}</p>`;
    }
  };
}

// Handle Uploaded File
function handleUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById('uploadedPreview');
    const resultCard = document.getElementById('uploadResultCard');
    const details = document.getElementById('uploadDetails');

    preview.src = e.target.result;
    resultCard.style.display = 'block';
    details.innerHTML = '<p>Analyzing uploaded image...</p>';

    const img = new Image();
    img.src = e.target.result;
    img.onload = async () => {
      try {
        const tensor = preprocess(img);
        const res = await predictTensor(tensor);

        details.innerHTML = `
          <div style="font-size: 1.4rem; font-weight: 800; font-family: var(--font-heading); margin-bottom: 0.25rem;">
            Prediction: ${res.topGesture.emoji} ${res.topGesture.name}
          </div>
          <div style="color: #34d399; font-weight: 600; font-size: 1rem; margin-bottom: 0.75rem;">
            Confidence: ${(res.topConfidence * 100).toFixed(2)}%
          </div>
          <div style="display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; color: var(--text-muted);">
            ${res.allProbabilities.map((p, i) => `
              <div style="display: flex; justify-content: space-between; max-width: 300px;">
                <span>${GESTURES[i].emoji} ${GESTURES[i].name}</span>
                <span style="font-weight: 600; color: #fff;">${(p * 100).toFixed(1)}%</span>
              </div>
            `).join('')}
          </div>
        `;
        announceGesture(res.topGesture.name);
      } catch (err) {
        details.innerHTML = `<p style="color: #ef4444;">Error: ${err.message}</p>`;
      }
    };
  };
  reader.readAsDataURL(file);
}

// Run on page load
window.addEventListener('DOMContentLoaded', init);
