const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const statusEl = document.getElementById('status');
const startCameraButton = document.getElementById('startCameraButton');
const confidenceInput = document.getElementById('confidence');
const confidenceValue = document.getElementById('confidenceValue');
const maxPosesSelect = document.getElementById('maxPoses');
const architectureSelect = document.getElementById('architecture');
const toggleButton = document.getElementById('toggleButton');
const reloadModelButton = document.getElementById('reloadModelButton');
const fpsEl = document.getElementById('fps');
const keypointCountEl = document.getElementById('keypointCount');

const COLORS = {
  point: '#34d399',
  skeleton: '#38bdf8',
  label: '#e5e7eb'
};

let detector;
let stream;
let cameraStarted = false;
let running = true;
let busy = false;
let confidenceThreshold = Number(confidenceInput.value);
let lastFrameTime = performance.now();

function setStatus(text) {
  statusEl.textContent = text;
}

function setBusy(isBusy) {
  busy = isBusy;
  startCameraButton.disabled = isBusy;
  reloadModelButton.disabled = isBusy;
  toggleButton.disabled = isBusy;
}

function resizeCanvasToVideo() {
  const { videoWidth, videoHeight } = video;
  if (!videoWidth || !videoHeight) {
    return;
  }

  canvas.width = videoWidth;
  canvas.height = videoHeight;
}

function clearOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  fpsEl.textContent = '0';
  keypointCountEl.textContent = '0';
}

function formatCameraError(error) {
  if (!error || !error.name) return error?.message || 'Unknown camera error.';
  if (error.name === 'NotAllowedError') return 'Camera permission was denied. Allow access and try again.';
  if (error.name === 'NotFoundError') return 'No camera device was found.';
  if (error.name === 'NotReadableError') return 'Camera is in use by another application.';
  if (error.name === 'OverconstrainedError') return 'Requested camera settings are unsupported.';
  if (error.name === 'SecurityError') return 'Camera access requires HTTPS or localhost.';
  return error.message || `Camera error: ${error.name}`;
}

function drawKeypoints(keypoints, minConfidence) {
  let visibleCount = 0;

  keypoints.forEach(({ score, position, part }) => {
    if (score < minConfidence) return;

    visibleCount += 1;
    ctx.beginPath();
    ctx.arc(position.x, position.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.point;
    ctx.fill();

    ctx.font = '12px sans-serif';
    ctx.fillStyle = COLORS.label;
    ctx.fillText(part, position.x + 8, position.y + 4);
  });

  return visibleCount;
}

function drawSkeleton(keypoints, minConfidence) {
  posenet.getAdjacentKeyPoints(keypoints, minConfidence).forEach(([from, to]) => {
    ctx.beginPath();
    ctx.moveTo(from.position.x, from.position.y);
    ctx.lineTo(to.position.x, to.position.y);
    ctx.strokeStyle = COLORS.skeleton;
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function drawFrame(poses) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  let totalVisible = 0;
  poses.forEach((pose) => {
    drawSkeleton(pose.keypoints, confidenceThreshold);
    totalVisible += drawKeypoints(pose.keypoints, confidenceThreshold);
  });

  ctx.restore();
  keypointCountEl.textContent = String(totalVisible);
}

function updateFps() {
  const now = performance.now();
  const delta = now - lastFrameTime;
  lastFrameTime = now;
  if (delta > 0) fpsEl.textContent = (1000 / delta).toFixed(1);
}

async function setupCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera access is not supported in this browser.');
  }

  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
  });

  video.srcObject = stream;
  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });

  resizeCanvasToVideo();
}

async function loadPoseNet() {
  if (!window.posenet) {
    throw new Error('PoseNet library failed to load. Check network/CDN access and refresh.');
  }

  setStatus('Loading PoseNet model…');
  detector = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 640, height: 480 },
    multiplier: Number(architectureSelect.value)
  });
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = undefined;
  }

  video.srcObject = null;
  cameraStarted = false;
  running = true;
  toggleButton.textContent = 'Pause Detection';
  startCameraButton.textContent = 'Enable Camera';
  clearOverlay();
  setStatus('Camera disabled. Click "Enable Camera" to start again.');
}

async function renderLoop() {
  if (!cameraStarted || !detector || !running) {
    requestAnimationFrame(renderLoop);
    return;
  }

  try {
    const maxDetections = Number(maxPosesSelect.value);
    const poses =
      maxDetections > 1
        ? await detector.estimateMultiplePoses(video, {
            flipHorizontal: true,
            maxDetections,
            scoreThreshold: confidenceThreshold,
            nmsRadius: 20
          })
        : [await detector.estimateSinglePose(video, { flipHorizontal: true })];

    drawFrame(poses);
    updateFps();
  } catch (error) {
    setStatus(`Detection error: ${error.message}`);
  }

  requestAnimationFrame(renderLoop);
}

startCameraButton.addEventListener('click', async () => {
  if (busy) return;

  if (cameraStarted) {
    stopCamera();
    return;
  }

  setBusy(true);
  setStatus('Requesting camera permission…');

  try {
    await setupCamera();
    await loadPoseNet();
    cameraStarted = true;
    startCameraButton.textContent = 'Disable Camera';
    setStatus('Detecting poses…');
  } catch (error) {
    setStatus(formatCameraError(error));
  } finally {
    setBusy(false);
  }
});

reloadModelButton.addEventListener('click', async () => {
  if (busy) return;

  if (!cameraStarted) {
    setStatus('Enable camera first, then reload the model.');
    return;
  }

  setBusy(true);

  try {
    await loadPoseNet();
    setStatus('Model reloaded. Detecting poses…');
  } catch (error) {
    setStatus(`Error reloading model: ${error.message}`);
  } finally {
    setBusy(false);
  }
});

toggleButton.addEventListener('click', () => {
  if (busy) return;

  if (!cameraStarted) {
    setStatus('Enable camera first to start detection.');
    return;
  }

  running = !running;
  toggleButton.textContent = running ? 'Pause Detection' : 'Resume Detection';
  setStatus(running ? 'Detecting poses…' : 'Detection paused.');

  if (!running) clearOverlay();
});

architectureSelect.addEventListener('change', () => {
  if (cameraStarted) {
    setStatus('Architecture changed. Click "Reload Model" to apply it.');
  }
});

confidenceInput.addEventListener('input', (event) => {
  confidenceThreshold = Number(event.target.value);
  confidenceValue.textContent = confidenceThreshold.toFixed(2);
});

window.addEventListener('resize', resizeCanvasToVideo, { passive: true });
window.addEventListener('pagehide', stopCamera);
renderLoop();
