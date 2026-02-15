# PoseNet Front End

A browser-based PoseNet demo powered by TensorFlow.js with:
- live webcam pose detection
- keypoint + skeleton overlays
- confidence / max-poses controls
- model multiplier switching and reload
- runtime FPS + visible keypoint metrics

## Run locally

Because this app uses camera APIs, run it from a secure context (`localhost` is allowed):

```bash
python3 -m http.server 4173
```

Then open `http://localhost:4173` and use:
- **Enable Camera** / **Disable Camera** to start or stop webcam access
- **Pause Detection** / **Resume Detection** to control pose inference
- **Reload Model** to reload PoseNet with the selected architecture
