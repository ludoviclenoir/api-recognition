const video = document.getElementById("video");
const videoContainer = document.getElementById("video-container");
const logs = document.getElementById("logs");
const MODEL_PATH = 'https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1';
// bypass cors const MODEL_PATH = 'https://cors-anywhere.herokuapp.com/https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1';

log("Start loading the model");
let model;

// Load the TensorFlow model
tf.loadGraphModel(MODEL_PATH).then((loadedModel) => {
  model = loadedModel;
  log("Model loaded.");
  playVideo();
}).catch((err) => {
  console.log(err);
});

function playVideo() {
  if (!navigator.mediaDevices) {
    console.error("mediaDevices not supported");
    return;
  }
  navigator.mediaDevices
    .getUserMedia({
      video: {
        width: { min: 640, ideal: 1280, max: 1920 },
        height: { min: 360, ideal: 720, max: 1080 },
      },
      audio: false,
    })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err) {
      console.log(err);
    });
}

video.addEventListener("play", () => {
  log("The video is starting to play.");
  detectObjects();
});

function detectObjects() {
  const canvas = document.createElement("canvas");
  videoContainer.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  // Adjust these as necessary
  canvas.width = 640;
  canvas.height = 480;

  const processFrame = async () => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const tfImg = tf.browser.fromPixels(canvas);
    const smallImg = tf.image.resizeBilinear(tfImg, [300, 300]); // size should match the one expected by the model
    const resized = tf.cast(smallImg, 'float32');
    const predictions = await model.executeAsync(resized);

    // Render predictions
    renderPredictions(predictions, ctx);

    tfImg.dispose();
    smallImg.dispose();
    resized.dispose();
    
    requestAnimationFrame(processFrame);
  };
  processFrame();
}

function renderPredictions(predictions, ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  // Draw predictions (bounding box, label, score)
  // This assumes predictions are in a format similar to COCO-SSD
  // You'll need to adapt based on how your model outputs predictions
  predictions.forEach(prediction => {
    const [x, y, width, height] = prediction['bbox'];
    ctx.strokeStyle = "#FF0015";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);

    ctx.fillStyle = "#FF0015";
    ctx.fillText(prediction['class'] + ": " + (prediction['score'] * 100).toFixed(2) + "%", x, y);
  });
}

function log(msg) {
  const message = document.createTextNode(msg);
  const li = document.createElement("li");
  li.appendChild(message);
  logs.appendChild(li);
  logs.scrollTop = logs.scrollHeight;
}
