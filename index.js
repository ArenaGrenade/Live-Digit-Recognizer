const RELATIVE_MODEL_URL = './models/model.json';
var model;

const status_drawer = document.getElementById("status");
const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext('2d');
const root = document.documentElement;
const fontMin = 16;
const fontMax = 60;

var lmb_active = false;
var rmb_active = false;

async function update_model_preds() {
    status_drawer.textContent = 'Model Predicting...';
    tf.engine().startScope()
    var tensor = await tf.browser.fromPixels(canvas, 4);    
    tensor = tf.cast(tensor, 'float32').div(tf.scalar(255.0))
    if (tensor.max().dataSync()[0] == 0) {
        status_drawer.textContent = 'Ready for drawing';
        for (var i = 0; i < 10; i++) {
            document.getElementById(`number${i}`).style.fontSize = "16px";
            document.getElementById(`number${i}`).style.opacity = "0.5";
        }
        return;
    }
    var model_in = tf.unstack(tensor, 2)[3].expandDims(2);
    model_in = tf.image.resizeBilinear(model_in, [28, 28], true);
    model_out = model.predict(model_in.expandDims(0));
    for (var i = 0; i < 10; i++) {
        document.getElementById(`number${i}`).style.fontSize = `${Math.round(model_out.dataSync()[i] * fontMax + fontMin)}px`;
        document.getElementById(`number${i}`).style.opacity = `${(model_out.dataSync()[i] / model_out.max().dataSync()[0]) * 0.5 + 0.5}`
    }
    status_drawer.textContent = `Model has predicted the number is a ${model_out.argMax(1).dataSync()[0]}`;
    tf.engine().endScope()
}

canvas.addEventListener('contextmenu', function (e) {
    e.preventDefault(); 
  }, false);

canvas.addEventListener("mousedown", function(e) {
    if (e.button == 0) lmb_active = true;
    else if (e.button == 2) rmb_active = true;
});
canvas.addEventListener("mouseup", function(e) {
    if (e.button == 0) lmb_active = false;
    else if (e.button == 2) rmb_active = false;
});

canvas.addEventListener("mousemove", function(e) {
    const cRect = canvas.getBoundingClientRect();
    const chunkX = Math.round((e.clientX - cRect.left) / 10);
    const chunkY = Math.round((e.clientY - cRect.top) / 10);
    if (lmb_active) ctx.fillRect(chunkX * 10, chunkY * 10, 10, 10);
    else if (rmb_active) ctx.clearRect(chunkX * 10 - 10, chunkY * 10 - 10, 20, 20);
    // if (lmb_active || rmb_active) update_model_preds();
});

async function init() {
    try {
        status_drawer.textContent = `Loading model locally from ${RELATIVE_MODEL_URL}...`;
        model = await tf.loadLayersModel(RELATIVE_MODEL_URL);
    } catch (err) {
        status.textContent = `Loading hosted model from ${HOSTED_MODEL_URL} ...`;
        model = await tf.loadLayersModel(HOSTED_MODEL_URL);
    }
    status_drawer.textContent = 'Done loading model. Ready for drawing';
    setInterval(update_model_preds, 1000);
}
  
init();