
// This is the script for 6.s198 Gan Assignment.

// Hyperparameters to experiment with
const BATCH_SIZE = 35;  // number of inputs per batch
const NUM_BATCHES = 25000; // Number of batches to run
const BATCH_INTERVAL = 100; // Print training results every BATCH_INTERVAL number of batches
const DO_TESTING = false; // Flag to control whether or not to do testing
const NUM_IMAGES_TO_TEST = 10;

// Global variable that can be examined using the Javascript console
var IMAGES_TO_EXAMINE = []

// Prepare the data
console.log('Starting script');
console.log('Importing and initializing the dataset');

const xhrDatasetConfigs = {
  MNIST: {
    data: [{
      name: 'images',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
      dataType: 'png',
      shape: [28, 28, 1],
    }, {
      name: 'labels',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
      dataType: 'uint8',
      shape: [10],
    }],
  },

  Fashion_MNIST: {
    data: [{
      name: 'images',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png',
      dataType: 'png',
      shape: [28, 28, 1],
    }, {
      name: 'labels',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8',
      dataType: 'uint8',
      shape: [10],
    }],
  },

  CIFAR_10: {
    "data": [{
      "name": "images",
      "path": "https://storage.googleapis.com/learnjs-data/model-builder/cifar10_images.png",
      "dataType": "png",
      "shape": [32, 32, 3]
    }, {
      "name": "labels",
      "path": "https://storage.googleapis.com/learnjs-data/model-builder/cifar10_labels_uint8",
      "dataType": "uint8",
      "shape": [10]
    }],
    },
}

const dataSets = {};

function populateDatasets() {
  for(const datasetName in xhrDatasetConfigs) {
    if (xhrDatasetConfigs.hasOwnProperty(datasetName)) {
      dataSets[datasetName] = new dl.XhrDataset(xhrDatasetConfigs[datasetName]);
    }
  }
}

populateDatasets();

// To change between MNIST and CFAR, change the definitions
// of dataSet and showColor by commenting and uncommenting the 
// lines below, and reload the page

// use these two lines for MNIST
// const dataSet = dataSets.MNIST;
// const showColor = false;

// use these two lines for Fashion MNIST
// const dataSet = dataSets.Fashion_MNIST;
// const showColor = false;

// use these two lines for CIFAR
const dataSet = dataSets.CIFAR_10;
const showColor = true;


// Functions for setting up the training data and the test data

const TRAIN_TEST_RATIO = 5 / 6;

function getTrainingData() {
  const [images, labels] = dataSet.getData();
  const end = Math.floor(TRAIN_TEST_RATIO * images.length);
  const newLabels = smoothLabels(labels);
  return [images.slice(0, end), newLabels.slice(0, end)];
}

function getTestData() {
  const data = dataSet.getData();
  if (data == null) { return null; }
  const [images, labels] = dataSet.getData();
  const start = Math.floor(TRAIN_TEST_RATIO * images.length);
  const newLabels = smoothLabels(labels);
  return [images.slice(start), newLabels.slice(start)];
}

console.log('DataSet initialized. Begin training when ready...')

function smoothLabels(labels) {
  newLabels = [];
  for (var i = 0; i < labels.length; i++) {
    if (labels[i] > 1/2) {
      newLabels.push(Math.random()*(.5)+.7);
    } else {
      newLabels.push(Math.random()*.3);
    }
  }
  return newLabels;
}
// Procedures to construct networks by adding layers to existing networks

// Construct a flatten layer
function addFlattenLayer(graph, previousLayers) {
  return previousLayers.map(previousLayer =>
    graph.reshape(previousLayer, [dl.util.sizeFromShape(previousLayer.shape)]));
}

// Construct a convolutional layer with specified field size, stride, zero padding, and output depth
function addConvLayer(graph, previousLayers, name, fieldSize, stride, zeroPad, outputDepth) {
  const inputShape = previousLayers[0].shape;
  const wShape = [fieldSize, fieldSize, inputShape[2], outputDepth];

  const w = dl.Array4D.randTruncatedNormal(wShape, 0, 0.1);
  const b = dl.Array1D.zeros([outputDepth]);

  const wTensor = graph.variable(`${name}-weights`, w);
  const bTensor = graph.variable(`${name}-bias`, b);

  return previousLayers.map(previousLayer =>
    graph.conv2d(previousLayer, wTensor, bTensor, fieldSize, outputDepth, stride, zeroPad));
}

// Construct a fully connected layer with a specified number of hidden units
function addFcLayer(graph, previousLayers, name, hiddenUnits) {
  const inputShape = previousLayers[0].shape;
  const inputSize = dl.util.sizeFromShape(inputShape);
  const wShape = [inputSize, hiddenUnits];

  const w = dl.Array2D.randTruncatedNormal(wShape, 0, 0.1);
  const b = dl.Array1D.zeros([hiddenUnits]);

  const wTensor = graph.variable(`${name}-weights`, w);
  const bTensor = graph.variable(`${name}-bias`, b);

  return previousLayers.map(previousLayer => graph.add(graph.matmul(previousLayer, wTensor), bTensor));
}

// Construct a max pool layer with specified field size, stide, and zero padding
function addMaxPoolLayer(graph, previousLayers, fieldSize, stride, zeroPad){
  return previousLayers.map(previousLayer =>
     graph.maxPool(previousLayer, fieldSize, stride, zeroPad));
}

// Construct a Rectified Linear Unit layer
function addReluLayer(graph, previousLayers){
  return previousLayers.map(previousLayer => graph.relu(previousLayer));
}

// Helper procedure for viewing the images
// These will appear on the HTML page
// The images have very low resolution, so this does not produce good views
function showimage(num) {
  const canvas = document.createElement("canvas");
  canvas.setAttribute('height', 32);
  canvas.setAttribute('width', 32);
  canvas.setAttribute('style', 'width: 150px; height: 150px;');
  document.getElementById("imageHolder").appendChild(canvas);
  renderToCanvas(IMAGES_TO_EXAMINE[num], canvas);
}

function renderToCanvas(a, canvas) {
  const [height, width, depth] = a.shape;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = a.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * depth;
    if (showColor) {
      imageData.data[j + 0] = Math.round(255 * data[k + 0]);
      imageData.data[j + 1] = Math.round(255 * data[k + 1]);
      imageData.data[j + 2] = Math.round(255 * data[k + 2]);
      imageData.data[j + 3] = 255;
    } else {
      const pixel = Math.round(255 * data[k]);
      imageData.data[j+0] = pixel;
      imageData.data[j+1] = pixel;
      imageData.data[j+2] = pixel;
      imageData.data[j+3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }
}


// Helper procedures for other random things
function indexOfMax(a) {
  return a.indexOf(a.reduce((max, value) => {return Math.max(max, value)}));
}
			    
function labelTag(n) {
  if (showColor) {
    return CIFAR_10_LABELS[n];
  } else {
    return n;
  }
}

function decimalToPercent(num) {
  return Math.floor(num * 100).toString() + '%';
}

function getOneInputProvider() {
  return {
    getNextCopy(math) {
      return dl.Array1D.new([0, 1]);
    },
    disposeCopy(math, copy) {
    }
  }
}

function getRandomInputProvider(shape) {
  return {
    getNextCopy(math) {
      return dl.NDArray.randNormal(shape);
    },
    disposeCopy(math, copy) {
    }
  }
}

function getZeroInputProvider() {
  return {
    getNextCopy(math) {
      return dl.Array1D.new([1, 0]);
    },
    disposeCopy(math, copy) {
    }
  }
}


// Here is the function that builds the networks and does the training and testing.
// Note that the variables defined in the body of runModel is
// local to the function activation, so you cannot directly examine them in the console.

// Fully connected GAN model for assignment 5
function runGanModel() {
  console.log('Building the model');

  // graph
  const graph = new dl.Graph();

  // inputs
  const inputShape = dataSet.getDataShape(0);
  const randomShape = [100]
  const randomInput = graph.placeholder('random', randomShape);

  // the real input images come from the input dataSet
  const realInput = graph.placeholder('input', inputShape);
  const oneTensor = graph.placeholder('one', [2]);
  const zeroTensor = graph.placeholder('zero', [2]);

  // The fake input images are produced by passing random inputs
  // through the generator network.   This is a very simple generator
  // consisting of a single FC layer followed by a ReLU layer.
  // TODO: Code a better generator here instead.
  let fakeInput;
  const [gFcLayer1] = addFcLayer(graph, [randomInput], 'g1', 784);
  const [grelu1] = addReluLayer(graph, [gFcLayer1]);
  const [gFcLayer2] = addFcLayer(graph, [grelu1], "g2", 3072);
  // const [grelu2] = addReluLayer(graph, [gFcLayer2]);
  const gReshapeLayer = graph.reshape(gFcLayer2, inputShape);
  const [gconv1] = addConvLayer(graph, [gReshapeLayer], "g3", 3, 1, 1, 3);
  const [grelu3] = addReluLayer(graph, [gconv1]);
  
  // const [grelu3] = addReluLayer(graph, [gconv1]);
  const [gconv2] = addConvLayer(graph, [grelu3], "g4", 3, 1, 1, 3);
  // const [grelu4] = addReluLayer(graph, [gconv2]);
  // const [gconv3] = addConvLayer(graph, [grelu4], "g5", 3, 1, 1, 3);

  fakeInput = graph.tanh(gconv2);

  // This is a very simple 1-layer discriminator network.
  // The input to the discriminator flatten layer has two parts: fakeInput and realInput.
  // This will produce a corresponding output with two parts: 
  // fakePrediction are the results for the fake inputs
  // realPrediction are the results for the real inputs
  // TODO: Code a better discriminator here.
  const dconv1 = addConvLayer(graph, [fakeInput, realInput], "dc1", 3,1,1,3);
  const drelu1 = addReluLayer(graph, dconv1);
  // const dmax = addMaxPoolLayer(graph, dconv1, 2,2,0);
  // const dconv2 = addConvLayer(graph, drelu1, "dc2", 4,1,1,1);
  // const dReluLayer2 = addReluLayer(graph, dconv2);
  const dFlattenLayer = addFlattenLayer(graph, drelu1);
  const dFcLayer1 = addFcLayer(graph, dFlattenLayer, 'd1', 128);
  const dReluLayer1 = addReluLayer(graph, dFcLayer1);
  const [fakePrediction, realPrediction] = addFcLayer(graph, dReluLayer1, 'd2', 2);
  // const dconv2 = addConvLayer(graph, dconv1, "dc2", 3,1,1,3);
  // const drelu1 = addReluLayer(graph, dconv1);
  // const dconv3 = addConvLayer(graph, drelu1, "d3", 3,1,1,1);
  // const drelu2 = addReluLayer(graph, dconv3);
  // const dmax = addMaxPoolLayer(graph, dconv3, 2,2,0);
  

  // Cost functions
  // This cost computation is the heart of what makes the GAN work.
  // TOTO: Explain this computation in your assignment writeup.
  const dCostFake = graph.softmaxCrossEntropyCost(fakePrediction, zeroTensor);
  const dCostReal = graph.softmaxCrossEntropyCost(realPrediction, oneTensor);
  const dCostTensor = graph.add(dCostReal, dCostFake);
  const gCostTensor = graph.softmaxCrossEntropyCost(fakePrediction, oneTensor);

  // Optimizers
  var learningRate = 0.00095;
  const gOptimizerNodes = graph.getNodes().filter((node) => node.name.startsWith('g'));
  var gOptimizer = new dl.SGDOptimizer(1.01* learningRate, gOptimizerNodes);
  const dOptimizerNodes = graph.getNodes().filter((node) => node.name.startsWith('d'));
  var dOptimizer = new dl.SGDOptimizer(learningRate, dOptimizerNodes);

  function recalculateGOptimizers(number) {
    learningRate = 0.00095*Math.pow(.95, number);
    gOptimizer = new dl.SGDOptimizer(1.01* learningRate, gOptimizerNodes);
    return gOptimizer;
  }
  function recalculateDOptimizers(number){
    learningRate = 0.00095*Math.pow(.95, number);
    dOptimizer = new dl.SGDOptimizer(learningRate, dOptimizerNodes);
    return dOptimizer;
  }
  // shuffled input of real data
  const data = getTrainingData();
  const shuffledInputProviderGenerator =
    new dl.InCPUMemoryShuffledInputProviderBuilder([data[0]]);  // note that for GANs we don't care about labels
  const [realProvider] =
    shuffledInputProviderGenerator.getInputProviders();

  // Feeds
  const dTrainFeeds = [
    {tensor: realInput, data: realProvider},
    {tensor: randomInput, data: getRandomInputProvider(randomShape)},
    {tensor: oneTensor, data: getOneInputProvider()},
    {tensor: zeroTensor, data: getZeroInputProvider()}
  ]
  const gTrainFeeds = [
    {tensor: randomInput, data: getRandomInputProvider(randomShape)},
    {tensor: oneTensor, data: getOneInputProvider()},
    {tensor: zeroTensor, data: getZeroInputProvider()}
  ]

  console.log('Model built');

  // train
  const math = dl.ENV.math;
  const session = new dl.Session(graph, math);
  IMAGES_TO_EXAMINE = [];  // clear images
  /*
  // create arrays for dCost and gCost
  var dCostVec = JSON.parse(localStorage.getItem("myGStorage"));
  var gCostVec = JSON.parse(localStorage.getItem("myDStorage"));
  var yVec = [];
  for (var i = 0; i < dCostVec.length; i++) {
    yVec.push(i);
  }
  // plot the graph
  var TESTER = document.getElementById('tester');

  var trace1 = {
    y: dCostVec,
    x: yVec,
    mode: 'lines+markers',
    name: 'Discriminator cost' 
  };
  var trace2 = {
    y: gCostVec,
    x: yVec,
    mode: 'lines+markers',
    name: 'Generator cost' 
  };
  var pData = [trace1, trace2];
  var layout = {
    title:'Cost models of discriminator and generator'
  };
  Plotly.plot( TESTER, pData, layout);
  */
  math.scope(async () => {
    console.log('begin training ', NUM_BATCHES, ' batches, will print progress every ', BATCH_INTERVAL, ' batches');
    trainStart = performance.now();
    for (let i = 0; i < NUM_BATCHES; i += 1) {
      if (i%1000 === 0 && i > 1) {
        dOptimizer = recalculateDOptimizers(i%1000);
        gOptimizer = recalculateGOptimizers(i%1000);
      }
      // The training costs are determined by dCostTensor and gCostTensor
      const dCost = session.train(
        dCostTensor, dTrainFeeds, 2*BATCH_SIZE, dOptimizer, dl.CostReduction.MEAN);
      const gCost = session.train(
        gCostTensor, gTrainFeeds, BATCH_SIZE, gOptimizer, dl.CostReduction.MEAN);

      // Compute the cost (by calling get), which requires transferring data
      // from the GPU.
      // save the cost for later examination
      // print a message and image every N batches
      dCostVec.push(dCost.get());
      gCostVec.push(gCost.get());
      if (!(i%BATCH_INTERVAL)){
        console.log(`batch ${i}, cost --- Generator: ${gCost.get()}; Discriminator ${dCost.get()}`);

        // add these values to the cost arrays 
        
        console.log('generating image to examine...')
        IMAGES_TO_EXAMINE.push(session.eval(fakeInput, gTrainFeeds));
        //show the image
        await showimage(IMAGES_TO_EXAMINE.length-1);
        await dl.nextFrame();
        console.log('image generated. continuing with training...')
      }
    }
    trainEnd = performance.now();
    console.log('training complete');
    console.log('training time:', Math.round(trainEnd - trainStart), 'milliseconds');

    // print full cost arrays
    console.log('generator cost finals', gCostVec);
    console.log('discriminator cost finals', dCostVec);
    
    // localStorage.setItem('myGStorage', JSON.stringify(gCostVec));
    // localStorage.setItem('myDStorage', JSON.stringify(dCostVec));
    IMAGES_TO_EXAMINE = [];  // clear images before last 10 examples post-training

    // display 10 final examples from the finished GAN
    const FINAL_EXAMPLE_COUNT = 10
    console.log(`generating ${FINAL_EXAMPLE_COUNT} final images to examine...`)
    for (let i = 0; i < FINAL_EXAMPLE_COUNT; i++) {  
      IMAGES_TO_EXAMINE.push(session.eval(fakeInput, gTrainFeeds))
    }
    for (let i = 0; i < IMAGES_TO_EXAMINE.length; i++) {
      showimage(i);
    }
    console.log('done')
    
  });
}

function runModel(model) {
  dataSet.fetchData().then(() => {
    dataSet.normalizeWithinBounds(0 /* 0 means normalize only images, not labels */, -1, 1);
    runGanModel();
  });
}
