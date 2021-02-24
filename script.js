//
async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const carsData = await carsDataResponse.json()
    const cleaned = carsData
        .map(car => ({mpg: car.Miles_per_Gallon, horsepower: car.Horsepower}))
        .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned
}

// to define model architecture which functions the model will run while it is
// executing
function createModel() {
    //creates the model
    const model = tf.sequential()
    // adds a single INPUT layer input shape is one because we have one input
    // (horsepower of given car) units is the weighting. (weight is mulitples its
    // inputs by a matrix) by setting to 1, there will be 1 weight for each input
    // features of the data
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}))
    //adds an OUTPUT layer set to one, because we want 1 number output
    model.add(tf.layers.dense({units: 1, unitBias: true}))
    return model
}

//creates the visalised graph with the car data
async function run() {
    const data = await getData()
    const values = data.map(d => ({x: d.horsepower, y: d.mpg}))

    tfvis
        .render
        .scatterplot({
            name: 'Horsepower v MPG'
        }, {
            values
        }, {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        })
    //creating an instance
    const model = createModel();
    //create a model and show the summary on the wepage
    tfvis
        .show
        .modelSummary({
            name: 'Ishaq Model Summary'
        }, model);
}
// Converting the data into tensors, that make training machine learing models
// practical Tensor is a data structure (scalars, vectors or matrices) Tensor
// can hold; ints floats and strings
function convertToTensor(data) {
    // wrap in a tidy will dispose any intermediate tensors STEP 1: shuffle the data
    //shuffle helps learn regardless of the order of the data
    //not be sensitive to subgroup bias
    return tf.tidy(() => {
        tf
            .util
            .shuffle(data)
    })
    //STEP 2: Convert data to tensor
    //
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    //STEP 3: normalise the data to the range 0 -1 using min-max scaling
    const inputMin = inputs.min()
    const inputMax = inputs.max()
    const labelMin = labels.min()
    const labelMax = labels.max()

    const normalisedInputs = inputTensor
        .sub(inputMin)
        .div(inputMax)
    const normaliseLabels = labelTensor
        .sub(labelMin)
        .div(labelMax)

    return {
        inputs: normalisedInputs,
        outputs: normaliseLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin
    }
}

document.addEventListener('DOMContentLoaded', run);
