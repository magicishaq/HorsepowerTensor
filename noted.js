//get the data ready for the model
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
    model.add(tf.layers.dense({units: 1, useBias: true}))
    return model
}








// Converting the data into tensors, that make training machine learing models
// practical Tensor is a data structure (scalars, vectors or matrices) Tensor
// can hold; ints floats and strings
function convertToTensor(data) {
    // wrap in a tidy will dispose any intermediate tensors STEP 1: shuffle the data
    // shuffle helps learn regardless of the order of the data not be sensitive to
    // subgroup bias
    return tf.tidy(() => {
        tf
            .util
            .shuffle(data)

        // STEP 2: Convert data to tensor make two array input is for training output is
        // the true values, known as labels in machine learing
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg)
        // converts each array into a 2d tensor //num of examples and num of features
        // per example
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
        const labelTensor = tf.tensor2d(labels, [labels.length, 1])

        // STEP 3: normalise the data to the range 0 -1 using min-max scaling got to put
        // the data within a range of 0-1 or -1-1 more success training models if the
        // data is normalised
        const inputMin = inputTensor.min()
        const inputMax = inputTensor.max()
        const labelMin = labelTensor.min()
        const labelMax = labelTensor.max()
        const normalizedInputs = inputTensor
            .sub(inputMin)
            .div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor
            .sub(labelMin)
            .div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs, outputs: normalizedLabels,
            //return min and max bounds so we can use them later
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }
    })
}








//Train the model
async function trainModel(model, inputs, labels) {
    // prepare model for training compling the model before it is trained optimizer
    // is the algorithm that is going to govern the updates to the model loss
    // function to tell the model how well it is learning meanSquaredError compares
    // the predictions with the true value
    model.compile({
        optimizer: tf
            .train
            .adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    });
    //batchSize refers to the size of the data subsets ranges from 32-512
    const batchSize = 32;
    // epochs refers to the times the model will look through the data. The
    // iterations threw the data
    const epochs = 50;
    // Starts the train loop async function returns promise to tell when training is
    // complete
    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis
            .show
            .fitCallbacks({
                name: 'Training Performance'
            }, [
                'loss', 'mse'
            ], {
                height: 200,
                callbacks: ['onEpochEnd']
            })
    });
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
    // creating an instance
    const model = createModel();
    //create a model and show the summary on the wepage
    tfvis
        .show
        .modelSummary({
            name: 'Model Summary'
        }, model);

    //convert the data to a form we can use for training
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;
    //train the model
    await trainModel(model, inputs, labels)
    console.log('Done training')
}




document.addEventListener('DOMContentLoaded', run);
