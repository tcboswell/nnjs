/*Copyright (C) <2017>  <Terry Boswell>
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.*/
    
//////////////////////////////////////////////////
///// RNN: Recurrent Neural Network With Long Short-Term Memory
///////////////////////////////////////////////////

RNN.prototype.relu = function (x) {
    var r = zeros(x.length);
    for (var j = 0; j < r.length; ++j) {
        r[j] = max([x[j], 0.01 * x[j]]);
    }
    return r;
}

RNN.prototype.backward_relu = function (x) {
    var r = zeros(x.length);
    for (var j = 0; j < r.length; ++j) {
        if (x[j] > 0) {
            r[j] = 1;
        }
        else { r[j] = 0.01; }
    }
    return r;
}

RNN.prototype.backward_tanh = function (x) {
    var r = zeros(x.length);
    var t;
    for (var j = 0; j < r.length; ++j) {
        t = Math.tanh(x[j]);
        r[j] = 1 - (t * t);
    }
    return r;
}

RNN.prototype.sigmoid = function (x) {
    var r = zeros(x.length)
    for (var j = 0; j < r.length; ++j) {
        r[j] = 1 / (1 + Math.exp(-x[j]));
    }
    return r;
}

RNN.prototype.backward_sigmoid = function (x) {
    var r = zeros(x.length);
    var s;
    for (var j = 0; j < r.length; ++j) {
        s = 1 / (1 + Math.exp(-x[j]));
        r[j] = (1 - s) * s;
    }
    return r;
}

RNN.prototype.softmax = function (x) {
    var v = zeros(x.length)
    var s = 0;
    for (var i = 0; i < x.length; ++i) {
        s = s + Math.exp(x[i]);
    }
    for (var j = 0; j < x.length; ++j) {
        v[j] = Math.exp(x[j]) / s;
    }
    return v
}

function RNN(params) {
    
    relu = RNN.prototype.relu;
    backward_relu = RNN.prototype.backward_relu;
    backward_tanh = RNN.prototype.backward_tanh;
    sigmoid = RNN.prototype.sigmoid;
    backward_sigmoid = RNN.prototype.backward_sigmoid;
    softmax = RNN.prototype.softmax;
    
    /* Default parameters: */
    
    this.hidden = 5;
    this.epochs = 10000;
    this.stopcriterion = 0.0;
    this.learningrate = 0.1;
    this.initialweightsbound = 0.01;
    this.windowsize = 10;
    this.variablewindowsize = false;
    this.LSTM = true;
    this.inneractivator = "sigmoid";
    this.outeractivator = "tanh";
    this.breakobject = undefined;
    
    /* Set parameters: */
    var i;
    if (params) {
        for (i in params)
            this[i] = params[i];
    }
    
    if (this.inneractivator == "sigmoid") {
        innerActivator = sigmoid;
        backwardInnerActivator = backward_sigmoid;
    }
    if (this.outeractivator == "sigmoid") {
        outerActivator = sigmoid;
        backwardOuterActivator = backward_sigmoid;
    }
    if (this.inneractivator == "relu") {
        innerActivator = relu;
        backwardInnerActivator = backward_relu;
    }
    if (this.outeractivator == "relu") {
        outerActivator = relu;
        backwardOuterActivator = backward_relu;
    }
    if (this.inneractivator == "tanh") {
        innerActivator = tanhVector;
        backwardInnerActivator = backward_tanh;
    }
    if (this.outeractivator == "tanh") {
        outerActivator = tanhVector;
        backwardOuterActivator = backward_tanh;
    }
    
}
RNN.prototype.train = function (X) {
    
    /* if X is a javascript array */
    if (Array.isArray(X)) {
        /* Create an inventory of the objects in the sequence by testing each element in objects for having
        *        the property of being the first element of its kind, then applying this test as a filter on objects */
        var objects = X.filter(function (element, index, X) {
            /* find the index of the first object in X which is like element */
            var firstIndex = X.indexOf(element);
            /* return true iff the index of element is firstIndex*/
            return index == firstIndex;
        });
    }
    /* if x is a laloliblab matrix retrieved with loaddata replace it with a javascript array and extract its objects into a set */
    else {
        var objects = [];
        var Xlist = [];
        for (var i=0; i < X.n; ++i) {
            Xlist.push(X.val[i])
            if (objects.indexOf(X.val[i]) < 0) {
                objects.push(X.val[i])
            }
        }
        X = Xlist;    
    }
    
    this.trainSequence(X, objects);
    
}

RNN.prototype.trainSequence = function (X, objects) {
    
    function calculateLoss(pointer, windowsize, inputSequence, targetSequence, objects, previousHiddenValue, previousMemoryCell,
                        weight, epsilon, LSTM) {
        
        var Whx = weight[0];
        var Whh = weight[1];
        var Wix = weight[2];
        var Wih = weight[3];
        var Wfx = weight[4];
        var Wfh = weight[5];
        var Wox = weight[6];
        var Woh = weight[7];
        var Wyh = weight[8];
        var Bhx = weight[9];
        var Bix = weight[10];
        var Bfx = weight[11];
        var Box = weight[12];
        var Byh = weight[13];
        
        var numberOfObjects = objects.length;
        var hiddenLayerSize = previousHiddenValue.length;
        
        /* javascript objects to hold the arrays at various points in time, these can then be used like matrices
        *        but with the advantage that it is possible to refer to an array with an index -1 within the object ; this facilitates
        *        reference to arrays from previous iterations in the recursive definitions of hidden values and memory cells */
        var inputValue = {};
        var hiddenValue = {};
        var linearHiddenValue = {};
        var squashedHiddenValue = {};
        var logitValue = {};
        var probValue = {};
        var inputGate = {};
        var forgetGate = {};
        var outputGate = {};
        var squashedInputGate = {};
        var squashedForgetGate = {};
        var squashedOutputGate = {};
        var memoryCell = {};
        
        /* difference variables for use in calculating deriatives */
        var delta_unHiddenValue;
        var delta_hiddenValue;
        var delta_logitValue;
        var delta_outputGate;
        var delta_inputGate;
        var delta_forgetGate;
        var delta_linearHiddenValue;
        var delta_previousHiddenValue;
        var delta_laterMemoryCell;
        
        var cost = 0;
        
        /* accumulators for gradients */
        var Delta_Whx;
        var Delta_Whh;
        var Delta_Wix;
        var Delta_Wih;
        var Delta_Wfx;
        var Delta_Wfh;
        var Delta_Wox;
        var Delta_Woh;
        var Delta_Wyh;
        var Delta_Bhx;
        var Delta_Bix;
        var Delta_Bfx;
        var Delta_Box;
        var Delta_Byh;
        
        /* set values of hidden states at previous point in time to the values from the previous iteration */
        hiddenValue[-1] = previousHiddenValue;
        /* similarly for memory cell values */
        memoryCell[-1] = previousMemoryCell;
        
        /* forward pass*/
        
        for (var i = 0; i < inputSequence.length; i++) {
            
            /* create array of object flags for time i, where a 1 at a given position in the array will signat that a given object is the input */
            inputValue[i] = zeros(numberOfObjects);
            
            /* set flag for object in ith position of input sequence */
            inputValue[i][inputSequence[i]] = 1;
            
            /* Output of hidden layer at ith time point, this output being a function of the input at time i and of the hidden
            *            value at time i-1 */
            linearHiddenValue[i] = vectorCopy(Bhx);
            gaxpy(Whh, hiddenValue[i - 1], linearHiddenValue[i])
            gaxpy(Whx, inputValue[i], linearHiddenValue[i])
            squashedHiddenValue[i] = outerActivator(linearHiddenValue[i]);
            
            /* if LSTM flag is not false */
            if (LSTM) {
                
                /* send input and previous hidden value through input gate */
                inputGate[i] = vectorCopy(Bix);
                gaxpy(Wih, hiddenValue[i - 1], inputGate[i]);
                gaxpy(Wix, inputValue[i], inputGate[i]);
                squashedInputGate[i] = innerActivator(inputGate[i]);
                
                /* send input and previous hidden value through forget gate */
                forgetGate[i] = vectorCopy(Bfx);
                gaxpy(Wfh, hiddenValue[i - 1], forgetGate[i]);
                gaxpy(Wfx, inputValue[i], forgetGate[i]);
                squashedForgetGate[i] = innerActivator(forgetGate[i]);
                
                /* send input and previous hidden value through output gate */
                outputGate[i] = vectorCopy(Box);
                gaxpy(Woh, hiddenValue[i - 1], outputGate[i]);
                gaxpy(Wox, inputValue[i], outputGate[i]);
                squashedOutputGate[i] = innerActivator(outputGate[i]);
                
                /* revise memory cell */
                memoryCell[i] = addVectors(entrywisemulVector(memoryCell[i - 1], squashedForgetGate[i]), entrywisemulVector(squashedHiddenValue[i], squashedInputGate[i]));
                
                /* squash memory cell values and apply output gate */
                hiddenValue[i] = entrywisemulVector(outerActivator(memoryCell[i]), squashedOutputGate[i]);
            }
            /* if LSTM flag is false then set the non-gated hidden value as the hidden value */
            else {
                hiddenValue[i] = squashedHiddenValue[i];
            }
            
            /* output layer before logistic transformation */
            logitValue[i] = vectorCopy(Byh);
            gaxpy(Wyh, hiddenValue[i], logitValue[i]);
            
            /* output values transformed into probabilities */
            probValue[i] = softmax(logitValue[i]);
            
            /* recalculation of total cost */
            cost = cost - Math.log(probValue[i][targetSequence[i]]);
            
        }
        
        /* reset gradient accumulators, neglecting those unneeded if the LSTM flag is false */
        if (LSTM) {
            Delta_Whx = zeros(hiddenLayerSize, numberOfObjects);
            Delta_Whh = zeros(hiddenLayerSize, hiddenLayerSize);
            Delta_Wix = zeros(hiddenLayerSize, numberOfObjects);
            Delta_Wih = zeros(hiddenLayerSize, hiddenLayerSize);
            Delta_Wfx = zeros(hiddenLayerSize, numberOfObjects);
            Delta_Wfh = zeros(hiddenLayerSize, hiddenLayerSize);
            Delta_Wox = zeros(hiddenLayerSize, numberOfObjects);
            Delta_Woh = zeros(hiddenLayerSize, hiddenLayerSize);
            Delta_Wyh = zeros(numberOfObjects, hiddenLayerSize);
            Delta_Bhx = zeros(hiddenLayerSize)
            Delta_Bix = zeros(hiddenLayerSize)
            Delta_Bfx = zeros(hiddenLayerSize)
            Delta_Box = zeros(hiddenLayerSize)
            Delta_Byh = zeros(numberOfObjects)
        }
        else {
            Delta_Whx = zeros(hiddenLayerSize, numberOfObjects);
            Delta_Whh = zeros(hiddenLayerSize, hiddenLayerSize);
            Delta_Wyh = zeros(numberOfObjects, hiddenLayerSize);
            Delta_Bhx = zeros(hiddenLayerSize)
            Delta_Byh = zeros(numberOfObjects)
        }
        
        /* reset differences, neglecting those unneeded if the LSTM flag is false */
        if (LSTM) {
            delta_laterMemoryCell = zeros(hiddenLayerSize);
            delta_logitValue = zeros(numberOfObjects);
            delta_unHiddenValue = zeros(hiddenLayerSize);
            delta_hiddenValue = zeros(hiddenLayerSize);
            delta_outputGate = zeros(hiddenLayerSize);
            delta_inputGate = zeros(hiddenLayerSize);
            delta_forgetGate = zeros(hiddenLayerSize);
            delta_linearHiddenValue = zeros(hiddenLayerSize);
            delta_previousHiddenValue = zeros(hiddenLayerSize);
        }
        else {
            delta_logitValue = zeros(numberOfObjects);
            delta_hiddenValue = zeros(hiddenLayerSize);
            delta_unHiddenValue = zeros(hiddenLayerSize);
            delta_previousHiddenValue = zeros(hiddenLayerSize);
        }
        
        /* backward pass*/
        
        for (var i = inputSequence.length - 1; i >= 0; i--) {
            
            /* let delta_logitValue be the derivative of the cross entropy loss function with respect
            *            to logitValue; this can be conveniently calculated as
            *            the respective probability value minus one */
            delta_logitValue = vectorCopy(probValue[i]);
            delta_logitValue[targetSequence[i]] = delta_logitValue[targetSequence[i]] - 1;
            
            /* since the partial derivative of the output function with respect to Wyh is hiddenValue
            *            the outer product of delta_logitValue and hiddenValue is the partial derivative
            *            of the cross entropy loss function with respect to Wyh, so let
            *            Delta_Wyh accumulate this derivative during each iteration in the backward pass,
            *            i.e. once for each traversal of the hidden nodes within a time window */
            Delta_Wyh = addMatrices(Delta_Wyh, outerprodVectors(delta_logitValue, hiddenValue[i]));
            
            /* likewise the partial derivative of the cross-entropy loss with respect to Byh
            *            shall be accumulated in */
            Delta_Byh = addVectors(Delta_Byh, delta_logitValue)
            
            /* the hidden value is calculated recursively, i.e. hiddenValue at time i depends on hiddenValue at time i-1,
            *            so the same value is used in two iterations, once as hiddenValue[i] and once as hiddenValue[i-1], thus
            *            the partial derivative of the cross-entropy loss
            *            function with respect to the hidden value, i.e. the error backpropagated to the hidden layer, shall be
            *            increased by addition of a quantity delta_previousHiddenValue which was caclulated in the previous iteration */
            delta_hiddenValue = delta_previousHiddenValue;
            gaxpy(transposeMatrix(Wyh), delta_logitValue, delta_hiddenValue)
            
            /* if the LSTM flag is not false*/
            if (LSTM) {
                /*as with hidden Value, the same memoryCell quantity occurs in two iterations, once as memoryCell[i] and once as memoryCell[i-1],
                *                so the partial derivative of the cross-entropy loss function with respect to memoryCell[i] shall be
                *                increased by addition of a quantity delta_laterMemory Cell which was calculated in the previous iteration */
                delta_memoryCell = addVectors(entrywisemulVector(
                    entrywisemulVector(
                        squashedOutputGate[i], delta_hiddenValue),
                        backwardOuterActivator(memoryCell[i])), delta_laterMemoryCell);
                
                /* let delta_ouputGate be the partial derivative of the loss with respect to the value of the output gate */
                delta_outputGate = entrywisemulVector(
                    backwardInnerActivator(outputGate[i]), entrywisemulVector(
                        outerActivator(memoryCell[i]), delta_hiddenValue));
                
                /* let delta_forgetGate be the partial derivative of the loss with respect to squashedForgetGate */
                delta_forgetGate = entrywisemulVector(
                    backwardInnerActivator(forgetGate[i]), entrywisemulVector(
                        memoryCell[i - 1], delta_memoryCell));
                
                /* let delta_inputGate be the partial derivative of the loss with respect to squashedInputGate */
                delta_inputGate = entrywisemulVector(
                    backwardInnerActivator(inputGate[i]), entrywisemulVector(
                        squashedHiddenValue[i], delta_memoryCell));
                
                /* let delta_linearHiddenValue be the partial derivative of the loss with respect to the hidden value before application of the outer activator */
                delta_linearHiddenValue = entrywisemulVector(backwardOuterActivator(linearHiddenValue[i]), entrywisemulVector(delta_memoryCell, squashedInputGate[i]));
                
                /* caclulate the derivative with respect to hiddenValue[i-1] which is to be
                *                saved for the next time point to be processed in backward traversal of the time series,
                *                i.e the previous point in the absolute time series,
                *                so that it can be added into the calculation of the derivative with respect to hiddenValue[i] */
                /* this must be calculated four times and summed since hiddenValue[i-1] enters into the forward pass at
                *                four independent points */
                delta_previousHiddenValue = mulMatrixVector(transpose(Whh), delta_linearHiddenValue);
                gaxpy(transpose(Woh), delta_outputGate, delta_previousHiddenValue);
                gaxpy(transpose(Wfh), delta_forgetGate, delta_previousHiddenValue);
                gaxpy(transpose(Wih), delta_inputGate, delta_previousHiddenValue);
                
                /* calculate the derivative with respect to memoryCell[i-1] which is to be saved for the next iteration and added into the
                *                calculation of the derivative with respect to the memory cell */
                delta_laterMemoryCell = entrywisemulVector(delta_memoryCell, squashedForgetGate[i]);
                
                /* add the derivatives with respect to the weights Whh, Wih, Wfh, Woh, to their respective
                *                accumulators */
                Delta_Whh = addMatrices(Delta_Whh, outerprodVectors(delta_linearHiddenValue, hiddenValue[i - 1]));
                Delta_Woh = addMatrices(Delta_Woh, outerprodVectors(delta_outputGate, hiddenValue[i - 1]));
                Delta_Wih = addMatrices(Delta_Wih, outerprodVectors(delta_inputGate, hiddenValue[i - 1]));
                Delta_Wfh = addMatrices(Delta_Wfh, outerprodVectors(delta_forgetGate, hiddenValue[i - 1]));
                
                /* add the derivatives with respect to the weights Bhx, Box, Bix, Bfx, to their respective
                *                accumulators */
                Delta_Bhx = addVectors(Delta_Bhx, delta_linearHiddenValue);
                Delta_Box = addVectors(Delta_Box, delta_outputGate);
                Delta_Bix = addVectors(Delta_Bix, delta_inputGate);
                Delta_Bfx = addVectors(Delta_Bfx, delta_forgetGate);
                
                /* add the derivatives with respect to the weights Whx, Wox, Wix, Wfx, to their respective
                *                accumulators */
                Delta_Whx = addMatrices(Delta_Whx, outerprodVectors(delta_linearHiddenValue, inputValue[i]));
                Delta_Wox = addMatrices(Delta_Wox, outerprodVectors(delta_outputGate, inputValue[i]));
                Delta_Wix = addMatrices(Delta_Wix, outerprodVectors(delta_inputGate, inputValue[i]));
                Delta_Wfx = addMatrices(Delta_Wfx, outerprodVectors(delta_forgetGate, inputValue[i]));
                
            }
            /* if the LSTM flag is false */
            else {
                /* let delta_unHiddenValue be delta_hiddenValue multiplied with
                *                the partial derivative of hiddenValue with respect to linearHiddenValue */
                delta_unHiddenValue = entrywisemulVector(backwardOuterActivator(linearHiddenValue[i]), delta_hiddenValue);
                
                /* the product thereof with hiddenValue[i-1] is the partial derivative of the loss function
                *                with respect to Whh, and this shall be accumulated in the hidden weight gradient accumulator */
                Delta_Whh = addMatrices(Delta_Whh, outerprodVectors(delta_unHiddenValue, hiddenValue[i - 1]))
                
                /* the product of delta_unHiddenValue and inputValue is the partial derivative of the loss funtion
                *                with respect to Whx and this shall be added to the appropriate entry weight gradient accumulator */
                Delta_Whx = addMatrices(Delta_Whx, outerprodVectors(delta_unHiddenValue, inputValue[i]))
                
                /* likewise the partial derivative of the loss with respect to Bhx
                *                shall be accumulated in Delta_Bhx during every iteration of the backward pass */
                Delta_Bhx = addVectors(Delta_Bhx, delta_unHiddenValue);
                
                /* the product of the hidden weight and the error backpropagated to the input layer
                *                shall be saved for the next time point to be processed in backward
                *                traversal of the time series, i.e the previous point in the absolute time series: */
                delta_previousHiddenValue = mulMatrixVector(transposeMatrix(Whh), delta_unHiddenValue);
                
            }
        }
        /* return accumulators together with the last hidden value vector and the last memory cell, which are
        *        to be used as input for the next forward pass, also return the cost and the probablity values for use when displaying progress */
        return [probValue, hiddenValue[inputSequence.length - 1], memoryCell[inputSequence.length - 1],
        [Delta_Whx, Delta_Whh, Delta_Wix, Delta_Wih, Delta_Wfx, Delta_Wfh,
        Delta_Wox, Delta_Woh, Delta_Wyh, Delta_Bhx, Delta_Bix, Delta_Bfx, Delta_Box, Delta_Byh], cost];
        
                        }
                        
                        function displayProgress(iteration, testOutput,
                                                    loss, objects, inputs, targets) {
                            
                            var digit;
                            var text = "predicted sequence: ";
                            var counter = 0;
                            
                            console.log("iteration: " + iteration);
                            console.log("loss: " + loss / inputs.length)
                            
                            /* create an array of key values for the output map */
                            var keys = [];
                            for (var key in testOutput) {
                                keys.push(parseInt(key));
                            }
                            /* for each key in that map find the output value with the greatest probability */
                            var textmatches = "";
                            for (var i = 0; i < keys.length; i++) {
                                /* find the index of the object with greatest probability */
                                digit = findmax(testOutput[i])
                                /* retrieve the object for display */
                                text = text + " " + objects[digit]
                                /* note whether the prediction is identical to the target and build a corresponding pattern of +s and -s */
                                if (digit == targets[i]) {
                                    counter += 1;
                                    textmatches = textmatches + " +";
                                }
                                else {
                                    textmatches = textmatches + " -";
                                }
                            }
                            var textinputs = "";
                            var texttargets = "";
                            /* retrieve input and target sequences for display */
                            for (var k = 0; k < inputs.length; k++) {
                                textinputs = textinputs + " " + objects[inputs[k]];
                                texttargets = texttargets + " " + objects[targets[k]];
                            }
                            console.log("input sequence: " + textinputs)
                            console.log("target sequence: " + texttargets)
                            console.log(text)
                            console.log((counter) + " of " + (targets.length) + " objects matched in pattern: " + textmatches)
                            text = "";
                                                    }
                                                    
                                                    function updateParameters(weight, gradient, gradientMemory, epsilon) {
                                                        
                                                        /* every time a paramter is updated divide the learning rate by a quantity which is a sort of memory of past gradients,
                                                        *        thus forcing parameters which have been updated more severely to be updated at a slower learning rate */
                                                        
                                                        var tmp0;
                                                        var tmp1;
                                                        var tmp2;
                                                        
                                                        for (var i = 0; i < weight.length; i++) {
                                                            if (type(weight[i]) == "matrix") {
                                                                tmp0 = mulScalarMatrix(epsilon, gradient[i]);
                                                                gradientMemory[i] = addMatrices(gradientMemory[i], entrywisemulMatrix(gradient[i], gradient[i]));
                                                                tmp2 = sqrtMatrix(addScalarMatrix(0.00000001, gradientMemory[i]));
                                                                tmp1 = divMatrices(tmp0, tmp2);
                                                                weight[i] = subMatrices(weight[i], tmp1)
                                                            }
                                                            if (type(weight[i]) == "vector") {
                                                                tmp0 = mulScalarVector(epsilon, gradient[i]);
                                                                gradientMemory[i] = addVectors(gradientMemory[i], entrywisemulVector(gradient[i], gradient[i]));
                                                                tmp2 = sqrtVector(addScalarVector(0.00000001, gradientMemory[i]))
                                                                tmp1 = divVectors(tmp0, tmp2);
                                                                weight[i] = subVectors(weight[i], tmp1);
                                                            }
                                                        }
                                                        
                                                        return [weight, gradientMemory];
                                                    }
                                                    
                                                    /* number of objects in data */
                                                    var N = X.length;
                                                    
                                                    /* number of distinct objects in data */
                                                    var numberOfObjects = objects.length;
                                                    
                                                    /* hyperparameters */
                                                    const hiddenLayerSize = this.hidden;
                                                    var windowsize = this.windowsize;
                                                    var epsilon = this.learningrate;
                                                    
                                                    /* stop iterating through data at this pass */
                                                    const maxEpochs = this.epochs;
                                                    
                                                    /* stop iterating through data at this error threshhold */
                                                    const stopCriterion = this.stopcriterion;
                                                    
                                                    /* weights */
                                                    var Whx;
                                                    var Whh;
                                                    var Wix;
                                                    var Wih;
                                                    var Wfx;
                                                    var Wox;
                                                    var Wfh;
                                                    var Woh;
                                                    var Wyh;
                                                    var Bhx;
                                                    var Bix;
                                                    var Bfx;
                                                    var Box;
                                                    var Byh;
                                                    /* initialize the weights, neglecting those unneeded if the LSTM flag is false */
                                                    if (this.LSTM) {
                                                        Whx = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, numberOfObjects), 0.5));
                                                        Whh = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, hiddenLayerSize), 0.5));
                                                        Wix = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, numberOfObjects), 0.5));
                                                        Wih = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, hiddenLayerSize), 0.5));
                                                        Wfx = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, numberOfObjects), 0.5));
                                                        Wox = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, numberOfObjects), 0.5));
                                                        Wfh = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, hiddenLayerSize), 0.5));
                                                        Woh = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, hiddenLayerSize), 0.5));
                                                        Wyh = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(numberOfObjects, hiddenLayerSize), 0.5));
                                                        Bhx = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(hiddenLayerSize), 0.5));
                                                        Bix = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(hiddenLayerSize), 0.5));
                                                        Bfx = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(hiddenLayerSize), 0.5));
                                                        Box = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(hiddenLayerSize), 0.5));
                                                        Byh = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(numberOfObjects), 0.5));
                                                    }
                                                    else {
                                                        Whx = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, numberOfObjects), 0.5));
                                                        Whh = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(hiddenLayerSize, hiddenLayerSize), 0.5));
                                                        Wyh = mulScalarMatrix(this.initialweightsbound, subMatrixScalar(rand(numberOfObjects, hiddenLayerSize), 0.5));
                                                        Bhx = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(hiddenLayerSize), 0.5));
                                                        Byh = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(numberOfObjects), 0.5));
                                                    }
                                                    /* collect the weights in an array to facilitate function calls*/
                                                    var weight = [Whx, Whh, Wix, Wih, Wfx, Wfh, Wox, Woh, Wyh, Bhx, Bix, Bfx, Box, Byh];
                                                    
                                                    /* accumulators for partial derivatives */
                                                    var Delta_Wyh;
                                                    var Delta_Whh;
                                                    var Delta_Wih;
                                                    var Delta_Wfh;
                                                    var Delta_Woh;
                                                    var Delta_Whx;
                                                    var Delta_Wix;
                                                    var Delta_Wfx;
                                                    var Delta_Wox;
                                                    var Delta_Byh;
                                                    var Delta_Bhx;
                                                    var Delta_Bix;
                                                    var Delta_Bfx;
                                                    var Delta_Box;
                                                    var gradient = [
                                                    Delta_Whx,
                                                    Delta_Whh,
                                                    Delta_Wix,
                                                    Delta_Wih,
                                                    Delta_Wfx,
                                                    Delta_Wfh,
                                                    Delta_Wox,
                                                    Delta_Woh,
                                                    Delta_Wyh,
                                                    Delta_Bhx,
                                                    Delta_Bix,
                                                    Delta_Bfx,
                                                    Delta_Box,
                                                    Delta_Byh
                                                    ];
                                                    
                                                    /* auxiliary variables */
                                                    var object;
                                                    var digit;
                                                    var this_digit;
                                                    var next_object;
                                                    var next_digit;
                                                    var inputs;
                                                    var targets;
                                                    var loss = 0;
                                                    var testOutput;
                                                    
                                                    /* initialize previous hidden value variables */
                                                    var previousHiddenValue = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(this.hidden), 0.5));
                                                    if (this.LSTM) {
                                                        /* initialize previous memory cell variables */
                                                        var previousMemoryCell = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(this.hidden), 0.5));
                                                    }
                                                    
                                                    /* counter for training iterations */
                                                    var iterator = 0;
                                                    /* position of the window within data */
                                                    var pointer;
                                                    
                                                    /* variables for adagrad */
                                                    var mWhx;
                                                    var mWhh;
                                                    var mWix;
                                                    var mWih;
                                                    var mWfx;
                                                    var mWfh;
                                                    var mWox;
                                                    var mWoh;
                                                    var mWyh;
                                                    var mBhx;
                                                    var mBix;
                                                    var mBfx;
                                                    var mBox;
                                                    var mByh;
                                                    /* initialize the adagrad variables, neglecting those unneeded if the LSTM flag is false */
                                                    if (this.LSTM) {
                                                        mWhx = zeros(hiddenLayerSize, numberOfObjects);
                                                        mWhh = zeros(hiddenLayerSize, hiddenLayerSize);
                                                        mWix = zeros(hiddenLayerSize, numberOfObjects);
                                                        mWih = zeros(hiddenLayerSize, hiddenLayerSize);
                                                        mWfx = zeros(hiddenLayerSize, numberOfObjects);
                                                        mWfh = zeros(hiddenLayerSize, hiddenLayerSize);
                                                        mWox = zeros(hiddenLayerSize, numberOfObjects);
                                                        mWoh = zeros(hiddenLayerSize, hiddenLayerSize);
                                                        mWyh = zeros(numberOfObjects, hiddenLayerSize);
                                                        mBhx = zeros(hiddenLayerSize);
                                                        mBix = zeros(hiddenLayerSize);
                                                        mBfx = zeros(hiddenLayerSize);
                                                        mBox = zeros(hiddenLayerSize);
                                                        mByh = zeros(numberOfObjects);
                                                    }
                                                    else {
                                                        mWhx = zeros(hiddenLayerSize, numberOfObjects);
                                                        mWhh = zeros(hiddenLayerSize, hiddenLayerSize);
                                                        mWyh = zeros(numberOfObjects, hiddenLayerSize);
                                                        mBhx = zeros(hiddenLayerSize);
                                                        mByh = zeros(numberOfObjects);
                                                    }
                                                    /* collect the variables in an array to facilitate function calls */
                                                    var gradientMemory = [mWhx, mWhh, mWix, mWih, mWfx, mWfh, mWox, mWoh, mWyh, mBhx, mBix, mBfx, mBox, mByh];
                                                    
                                                    /* set the variable to store the error value high so that the while loop will be entered initially;
                                                    *    after the loop has been traversed the variable will contain the current average error */
                                                    var stop = 9;
                                                    
                                                    var subsequenceStart = 0;
                                                    var breakFlag = false;
                                                    
                                                    while (iterator < maxEpochs && stop > stopCriterion) {
                                                        
                                                        /* at first iteration or if there is no room left in the window reset previous hidden nodes and memory nodes,
                                                        *        and reset pointer to beginnng of data */
                                                        
                                                        /* if the data consists of subsequences each of which should be processed separately,
                                                        *        in which case an object designating breaks in the sequence has been defined */
                                                        if (this.breakobject != undefined) {
                                                            /* set the window size to be the entire subsequence */
                                                            windowsize = X.slice(subsequenceStart, X.length).indexOf(this.breakobject);
                                                            /* unless the last breakobject has already been processed so that the windowsize has now become -1 */
                                                            if (windowsize != -1) {
                                                                subsequenceStart = subsequenceStart + windowsize + 1;
                                                                /* advance the pointer within the entire sequence one step so that the break object is skipped */
                                                                ++pointer;
                                                            }
                                                            else {
                                                                /* set the window size as the length of the initial subsequence */
                                                                windowsize = X.slice(0, X.length).indexOf(this.breakobject);
                                                                /* mark the start of the next subsequence to be after the window */
                                                                subsequenceStart = windowsize + 1;
                                                                /* set a flag to indicate that the hidden state and memory cell should be reset */
                                                                breakFlag = true;
                                                            }
                                                        }
                                                        
                                                        if (pointer + windowsize + 1 >= N || iterator == 0 || breakFlag == true) {
                                                            /* reset the flag */
                                                            breakFlag = false;
                                                            previousHiddenValue = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(this.hidden), 0.5));
                                                            if (this.LSTM) {
                                                                previousMemoryCell = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(this.hidden), 0.5));
                                                            }
                                                            if (this.variablewindowsize) {
                                                                /* randomize the processing of data by randomly changing the window size, within the limits of windowsize and a minimum of three,
                                                                *                every time the data is passed through */
                                                                windowsize = Math.floor(Math.random() * (this.windowsize - 2)) + 3;
                                                            }
                                                            pointer = 0;
                                                        }
                                                        
                                                        inputs = zeros(windowsize);
                                                        targets = zeros(windowsize);
                                                        
                                                        /* for each point in the window create binary input and output vectors, containing zeros except for a one which flags an object's identity */
                                                        for (var i = pointer; i < pointer + windowsize; i++) {
                                                            /* find the object which the pointer is directed at */
                                                            object = X[i];
                                                            /* find the index of that object within the array of objects */
                                                            digit = objects.indexOf(object);
                                                            /* use that index to denote the object within an array of inputs */
                                                            inputs[i - pointer] = digit;
                                                            /* find the object after the pointer */
                                                            next_object = X[i + 1];
                                                            /* find the index of that object within the array of objects */
                                                            next_digit = objects.indexOf(next_object);
                                                            /* use that index to denote the object within an array of targets */
                                                            targets[i - pointer] = next_digit;
                                                        }
                                                        
                                                        /* calculate loss and gradients, also retrieving the previous hidden and memory values as well as the momentary output for
                                                        *        display of progress */
                                                        [testOutput, previousHiddenValue, previousMemoryCell, gradient, cost] = calculateLoss(pointer, windowsize, inputs, targets,
                                                                                                                                            objects, previousHiddenValue, previousMemoryCell, weight, epsilon, this.LSTM);
                                                        
                                                        /* display progress of training at regular intervals */
                                                        if (iterator % 100 == 0) {
                                                            console.log("pointer position: " + pointer)
                                                            displayProgress(iterator, testOutput, cost,
                                                                            objects, inputs, targets);
                                                        }
                                                        /* record the average cost for checking whether a threshhold has been passed */
                                                        stop = cost / inputs.length;
                                                        
                                                        /* update weights and adagrad variables */
                                                        [weight, gradientMemory] = updateParameters(weight, gradient, gradientMemory, epsilon);
                                                        
                                                        /* count training pass */
                                                        iterator = iterator + 1;
                                                        
                                                        /* move window forward */
                                                        pointer = pointer + windowsize;
                                                        
                                                    }
                                                    
                                                    /* store the inventory of objects and the final weight values for use in the prediction function */
                                                    this.objects = objects;
                                                    this.Whx = weight[0];
                                                    this.Whh = weight[1];
                                                    this.Wix = weight[2];
                                                    this.Wih = weight[3];
                                                    this.Wfx = weight[4];
                                                    this.Wfh = weight[5];
                                                    this.Wox = weight[6];
                                                    this.Woh = weight[7];
                                                    this.Wyh = weight[8];
                                                    this.Bhx = weight[9];
                                                    this.Bix = weight[10];
                                                    this.Bfx = weight[11];
                                                    this.Box = weight[12];
                                                    this.Byh = weight[13];
}

RNN.prototype.predict = function (initialSequence, sequenceLength) {
    return this.predictSequence(initialSequence, sequenceLength);
}

RNN.prototype.predictSequence = function (initialSequence, sequenceLength) {
    
    var digit;
    
    inputs = zeros(initialSequence.length);
    
    /* for each point in the initial sequence create binary input and output vectors, containing 0s except for a 1 which flags an object's identity */
    for (var i = 0; i < initialSequence.length; i++) {
        /* find the object at this position in the sequence */
        object = initialSequence[i];
        /* find the index of that object within the array of objects */
        digit = this.objects.indexOf(object);
        /* use that index to denote the object within an array of inputs */
        inputs[i] = digit;
    }
    
    var previousHiddenValue = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(this.hidden), 0.5));
    var previousMemoryCell = mulScalarVector(this.initialweightsbound / 10, subVectorScalar(rand(this.hidden), 0.5));
    
    var hiddenValue;
    var linearHiddenValue;
    var inputGate;
    var forgetGate;
    var outputGate;
    var squashedHiddenValue;
    var squashedInputGate;
    var squashedForgetGate;
    var squashedOutputGate;
    var memoryCell;
    var logitValue;
    var probValue;
    var sequence = [];
    
    var inputValue = {};
    var numberOfObjects = this.objects.length;
    
    for (var i = 0; i < initialSequence.length; i++) {
        
        /* create array of object flags for time i, where a 1 at a given position in the array will signal that a given object is the input */
        inputValue[i] = zeros(numberOfObjects);
        
        /* set flag for object in ith position of input sequence */
        inputValue[i][inputs[i]] = 1;
        
        /* Output of hidden layer at ith time point, this output being a function of the input at time i and of the hidden
        *        value at time i-1 */
        linearHiddenValue = vectorCopy(this.Bhx);
        gaxpy(this.Whh, previousHiddenValue, linearHiddenValue);
        gaxpy(this.Whx, inputValue[i], linearHiddenValue);
        squashedHiddenValue = outerActivator(linearHiddenValue);
        
        /* if the LSTM flag is not false */
        if (this.LSTM) {
            inputGate = vectorCopy(this.Bix);
            gaxpy(this.Wih, previousHiddenValue, inputGate);
            gaxpy(this.Wix, inputValue[i], inputGate);
            squashedInputGate = innerActivator(inputGate);
            
            forgetGate = vectorCopy(this.Bfx);
            gaxpy(this.Wfh, previousHiddenValue, forgetGate);
            gaxpy(this.Wfx, inputValue[i], forgetGate);
            squashedForgetGate = innerActivator(forgetGate);
            
            outputGate = vectorCopy(this.Box);
            gaxpy(this.Woh, previousHiddenValue, outputGate);
            gaxpy(this.Wox, inputValue[i], outputGate);
            squashedOutputGate = innerActivator(outputGate);
            
            memoryCell = addVectors(entrywisemulVector(previousMemoryCell, squashedForgetGate), entrywisemulVector(squashedHiddenValue, squashedInputGate));
            
            hiddenValue = entrywisemulVector(outerActivator(memoryCell), squashedOutputGate);
        }
        /* if the LSTM flag is false */
        else {
            hiddenValue = squashedHiddenValue;
        }
        
        /* save hidden values as input to next calculation */
        previousHiddenValue = hiddenValue;
        
        if (this.LSTM) {
            /* save memory cell as input to next calculation if LSTM flag is set */
            previousMemoryCell = memoryCell;
        }
        
    }
    
    /* Output layer before logistic transformation */
    logitValue = vectorCopy(this.Byh);
    gaxpy(this.Wyh, hiddenValue, logitValue);
    
    /* Output of output layer afer logistic transformation */
    probValue = softmax(logitValue);
    
    /* find the position of the highest probability */
    digit = findmax(probValue);
    sequence.push(this.objects[digit]);
    
    /* save object with maximum probability as the initial object in the following sequence */
    initialObjectVector = zeros(this.objects.length)
    initialObjectVector[digit] = 1;
    
    for (var i = 0; i < sequenceLength - 1; i++) {
        
        /* Output of hidden layer at ith time point, this output being a function of the input at time i and of the hidden
        *        value at time i-1 */
        linearHiddenValue = vectorCopy(this.Bhx);
        gaxpy(this.Whh, previousHiddenValue, linearHiddenValue);
        gaxpy(this.Whx, initialObjectVector, linearHiddenValue);
        squashedHiddenValue = outerActivator(linearHiddenValue);
        
        /* if the LSTM flag is not false */
        if (this.LSTM) {
            inputGate = vectorCopy(this.Bix);
            gaxpy(this.Wih, previousHiddenValue, inputGate);
            gaxpy(this.Wix, initialObjectVector, inputGate);
            squashedInputGate = innerActivator(inputGate);
            
            forgetGate = vectorCopy(this.Bfx);
            gaxpy(this.Wfh, previousHiddenValue, forgetGate);
            gaxpy(this.Wfx, initialObjectVector, forgetGate);
            squashedForgetGate = innerActivator(forgetGate);
            
            outputGate = vectorCopy(this.Box);
            gaxpy(this.Woh, previousHiddenValue, outputGate);
            gaxpy(this.Wox, initialObjectVector, outputGate);
            squashedOutputGate = innerActivator(outputGate);
            
            memoryCell = addVectors(entrywisemulVector(previousMemoryCell, squashedForgetGate), entrywisemulVector(squashedHiddenValue, squashedInputGate));
            
            hiddenValue = entrywisemulVector(outerActivator(memoryCell), squashedOutputGate);
        }
        /* if the LSTM flag is false */
        else {
            hiddenValue = squashedHiddenValue;
        }
        /* Output layer before logistic transformation */
        logitValue = vectorCopy(this.Byh);
        gaxpy(this.Wyh, hiddenValue, logitValue);
        
        /* Output of output layer afer logistic transformation */
        probValue = softmax(logitValue);
        
        /* find the position of the highest probability */
        digit = findmax(probValue);
        sequence.push(this.objects[digit]);
        
        /* save object with maximum probability as the initial object in the following sequence */
        initialObjectVector = zeros(this.objects.length)
        initialObjectVector[digit] = 1;
        
        /* save hidden values as input to next calculation */
        previousHiddenValue = hiddenValue;
        
        if (this.LSTM) {
            /* save memory cell as input to next calculation if LSTM flag is set */
            previousMemoryCell = memoryCell;
        }
        
    }
    
    console.log("prediction for sequence of " + sequenceLength + " objects following\n" + initialSequence + ": \n" + sequence)
    return array2mat(sequence);
}
