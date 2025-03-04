//Objective 6
//OOP
//Inheritance
//Polymorphism
//Neural Network
//Vectors
//Arrays

//math
using NumSharp;

//TFPPOModel ONLY!! CreateModel and its child classes DO NOT USE!! 
using TorchSharp;
using static TorchSharp.torch.nn;

//system usings
using System;

namespace FCNN
{
    #region activations
    //Objective 6c
    //activation functions
    public class Activation //parent class
    {
        public virtual float DoActivation(float x)
        {
            return x;
        }

        public virtual NDArray DoActivationArray(NDArray x)
        {
            return x;
        }

        public virtual NDArray DerivativeActivationArray(NDArray x, NDArray outputs)
        {
            return x;
        }
    }

    //Objectives 6ci\1, 6cii\1
    //Arrays
    public class ReLU : Activation //child classes
    {
        public override float DoActivation(float x)
        {
            return MathF.Max(0, x);
        }

        public override NDArray DoActivationArray(NDArray x)
        {
            
            return np.maximum(np.zeros(x.shape), x);
        }

        public override NDArray DerivativeActivationArray(NDArray x, NDArray outputs)
        {
            var mask = np.zeros(outputs.shape);
            
            for (int i = 0; i < outputs.shape[0]; i++)
                for (int j = 0; j < outputs.shape[1]; j++)
                {
                    float value = outputs[i, j];
                    mask[i, j] = (value > 0) ? 1 : 0;
                }
            

            return np.multiply(x, mask);
        }
    }
    
    //Objectives 6ci\3, 6cii\3
    //Arrays
    //Vectors
    public class Softmax : Activation{
        public override NDArray DoActivationArray(NDArray x)
        {
            NDArray output = np.zeros(x.shape);

            // Exponentiate each element (e^x)
            var expX = np.exp(x);

            // Create an array to store the sum of each row
            var sumExp = new double[expX.shape[0]];

            // Manually calculate the sum of each row
            for (int i = 0; i < expX.shape[0]; i++)
            {
                double rowSum = 0.0;
                for (int j = 0; j < expX.shape[1]; j++)
                {
                    rowSum += expX[i, j];
                }
                sumExp[i] = rowSum;
            }

            // Normalize each element by the row sum
            for (int i = 0; i < expX.shape[0]; i++)
            {
                for (int j = 0; j < expX.shape[1]; j++)
                {
                    output[i, j] = expX[i, j] / sumExp[i];
                }
            }

            return output;
        }

        public override NDArray DerivativeActivationArray(NDArray x, NDArray outputs)
        {
            NDArray derivative = np.zeros(x.shape);
            for (int i = 0; i < derivative.shape[0]; i++){
                NDArray gradient = derivative[i];

                if (gradient.shape.Length == 0){ //for single instance
                    gradient.reshape(-1, 1);
                }
                //create diagflat becaues numsharp does not have diagflat function 
                NDArray diagflat = np.zeros(gradient.shape[0], gradient.shape[0]);
                for (int j = 0; j < diagflat.shape[0]; j++){
                    diagflat[j, j] = gradient[j];
                }
                //do dot product manually because numsharp doesnt like it
                //Vectors
                NDArray gradientDotProduct = np.zeros(gradient.shape[0], gradient.shape[0]);
                for (int j = 0; j < gradientDotProduct.shape[0]; j++){
                    for (int k = 0; k < gradientDotProduct.shape[1]; k++){
                        gradientDotProduct[j, k] = gradient[j] * gradient[k];
                    }
                }

                //final calculations
                NDArray jacobian = diagflat - gradientDotProduct;
                derivative[i] = np.dot(jacobian, outputs[i].reshape(outputs[i].shape[0], 1));
            }
            return derivative;
        }
    }

    //Objective 6ci\2, 6cii\2
    //Arrays
    public class LReLU : Activation
    {
        public override float DoActivation(float x)
        {
            //if greater than 0, return value. 
            //else, return small amount of value.
            if (x >= 0) {
                return x;
            } else {
                return 0.01f * x;
            }
        }

        public override NDArray DerivativeActivationArray(NDArray x, NDArray outputs)
        {
            var mask = np.zeros(outputs.shape);
            
            for (int i = 0; i < outputs.shape[0]; i++)
                for (int j = 0; j < outputs.shape[1]; j++)
                {
                    float value = outputs[i, j];
                    mask[i, j] = (value > 0) ? 1 : 0.01; //like ReLU, but instead of going to 0 we go to 0.01
                }

            return np.multiply(x, mask);
        }

        public override NDArray DoActivationArray(NDArray x)
        {
            NDArray returnArray = np.zeros(x.shape);
            //for each value in input array, do activation
            for(int i = 0; i < x.shape[0]; i++){
                for(int j = 0; j < x.shape[1]; j++){
                    returnArray[i, j] = DoActivation(x[i, j]);
                }
            }
            return returnArray;
        }
    }

    //Objective 6cii\4
    public class Sigmoid : Activation
    {
        public override float DoActivation(float x)
        {
            float result = 1 / (1 + MathF.Exp(-x));
            return result;
        }
    }

    //Objective 6cii\5
    public class Tanh : Activation
    {
        public override float DoActivation(float x)
        {
            float result = MathF.Tanh(x);
            return result;
        }
    }
    #endregion

    #region FCLayer


    /* Implements MLP with Adam and other cool features, by using one class for a layer and another to fully construct the network. 
     * this is useful because we may require multiple different types of NN (ie A2N, PPO strategies*/
    public class FCLayer
    {
        public FCLayer(int inputSizeInput, int outputSizeInput, Activation activationInput)
        {
            inputSize = inputSizeInput;
            outputSize = outputSizeInput;
            activation = activationInput;
            weights = np.random.randn(inputSize, outputSize) * np.sqrt(2.0 / inputSize); //initialise weights according to HE-Initialisation
            biases = np.zeros(1, outputSize);
            mWeights = np.zeros(inputSize, outputSize);
            vWeights = np.zeros(inputSize, outputSize);
            mBiases = np.zeros(1, outputSize);
            vBiases = np.zeros(1, outputSize);
            //initialise hyperparameters for Adam optimiser
            beta1 = 0.9f;
            beta2 = 0.999f;
            epsilon = 1e-8f;
        }
        public int inputSize { get; }
        public int outputSize { get; }
        public Activation activation { get; }
        //regular weights and biases
        public NDArray weights;
        private NDArray biases;
        //derivatives of weights and biases
        public NDArray dWeights;
        private NDArray dBiases;
        /* Define m & v for weights and biases
         * These variables are used in Adam optimisation */
        private NDArray mWeights;
        private NDArray mBiases;
        private NDArray vWeights;
        private NDArray vBiases;
        private NDArray mHatBiases;
        private NDArray vHatBiases;
        private NDArray mHatWeights;
        private NDArray vHatWeights;
        //hyperparameters for Adam - ADD AS PARAMETERS FOR NN?
        private readonly float beta1; 
        private readonly float beta2;
        private readonly float epsilon;
        public NDArray output;
        private NDArray x;
        //Objective 6aii
        public NDArray Forward(NDArray X) //x is a layer input array to perform the forward pass on
        {
            x = X;
            NDArray z;
            //calculate the layer output z
            z = biases + np.dot(x, weights); //Objective 6aii\1
            //remove linearity using activation function
            output = activation.DoActivationArray(z); //Objective 6aii\2
            return output;
        }

        private NDArray dInputs;
        //Objective 6bii
        //Neural Network
        //Adam
        public NDArray Backward(NDArray dValues, float lr, int t) //dValues is the derivative of the output, t is the timestep
        {
            dValues = activation.DerivativeActivationArray(dValues, output); //Objective 6bii\1

            //Objective 6bii\2
            //calculte derivatives wrt weight and bias
            dWeights = np.dot(x.T, dValues);
            dBiases = SumOverAxis0(dValues);
            //limit derivatives to avoid really big or really small numbers
            dWeights = np.clip(dWeights, -1, 1);
            dBiases = np.clip(dBiases, -1, 1);


            //Objective 6bii\3
            //calculate gradient wrt to inputs
            dInputs = np.dot(dValues, weights.T);
            
            //Objective 6bii\4
            //update weights and biases using learning rate and derivatives
            weights -= lr * dWeights;
            biases -= lr * dBiases;

            //Objective 6bii\5
            //Adam
            //update weights using m and v values (Adam)
            mWeights = beta1 * mWeights + (1 - beta1) * dWeights;
            vWeights = beta2 * vWeights + (1 - beta2) * np.power(dWeights, 2);
            mHatWeights = mWeights / (1 - np.power(beta1, t));
            vHatWeights = vWeights / (1 - np.power(beta2, t));
            weights -= lr * mHatWeights / (np.sqrt(vHatWeights) + epsilon);

            //update biases using m and v values (Adam)
            mBiases = beta1 * mBiases + (1 - beta1) * dBiases;
            vBiases = beta2 * vBiases + (1 - beta2) * np.power(dBiases, 2);
            mHatBiases = mBiases / (1 - np.power(beta1, t));
            vHatBiases = vBiases / (1 - np.power(beta2, t));
            biases -= lr * mHatBiases / (np.sqrt(vHatBiases) + epsilon);

            Console.WriteLine(weights.ToString());
            Console.WriteLine(biases.ToString());

            //Objective 6bii\6
            return dInputs;
        }
        static NDArray SumOverAxis0(NDArray array) //fine... i'll do it myself. implements np.sum(array, axis:0, keepDims = True).
        {
            int rows = array.shape[0];
            int cols = array.shape[1];
            NDArray result = np.zeros(cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j] += array[i, j];
                }
            }

            return result.reshape(1, cols);
        }
    }

    #endregion

    #region CreateModel

    public class CreateModel
    {

        public int inputSize { get; set; }
        public int outputSize { get; set;}
        public int[] hiddenSizes { get; set;}
        public FCLayer layer1 { get; set; }
        public FCLayer layer2 { get; set; }
        public FCLayer layer3 { get; set; }
        public CreateModel(int inputSizeInput, int outputSizeInput, int[] hiddenSizesInput)

        {
            inputSize = inputSizeInput;
            outputSize = outputSizeInput;
            hiddenSizes = hiddenSizesInput;

            layer1 = new FCLayer(inputSize, hiddenSizes[0], new ReLU());
            layer2 = new FCLayer(hiddenSizes[0], hiddenSizes[1], new ReLU());
            layer3 = new FCLayer(hiddenSizes[1], outputSize, new Softmax());
        }

        //Objective 6ai
        public NDArray Forward(NDArray inputs)
        {
            //Objective 6ai\1
            NDArray output1 = layer1.Forward(inputs);
            NDArray output2 = layer2.Forward(output1);
            NDArray output3 = layer3.Forward(output2);

            return output3;
        }

        private int t = 0;
        private float lr;
        private NDArray outputGrad;
        private NDArray grad3;
        private NDArray grad2;
        private NDArray grad1;
        //Objective 6bi\
        public void Train(NDArray inputs, NDArray targets, int nEpochs, float initialLr, float decay)
        {
            for (int epoch = 0; epoch < nEpochs; epoch++)
            {
                //forward pass
                NDArray output = Forward(inputs);//Objective 6bi\1

                //backwards pass
                outputGrad = 6 * (output - targets) / output.shape[0]; //Objective 6bi\2
                t++;
                lr = initialLr / (1 + decay * epoch); //Objective 6bi\3
                //Objective 6bi\4
                grad3 = layer3.Backward(outputGrad, lr, t);
                grad2 = layer2.Backward(grad3, lr, t);
                grad1 = layer1.Backward(grad2, lr, t);
            }
        }
    }
    #endregion
    #region PPOModel
    //OOP
    public class PPOModel : CreateModel
    {
        public PPOModel(int inputSize, int outputSize, int[] hiddenSizes) : base(inputSize, outputSize, hiddenSizes)
        {
            base.inputSize = inputSize;
            base.outputSize = outputSize;
            base.hiddenSizes = hiddenSizes;
        }
    }
    #endregion
    #region DQN
    //OOP
    public class DQNModel : CreateModel
    {
        public DQNModel(int inputSize, int outputSize, int[] hiddenSizes) : base(inputSize, outputSize, hiddenSizes)
        {
            base.inputSize = inputSize;
            base.outputSize = outputSize;
            base.hiddenSizes = hiddenSizes;
        }

    }

    #endregion
    #region TFPPO

    public class TFPPOModel : Module
    {
        public int inDim;
        public int outDim;
        public TorchSharp.Modules.Linear layer1, layer2, layer3;
        public TFPPOModel(int inDimInputs, int outDimInputs, int[] hiddenDimInputs) : base("TFPPOModel"){
            inDim = inDimInputs;
            outDim = outDimInputs;
            layer1 = torch.nn.Linear(inDim, hiddenDimInputs[0]);
            layer2 = torch.nn.Linear(hiddenDimInputs[0], hiddenDimInputs[1]);
            layer3 = torch.nn.Linear(hiddenDimInputs[1], outDim);
        }
        public NDArray forward(NDArray inputs){
            //convert to tensor for nn
            torch.Tensor inputsTens = torch.tensor((long)inputs, torch.ScalarType.Float64, null, false);
            //forward through nn
            var activation1 = functional.relu(layer1.forward(inputsTens));
            var activation2 = functional.relu(layer2.forward(activation1));
            var output = layer3.forward(activation2);
            //convert to float
            float[] netArray = output.data<float>().ToArray();
            //convert to ndarray and return
            return np.array(netArray);
        }
    }

    #endregion

}
