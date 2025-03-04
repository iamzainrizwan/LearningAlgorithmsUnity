//Objective 3
//Class
//Neural Network
//Procedures & Functions
//DQN
//Bellman-Optimality
//Epsilon-Greedy

//neural network
using FCNN;

//math
using NumSharp;

//system usings
using System.Collections.Generic;
using UnityEngine;

//OOP
using Hyperparamters;

namespace DQNImplementaion
{
    public class DQNHyperparameters : Hyperparameters
    {
        public float lr = 0.001f; //learning rate
        public float gamma = 0.99f; //discounted fctor
        public float explorationProbInit = 1.0f; //starting exploration probability
        public float explorationProbDecay = 0.005f; //exploration probability decay
        public int[] hiddenLayersSizes = new int[] { 24, 24 }; //sizes of hidden layers of network
        public int batchSize = 32; //size of experiences we sample to train the neural network
        public int memoryBufferSize = 2000; //size of memory buffer (duh)
    }

    public class DQN
    {

        public class Epsiode
        {
            public NDArray currentState;
            public bool[] action;
            public float reward;
            public NDArray nextState;
            public bool done;
        }

        public int inputs { get; set; }
        public int outputs { get; set; }
        public DQNHyperparameters hyperparameters { get; set; }
        public DQNModel model;
        public float explorationProb;
        public System.Random rnd = new();
        public List<Epsiode> memoryBuffer = new();

        public DQN(DQNHyperparameters input_hyperparameters)
        {
            inputs = input_hyperparameters.obsDim;
            outputs = input_hyperparameters.actDim;
            hyperparameters = input_hyperparameters;
            model = new DQNModel(inputs, outputs, hyperparameters.hiddenLayersSizes);
            explorationProb = hyperparameters.explorationProbInit;
        }

        //Objective 3c
        //Neural Network
        public bool[] ComputeAction(NDArray state)
        {
            //we sample a variable uniformly over [0, 1]
            //if the variable is less than the exploration probability
            //  we chose an action randomly
            //else
            //  we forward the state through the network and choose the action with the highest Q-value

            bool[] outputsArray = new bool[5];
            for (int i = 0; i < 5; i++)
            {
                outputsArray[i] = false;
            }

            float prob = (float)rnd.NextDouble();
            if (prob < explorationProb) //Objective 3ci
            { 
                //choose random action
                outputsArray[rnd.Next(0, 5)] = true; //Objective 3ciii
            }
            else
            {
                Debug.Log("neural network step");
                //choose highest q-value
                NDArray qValues = model.Forward(state.reshape(1, hyperparameters.obsDim));
                int index = 0;
                for (int i = 0; i < qValues.shape[1]; i++)
                {
                    index = (qValues[0, i] > qValues[0, index]) ? i : index; //if value is greater, change index. otherwise, keep the same
                }
                outputsArray[index] = true; //Objective 3cii
            }

            return outputsArray;
        }

        //Objective 3diii
        //Epsilon-Greedy
        public void UpdateExplorationProbability()
        {
            //update exploration probability using epsilon-greedy algorithm
            explorationProb *= Mathf.Exp(-1 * hyperparameters.explorationProbDecay);
        }

        //Objective 3d
        //Bellman-Optimality
        //Neural Network
        public void Train()
        {
            //Objective 3di
            //choose set of indexes of episodes to be used
            NDArray inds = np.arange(memoryBuffer.Count);
            np.random.shuffle(inds);
            List<Epsiode> batchSample = new();

            //create sample from buffer
            foreach (int i in inds)
            {
                batchSample.Add(memoryBuffer[i]);
            }

            //Objective 3dii
            //iterate over selected episodes    
            foreach (Epsiode ep in batchSample)
            {
                //compute q-values of s_t
                NDArray qValueCurrentState = model.Forward(ep.currentState.reshape(1, hyperparameters.obsDim));
                //compute q-target using Bellman-Optimality equation
                float qValueTarget = ep.reward;
                if (!ep.done)
                {
                    qValueTarget += hyperparameters.gamma * np.max(model.Forward(ep.nextState.reshape(1, hyperparameters.obsDim)));
                }
                int index = 0;
                int i = 0;
                foreach (bool action in ep.action)
                {
                    index = (action) ? i : index; //if the action is the one taken, that is the index we need to update
                    i++; //otherwise, move on to next index
                }
                //update q value for action taken
                qValueCurrentState[0, index] = qValueTarget;

                //Objective 3dii
                //train the model
                model.Train(ep.currentState.reshape(1, hyperparameters.obsDim), qValueCurrentState.reshape(1, hyperparameters.actDim), 1, hyperparameters.lr, 0);

            }
        }
    }
}