//Objective 4
//PPO
//Generalised Advantage Estimation
//Multivariate Normal Distribution
//Lists
//Procedures & Functions
//OOP
//Neural Network

//neural network
using FCNN;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using TorchSharp.Modules;

//math
using Accord.Statistics.Distributions.Multivariate;
using NumSharp;
using Accord.Math;

//system usings
using System.Collections.Generic;
using System;

//OOP
using Hyperparamters;


namespace PPOImplementation
{
    public class PPOHyperparameters : Hyperparameters
    {
        public int timestepsPerBatch = 4800; //number of timesteps to run per batch
        public int maxTimestepsPerEpisode = 30 * 30; //maximum number of timesteps per episode
        public int nUpdatesPerIteration = 5; //number of times to update networks per iteration
        public float lr = 0.0005f; //learning rate of actor optimiser
        public float gamma = 0.95f; //discount factor to be applied when calculating rewards to go
        public float lam = 0.98f; //lambda paramter for Generalised Advantage Estimation
        public float clip = 0.2f; //recommended 0.2. helps define threshold to clip ratio during surrogate loss calculation
        public int numMinibatches = 6; //number of mini-batches for mini-batch update
        public float entCoef = 0; //entropy coeffecient for entropy regularisation
        public float maxGradNorm = 0.5f; //gradient clipping threshold
        //sizes for neural networks
        public int[] actorNetworkSizes = new int[] { 32, 32 };
        public int[] criticNetworkSizes = new int[] { 32, 32 };
        public int totalTimesteps = 0; //total timesteps for the whole program to run for
        public int maxEpsisodesInBatch = 5; //max episodes in a batch

    }

    public class PPO
    {

        public class Epsiode
        {
            public List<NDArray> epObs, epActs, epNextObs;
            public List<float> epLogProbs, epRewards, epVals, epDones;
            public int epLen;
            //Objective 4b
            public Epsiode()
            {
                epObs = new(); 
                epActs = new();
                epLogProbs = new();
                epRewards = new();
                epVals = new();
                epDones = new();
                epNextObs = new();
                epLen = 0;
            }
        }

        public int inputs { get; set; }
        public int outputs { get; set; }
        public PPOHyperparameters hyperparameters { get; set; }
        public PPOModel actor;
        public PPOModel critic;
        public TFPPOModel actorTF;
        public TFPPOModel criticTF;
        public Adam actorOptim;
        public Adam criticOptim;
        public System.Random rnd = new();
        public double[,] covMat;
        public float covMatDet;
        public List<Epsiode> batch;
        public PPO(PPOHyperparameters input_hyperparameters)
        {
            hyperparameters = input_hyperparameters;
            inputs = hyperparameters.obsDim;
            outputs = hyperparameters.actDim;
            covMat = new double[outputs, outputs];

            for (int i = 0; i < outputs; i++)
            {
                for (int j = 0; j < outputs; j++)
                {
                    covMat[i, j] = 0.5;
                }
            }

            covMatDet = (float)covMat.Determinant();
            actorTF = new TFPPOModel(inputs, outputs, hyperparameters.actorNetworkSizes);
            criticTF = new TFPPOModel(inputs, 1, hyperparameters.criticNetworkSizes);
            actorOptim = Adam(actorTF.parameters(), hyperparameters.lr);
            criticOptim = Adam(criticTF.parameters(), hyperparameters.lr);

            batch = new();
        }

        //Objective 4c
        //Neural Network
        //Multivariate Normal Distribution
        public (NDArray, NDArray) Forward(NDArray state, Device device)
        {
            using var scope = torch.no_grad();
            //get actor output
            NDArray mean = actorTF.forward(state); //Objective 4bi
            //convert mean to array for distribution
            double[] meanArray = new double[mean.shape[0]];
            for (int i = 0; i < mean.shape[0]; i++)
            {
                meanArray[i] = mean[i];
            }

            //Objective 4bii
            var dist = new MultivariateNormalDistribution(meanArray, covMat);
            var action = dist.Generate();

            //Objective 4biii
            //calculate log probs of actions
            var logProbs = dist.LogProbabilityDensityFunction(action);

            //output as NDArrays
            return (np.array(action), np.array(logProbs));
        }

        //Objective 4diii
        //Generalised Advantage Estimation 
        public List<float> CalculateGAE()
        {
            List<float> batchRewards = new();
            foreach (Epsiode epsiode in batch)
            {
                List<float> advantages = new(); //stores advantages for current episode
                float lastAdvantage = 0;
                float delta;

                for (int t = epsiode.epLen; t > 0; t--)
                {
                    if (t + 1 < epsiode.epLen)
                    {
                        //calculate temporal difference error for the current timestep
                        delta = epsiode.epRewards[t] + hyperparameters.gamma * epsiode.epVals[t + 1] * (1 - epsiode.epDones[t + 1]) - epsiode.epVals[t];
                    }
                    else
                    {
                        //special case at last timestep
                        delta = epsiode.epRewards[t] - epsiode.epVals[t];
                    }

                    //calculate Generalised Advantage Estimation (GAE) for current timestep
                    float advantage = delta + hyperparameters.gamma * hyperparameters.lam * (1 - epsiode.epDones[t]) * lastAdvantage;
                    lastAdvantage = advantage; //update lastAdvantage
                    advantages.Insert(0, advantage); //insert at beginning of list
                }

                batchRewards.AddRange(advantages); //add onto end of already existing list
            }

            return batchRewards;
        }

        //Objective 4a
        public void AddEpisodeToBatch(Epsiode ep)
        {
            if (batch.Count >= hyperparameters.maxEpsisodesInBatch)
            {
                batch.RemoveAt(0);
            }

            batch.Add(ep);
        }

        public float CriticForward(NDArray state){
            return criticTF.forward(state);
        }

        //Objective 4d
        //Neural Network
        public void Learn(int currentT)
        {
            //calculate advantage using GAE
            var A_k = CalculateGAE(); //Objective 4di

            //Objectives 4dii, 4diii

            List<float> V = new();
            foreach (Epsiode epsiode in batch)
            {
                foreach (NDArray obs in epsiode.epObs)
                {
                    V.Add(criticTF.forward(obs));
                }
            }

            List<float> batchRTGs = new();
            for (int i = 0; i < V.Count; i++)
            {
                batchRTGs.Add(A_k[i] + V[i]);
            }

            //normalise advantages to decrease variance of advantages 
            //+ makes convergence more stable and faster
            float sigx2 = 0;
            float sigX = 0;
            int n = A_k.Count;

            for (int i = 0; i < n; i++)
            {
                sigx2 += (float)Math.Pow((double)A_k[i], 2);
                sigX += A_k[i];
            }

            float A_k_mean = sigX / n;
            float A_k_std = (float)Math.Sqrt((sigx2 - n * Math.Pow((double)A_k_mean, 2)) / (n - 1 + 1e-10f));

            for (int i = 0; i < A_k.Count; i++)
            {
                A_k[i] = (A_k[i] - A_k_mean) / (A_k_std + 1e-10f);
            }

            //calculate step for loop updating networks
            int step = 0;
            foreach (var episode in batch)
            {
                step += episode.epLen;
            }


            //update the network for a given number of epochs
            NDArray inds = np.arange(step);
            int minibatchSize = step / hyperparameters.numMinibatches;
            for (int i = 0; i < hyperparameters.nUpdatesPerIteration; i++)
            {
                //learning rate annealing
                float frac = (float)(currentT - 1) / hyperparameters.totalTimesteps;
                float newLr = hyperparameters.lr * (1 - frac);

                //make sure learning rate does not go below 0
                newLr = (newLr > 0) ? newLr : 0;

                //set learning rates
                actorOptim = Adam(actorTF.parameters(), newLr);
                criticOptim = Adam(criticTF.parameters(), newLr);

                np.random.shuffle(inds); //shuffling

                //mini batch update
                for (int j = 0; j < step; j += minibatchSize)
                {
                    //init
                    List<NDArray> minibatchObs = new();
                    List<NDArray> minibatchActs = new();
                    List<float> minibatchLogProbs = new();
                    List<float> minibatchAdvantage = new();
                    List<float> minibatchRTGs = new();

                    //define what parts of ind to use
                    int end = j + minibatchSize; //want to use indexes from j to end
                    List<int> idx = new();

                    //extract desired indexes
                    for (int k = 0; k < end; k++)
                    {
                        idx.Add(inds[k]);
                    }

                    //extract observation data for given index
                    foreach (int k in inds)
                    {
                        //get episode & index to extract data
                        (Epsiode, int) output = FindEpsiodeInstanceFromIndex(k);
                        Epsiode miniEp = output.Item1;
                        int index = output.Item2;
                        //extract data for a given minibatch
                        minibatchObs.Add(miniEp.epObs[index]);
                        minibatchActs.Add(miniEp.epActs[index]);
                        minibatchLogProbs.Add(miniEp.epLogProbs[index]);
                        minibatchAdvantage.Add(A_k[k]);
                        minibatchRTGs.Add(batchRTGs[k]);
                    }

                    //calculate V_phi, pi_theta(a_t | s_t) and entropy
                    var eval = Evaluate(minibatchObs, minibatchActs);
                    V = eval.Item1;
                    var currentLogProbs = eval.Item2;
                    var entropy = eval.Item3;

                    //calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    List<float> logRatios = new();
                    List<float> ratios = new();
                    List<float> approxKlList = new();
                    for (int k = 0; k < currentLogProbs.Count; k++)
                    {
                        //note that subtracting log == dividiing
                        logRatios.Add(currentLogProbs[k] - minibatchLogProbs[k]);
                        ratios.Add(MathF.Exp(logRatios[k]));
                        approxKlList.Add(ratios[k] - 1 - logRatios[k]);
                    }

                    //calculate surrogate losses
                    List<float> surr1 = new();
                    List<float> surr2 = new();
                    for (int k = 0; k < ratios.Count; k++)
                    {
                        surr1.Add(ratios[k] * minibatchAdvantage[k]);
                        surr2.Add((float)(clamp(ratios[k], 1 - hyperparameters.clip, 1 + hyperparameters.clip) * minibatchAdvantage[k]));
                    }
                    Tensor surr1T = tensor(surr1, ScalarType.Float64, null, false);
                    Tensor surr2T = tensor(surr2, ScalarType.Float64, null, false);
                    Tensor actorLosses = (-min(surr1T, surr2T)).mean();

                    //entropy calculation
                    float total = 0;
                    n = 0;
                    foreach (float val in entropy)
                    {
                        total += val;
                        n++;
                    }

                    //entropy regularisation
                    float entropyLoss = total / n;
                    actorLosses -= hyperparameters.entCoef * entropyLoss;
                    //calculate gradients and perform backward propogation for actor network
                    actorOptim.zero_grad();
                    actorLosses.backward(); //Objective 4dii
                    torch.nn.utils.clip_grad_norm_(actorTF.parameters(), hyperparameters.maxGradNorm); //gradient clipping
                    actorOptim.step();

                    //calculate MSE between V and minibatchRTGs
                    float cumMSE = 0;
                    n = 0;
                    for (int k = 0; k < V.Count; k++)
                    {
                        cumMSE = MathF.Pow(V[k] - minibatchRTGs[k], 2);
                        n++;
                    }
                    Tensor criticLosses = tensor(cumMSE / n, ScalarType.Float64, null, false);

                    //calculate gradients and perform backward propogation for critic network
                    criticOptim.zero_grad();
                    criticLosses.backward(); //Objective 4diii
                    torch.nn.utils.clip_grad_norm_(criticTF.parameters(), hyperparameters.maxGradNorm); //gradient clipping
                    criticOptim.step();

                }

            }

        }

        public (Epsiode, int) FindEpsiodeInstanceFromIndex(int index)
        {

            //create set of lengths 
            List<int> lens = new();
            foreach (var ep in batch)
            {
                lens.Add(ep.epLen);
            }

            //create set of cumulative lengths
            List<int> cumLens = new();
            int currentTotal = 0;
            foreach (int num in lens)
            {
                currentTotal += num;
                cumLens.Add(currentTotal);
            }

            //find episode index
            int epIndex = 0;
            while (cumLens[epIndex] < index)
            {
                epIndex++;
            }
            //find observation index within given episode
            int obsIndex = index - (epIndex > 0 ? cumLens[epIndex - 1] : 0); //if at first episode, should be 0

            //return values
            return (batch[epIndex], obsIndex);
        }

        //Multivariate Normal Distribution
        public (List<float>, List<float>, List<float>) Evaluate(List<NDArray> miniObs, List<NDArray> miniActs)
        {
            //Estimate the values of each observation, and the log probs of each action.

            //query critic network for a value V for each miniObs
            List<float> V_ = new();
            List<float> logProbs = new();
            List<float> entropy = new();

            for (int i = 0; i < miniObs.Count; i++)
            {

                //query critic network for a value for given obs and add to list
                V_.Add(criticTF.forward(miniObs[i]));

                //query actor network for mean action for given obs
                NDArray meanND = actorTF.forward(miniObs[i]);
                double[] mean = new double[meanND.size];
                for (int j = 0; j < meanND.size; j++)
                {
                    mean[j] = meanND[j];
                }

                //create distribution
                var dist = new MultivariateNormalDistribution(mean, covMat);

                //query for log probs and add to list
                logProbs.Add((float)dist.LogProbabilityDensityFunction(miniActs[i]));

                //calculate entropy and add to list
                float k = dist.Dimension;
                entropy.Add((float)(0.5 * Math.Log(covMatDet) + 0.5 * k * (1 + Math.Log(2 + Math.PI))));
            }

            return (V_, logProbs, entropy);
        }
    }
}