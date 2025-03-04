//Objectives 3, 4, 5
//DQN
//PPO
//NEAT
//OOP
//Polymorphism
//Inheritance
//Procedures & Functions
//Lists

//MY IMPLEMENTATIONS!!
using Hyperparamters;
using DQNImplementaion;
using PPOImplementation;
using NEATImplementation;

//math & conversions
using NumSharp;

namespace Hyperparamters
{
    public abstract class Hyperparameters
    {
        public int obsDim;
        public int actDim;
        public float activationThreshold;
    }
}

namespace LearningAlgorithms
{

    public abstract class LearningAlgorithm
    {
        public Hyperparameters algorithmParam;
        public string algorithmID;
        public abstract bool[] GetOutput(float[] state, bool done, float reward);
        public abstract void Learn(int currentRound);
    }
    #region DQN
    public class DQNAlgorithm : LearningAlgorithm
    {
        //inits
        public DQN DQNAgent;
        public DQNAlgorithm(DQNHyperparameters inputHyperparameters) {
            algorithmParam = inputHyperparameters;
            algorithmID = "DQN";
            DQNAgent = new DQN(inputHyperparameters);
        }
        //Objective 3a
        //Objective 3b
        //get dqn action
        public override bool[] GetOutput(float[] state, bool done, float reward){
            //cast param to DQN type to check if memory buffer is too large
            if (algorithmParam is DQNHyperparameters algorithmDQNParam && DQNAgent.memoryBuffer.Count >= algorithmDQNParam.memoryBufferSize){
                DQNAgent.memoryBuffer.RemoveAt(0);
            }

            //get outputs
            var algorithmsOutputs = DQNAgent.ComputeAction(np.array(state));
            
            //create new episode and store
            DQN.Epsiode currentEpsiode = new()
            {
                currentState = np.array(state),
                action = algorithmsOutputs,
                reward = reward
            };
            DQNAgent.memoryBuffer.Add(currentEpsiode);

            //edit past episode
            DQNAgent.memoryBuffer[^1].nextState = np.array(state);
            return algorithmsOutputs;
        }

        public override void Learn(int currentRound)
        {
            DQNAgent.Train();
            DQNAgent.UpdateExplorationProbability();
        }
    }
    #endregion
    #region NEAT
    public class NEATAlgorithm : LearningAlgorithm
    {
        public Brain NEATBrain;
        public NEATAlgorithm(NEATHyperparameters inputHyperparameters){
            algorithmParam = inputHyperparameters;
            algorithmID = "NEAT";
            NEATBrain = new(inputHyperparameters);
        }

        //Objective 5a
        public override bool[] GetOutput(float[] state, bool done, float reward){
            //get output from selected member of selected species
            var output = NEATBrain.species[NEATBrain.currentSpecies].members[NEATBrain.currentGenome].Forward(state);
            
            //convert to bool using activation threshold & return
            bool[] outputBool = new bool[5];
            for(int i = 0; i < 5; i ++){
                outputBool[i] = output[i] > algorithmParam.activationThreshold;
            }
            return outputBool;
        }

        public override void Learn(int currentRound)
        {
            NEATBrain.Evolve();
        }

        public void IncrementAgents(){
            NEATBrain.NextIteration();
        }
    }
    #endregion
    
    #region PPO
    public class PPOAlgorithm : LearningAlgorithm
    {
        public PPO PPOAgent;
        public PPO.Epsiode currentEpisode;
        public PPOAlgorithm (PPOHyperparameters inputHyperparameters){
            algorithmParam = inputHyperparameters;
            algorithmID = "PPO";
            PPOAgent = new PPO (inputHyperparameters);
        }

        

        //Objective 4a
        //Objective 4b
        public override bool[] GetOutput(float[] state, bool done, float reward)
        {
            //get output from PPO
            var output = PPOAgent.Forward(np.array(state), null);

            //convert to bool
            bool[] actionOutput = new bool[5];
            int i = 0;
            foreach (float outputValue in output.Item1){
                actionOutput[i] = outputValue > algorithmParam.activationThreshold;
            }

            //episode adds + small amount of admin req for them
            currentEpisode.epActs.Add(output.Item1);
            float doneFloat = (done) ? 1 : 0;
            currentEpisode.epDones.Add(doneFloat);
            currentEpisode.epLogProbs.Add(output.Item2);
            //replace the one before the current with the current state, and refill current with zero to be replaced
            currentEpisode.epNextObs[^1] = np.array(state);
            currentEpisode.epNextObs.Add(np.zeros(1));
            currentEpisode.epObs.Add(np.array(state));
            currentEpisode.epRewards.Add(reward);
            //epVals, epLen to be done in Learn()
            return actionOutput;
            
        }

        public override void Learn(int currentRound)
        {
            //calculate epLen and epObs
            currentEpisode.epLen = currentEpisode.epActs.Count;
            foreach(NDArray obs in currentEpisode.epObs){
                currentEpisode.epVals.Add(PPOAgent.CriticForward(obs));
            }

            //add finalised episode to batch
            PPOAgent.AddEpisodeToBatch(currentEpisode);

            //learn
            PPOAgent.Learn(currentRound);
        }
    }
    #endregion
}