//Objectives 2, 7, 8
//Procedures & Functions
//Unity 3D Simulation
//Unity UI System
//Arrays
//Dictionary
//Write to File

//unity/system usings
using UnityEngine;
using TMPro;
using System.Collections.Generic;

//OOP usings
using LearningAlgorithms;
using NEATImplementation;
using DQNImplementaion;
using PPOImplementation;
using System.IO;


public class GameManager : MonoBehaviour
{

    //assigns
    [Header("references")]
    public GameObject runnerObject;
    public Transform runnerFoot;
    public GameObject taggerObject;
    public Transform taggerFoot;
    public Transform runnerSpawn;
    public Transform taggerSpawn;
    public LayerMask groundedLayers;
    [Header("managers")]
    [Space]
    public SimStatsManager simStatsManager;
    public UInterfaceManager uIController;
    [Header("ui references")]
    [Space]

    //ui elements
    public TMP_Text roundText;
    public TMP_Text roundPausedText;
    public TMP_Text gameStatusText;
    public TMP_Text secondsText;
    [Header("player movement variables")]
    [Space]

    //player movement variables
    public float forceMultiplier = 100;
    public float jumpMultiplier = 0.04f;
    public float rotationMultiplier = 10;
    private bool isGrounded;
    [Header("player learning algorithms")]
    [Space]

    //player algorithms
    public LearningAlgorithm taggerAlgorithm;
    public LearningAlgorithm runnerAlgorithm;

    //internal variables

    //rounds
    private bool taggerWin = false;
    private bool runnerWin = false;
    private int currentRound = 0;
    private int currentSubRound = 0;

    //simulation
    private float maxDistance = 0;
    private float distance = 0;
    [SerializeField] private int currentFrame = 0;
    [Header("sim configs + general info")]
    [Space]
    public bool gameRunning = false;
    public bool roundPaused = false;
    public bool pausingEnabled = false;
    public bool waitingForNextRound = false;    
    public int pauseInterval = 0;
    public int totalRounds;
    public string filepath;

    //runner/tagger
    private Transform runnerTransform;
    private Transform taggerTransform;
    private Collider runnerCollider;
    private Collider taggerCollider;

    //ui
    private string lastRoundWinner = @"N\A";
    private int taggerOverallRoundsWon = 0;
    private int runnerOverallRoundsWon = 0;
    private int taggerSubRoundsWon = 0;
    private int runnerSubRoundsWon = 0;
    private int numSubRounds = 0;

    //stats
    private float taggerRoundsWon = 0;
    private float runnerRoundsWon = 0;
    private float avgDistance = 0;
    private bool isLearningDone = true;

    //stuff i wanna see ig
    [Header("reward functions")]
    [Space]
    [SerializeField] private float runnerReward;
    [SerializeField] private float taggerReward;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        //cap fps at my display fps to prevent unnecessary render

        Application.targetFrameRate = 240;
        //get components 
        runnerTransform = runnerObject.GetComponent<Transform>();
        taggerTransform = taggerObject.GetComponent<Transform>();
        runnerCollider = runnerObject.GetComponent<Collider>();
        taggerCollider = taggerObject.GetComponent<Collider>();

        //initialise rewards
        runnerReward = 0;
        taggerReward = 0;
    }

    //Objective 2a
    //Dictionary
    //Array
    public void InitaliseSimulation(string[] simulationParams, string[] taggerParams, string[] runnerParams)
    {
        //assign agents and params as needed
        Hyperparamters.Hyperparameters taggerHyperparameters;
        Hyperparamters.Hyperparameters runnerHyperparameters;

        //tagger hyperparameters
        switch (taggerParams[0])
        {
            case "NEAT":
                taggerHyperparameters = new NEATHyperparameters()
                {
                    deltaThreshold = float.Parse(taggerParams[1]),
                    maxFitnessHistory = int.Parse(taggerParams[2]),
                    populationSize = int.Parse(taggerParams[3]),
                    distanceWeights = new Dictionary<string, float>{
                        {"edge" , float.Parse(taggerParams[4])},
                        {"weight" , float.Parse(taggerParams[5])},
                        {"bias" , float.Parse(taggerParams[6])}
                    },
                    breedProbabilities = new Dictionary<string, float>{
                        {"asexual" , float.Parse(taggerParams[7])},
                        {"sexual" , float.Parse(taggerParams[8])}
                    },
                    mutationProbabilities = new Dictionary<string, float>{
                        {"node" , float.Parse(taggerParams[9])},
                        {"edge" , float.Parse(taggerParams[10])},
                        {"weight perturb" , float.Parse(taggerParams[11])},
                        {"weight set" , float.Parse(taggerParams[12])},
                        {"bias perturb", float.Parse(taggerParams[13])},
                        {"bias set", float.Parse(taggerParams[14])}
                    },
                };
                numSubRounds = int.Parse(taggerParams[3]);
                taggerAlgorithm = new NEATAlgorithm((NEATHyperparameters)taggerHyperparameters);
                break;
            case "DQN":
                taggerHyperparameters = new DQNHyperparameters()
                {
                    lr = float.Parse(taggerParams[1]),
                    explorationProbInit = float.Parse(taggerParams[2]),
                    explorationProbDecay = float.Parse(taggerParams[3]),
                    batchSize = int.Parse(taggerParams[4]),
                    memoryBufferSize = int.Parse(taggerParams[5]),
                    hiddenLayersSizes = new int[2] { int.Parse(taggerParams[6]), int.Parse(taggerParams[7]) },
                    actDim = 5,
                    obsDim = 7
                };

                taggerAlgorithm = new DQNAlgorithm((DQNHyperparameters)taggerHyperparameters);
                break;
            case "PPO":
                taggerHyperparameters = new PPOHyperparameters()
                {
                    maxEpsisodesInBatch = int.Parse(taggerParams[1]),
                    timestepsPerBatch = int.Parse(taggerParams[2]),
                    nUpdatesPerIteration = int.Parse(taggerParams[3]),
                    lr = float.Parse(taggerParams[4]),
                    gamma = float.Parse(taggerParams[5]),
                    lam = float.Parse(taggerParams[6]),
                    clip = float.Parse(taggerParams[7]),
                    maxGradNorm = float.Parse(taggerParams[8]),
                    actorNetworkSizes = new int[2] { int.Parse(taggerParams[9]), int.Parse(taggerParams[10]) },
                    criticNetworkSizes = new int[2] { int.Parse(taggerParams[11]), int.Parse(taggerParams[12]) },
                    actDim = 5,
                    obsDim = 7
                };

                taggerAlgorithm = new PPOAlgorithm((PPOHyperparameters)taggerHyperparameters);
                break;
        }

        //runner hyperparameters
        switch (runnerParams[0])
        {
            case "NEAT":
                runnerHyperparameters = new NEATHyperparameters()
                {
                    deltaThreshold = float.Parse(runnerParams[1]),
                    maxFitnessHistory = int.Parse(runnerParams[2]),
                    populationSize = int.Parse(runnerParams[3]),
                    distanceWeights = new Dictionary<string, float>{
                        {"edge" , float.Parse(runnerParams[4])},
                        {"weight" , float.Parse(runnerParams[5])},
                        {"bias" , float.Parse(runnerParams[6])}
                    },
                    breedProbabilities = new Dictionary<string, float>{
                        {"asexual" , float.Parse(runnerParams[7])},
                        {"sexual" , float.Parse(runnerParams[8])}
                    },
                    mutationProbabilities = new Dictionary<string, float>{
                        {"node" , float.Parse(runnerParams[9])},
                        {"edge" , float.Parse(runnerParams[10])},
                        {"weight perturb" , float.Parse(runnerParams[11])},
                        {"weight set" , float.Parse(runnerParams[12])},
                        {"bias perturb", float.Parse(runnerParams[13])},
                        {"bias set", float.Parse(runnerParams[14])}
                    },
                };
                //update number of subrounds if greater population size
                numSubRounds = (int.Parse(runnerParams[3]) > numSubRounds) ? int.Parse(runnerParams[3]) : numSubRounds;
                runnerAlgorithm = new NEATAlgorithm((NEATHyperparameters)runnerHyperparameters);
                break;
            case "DQN":
                runnerHyperparameters = new DQNHyperparameters()
                {
                    lr = float.Parse(runnerParams[1]),
                    explorationProbInit = float.Parse(runnerParams[2]),
                    explorationProbDecay = float.Parse(runnerParams[3]),
                    batchSize = int.Parse(runnerParams[4]),
                    memoryBufferSize = int.Parse(runnerParams[5]),
                    hiddenLayersSizes = new int[2] { int.Parse(runnerParams[6]), int.Parse(runnerParams[7]) },
                    actDim = 5,
                    obsDim = 7
                };
                numSubRounds = 0;
                runnerAlgorithm = new DQNAlgorithm((DQNHyperparameters)runnerHyperparameters);
                break;
            case "PPO":
                runnerHyperparameters = new PPOHyperparameters()
                {
                    maxEpsisodesInBatch = int.Parse(runnerParams[1]),
                    timestepsPerBatch = int.Parse(runnerParams[2]),
                    nUpdatesPerIteration = int.Parse(runnerParams[3]),
                    lr = float.Parse(runnerParams[4]),
                    gamma = float.Parse(runnerParams[5]),
                    lam = float.Parse(runnerParams[6]),
                    clip = float.Parse(runnerParams[7]),
                    maxGradNorm = float.Parse(runnerParams[8]),
                    actorNetworkSizes = new int[2] { int.Parse(runnerParams[9]), int.Parse(runnerParams[10]) },
                    criticNetworkSizes = new int[2] { int.Parse(runnerParams[11]), int.Parse(runnerParams[12]) },
                    actDim = 5,
                    obsDim = 7
                };

                runnerAlgorithm = new PPOAlgorithm((PPOHyperparameters)runnerHyperparameters);
                break;
        }

        //simulation parameters
        totalRounds = int.Parse(simulationParams[1]);
        pausingEnabled = simulationParams[4] == "true";
        if (pausingEnabled)
        {
            pauseInterval = int.Parse(simulationParams[2]);
            roundText.text = $"Current Round: 0" + @"\" + $"Next Pause: {pauseInterval}";
        }
        else
        {
            roundText.text = $"Current Round: 0";
        }
        filepath = Path.Combine(simulationParams[3], "stats.txt");
    }

    //Objective 7aiii
    void AgentsLearn()
    {
        //taggerAlgorithm.Learn(currentRound);
        //runnerAlgorithm.Learn(currentRound);
        isLearningDone = true;
        gameStatusText.text = "";
    }

    //Objective 7a, 7b, 7c, 7d
    public void UpdateRound()
    {
        //increment frame
        currentFrame++;

        //update time text
        secondsText.text = $"Time: {(currentFrame / 30f).ToString("F1")} s";

        //rewards
        taggerReward = CalculateTaggerReward();
        runnerReward = CalculateRunnerReward();

        //have agents take action
        var taggerSimInput = TakeTaggerInput();
        var runnerSimInput = TakeRunnerInput();
        var taggerOutput = taggerAlgorithm.GetOutput(taggerSimInput, false, taggerReward);
        var runnerOutput = runnerAlgorithm.GetOutput(runnerSimInput, false, taggerReward);
        //OutputsToProgram(taggerOutput, taggerObject, taggerFoot);
        //OutputsToProgram(runnerOutput, runnerObject, runnerFoot);

        //update average distance
        avgDistance += (distance - avgDistance) / (currentFrame + 1e-10f);

        //update stats panel if active
        if (simStatsManager.isActive)
        {
            simStatsManager.taggerSimInput = taggerSimInput;
            simStatsManager.runnerSimInput = runnerSimInput;
            simStatsManager.taggerSimOutput = taggerOutput;
            simStatsManager.runnerSimOutput = runnerOutput;
            simStatsManager.taggerReward = taggerReward;
            simStatsManager.runnerReward = runnerReward;
            simStatsManager.taggerRoundsWon = taggerOverallRoundsWon;
            simStatsManager.runnerRoundsWon = runnerOverallRoundsWon;
            simStatsManager.lastRoundWinnerName = lastRoundWinner;
            simStatsManager.UpdateStatsPanel();
        }

        //check if round is over (if a given agent has won)
        if (currentFrame >= 30 * 30)
        { //time up, runner has got away //Objective 7bi
            Debug.Log("RUNNER WIN");
            taggerWin = false;
            lastRoundWinner = "Runner";
            runnerSubRoundsWon++;
            runnerOverallRoundsWon++;
            gameStatusText.text = "ROUND PAUSED FOR TRAINING";
            runnerWin = true;
        }
        else if (runnerCollider.bounds.Intersects(taggerCollider.bounds))
        { //collision, tagger has caught runner //Objective 7bii
            runnerWin = false;
            lastRoundWinner = "Tagger";
            taggerOverallRoundsWon++;
            taggerSubRoundsWon++;
            gameStatusText.text = "ROUND PAUSED FOR TRAINING";
            taggerWin = true;
        }

        //round over operations
        if (taggerWin || runnerWin)
        {
            ResetRound();
            currentFrame = 0;
            if (currentSubRound == numSubRounds) //note that this defaults to 0 for no NEAT
            {
                if (taggerWin) { taggerRoundsWon++; }
                if (runnerWin) { runnerRoundsWon++; }
                StoreStats(taggerSubRoundsWon, runnerSubRoundsWon);
                currentSubRound = 0;
                taggerSubRoundsWon = 0;
                runnerSubRoundsWon = 0;
                currentRound++;
                isLearningDone = false;
            }
            else
            {
                currentSubRound++;
                if (taggerWin) { taggerSubRoundsWon++; }
                if (runnerWin) { runnerSubRoundsWon++; }
                if (taggerAlgorithm is NEATAlgorithm taggerNEAT)
                {
                    taggerNEAT.IncrementAgents();
                }
                if (runnerAlgorithm is NEATAlgorithm runnerNEAT)
                {
                    runnerNEAT.IncrementAgents();
                }
            }

            Time.timeScale = 1;
        }
    }

    void FixedUpdate()
    {
        distance = Vector3.Distance(taggerTransform.position, runnerTransform.position);
        maxDistance = (distance > maxDistance) ? distance : maxDistance;
        //skip all if the game isnt running
        if (!gameRunning)
        {
            return;
        }
        else if (!isLearningDone)
        {
            AgentsLearn();
            ResetPositions();
        }
        else if (waitingForNextRound){
            return;
        }
        else
        { //if game is running, run the round
            UpdateRound();
            //end sim if at end
            if (currentRound >= totalRounds)
            {
                gameRunning = false;
            }

        }
    }

    //Objective 7ai
    float CalculateRunnerReward()
    {
        float reward = 0;
        reward += distance / maxDistance; //encourage staying away
        reward += currentFrame * 2; //encourage staying alive
        return reward;
    }

    //Objective 7ai
    float CalculateTaggerReward()
    {
        float reward = 0;
        reward -= distance / maxDistance; //encourage closing distance
        reward -= currentFrame * 2; //encourage being quick
        return reward;
    }


    //Objective 7ai
    float[] TakeRunnerInput()
    {
        float[] inputs = new float[7];
        Vector2 forward = new(1, 0);

        Vector2 runnerForward = new(runnerTransform.forward.x, runnerTransform.forward.z);
        Vector2 taggerToRunner = new(taggerTransform.position.x - runnerTransform.position.x, taggerTransform.position.z - runnerTransform.position.z);


        /* xpos */
        inputs[0] = runnerTransform.position.x;
        /* ypos */
        inputs[1] = runnerTransform.position.y;
        /* zpos */
        inputs[2] = runnerTransform.position.z;
        /* orientation wrt forward */
        inputs[3] = Vector2.Angle(forward, runnerForward);
        /* distance to tagger */
        inputs[4] = Vector3.Distance(taggerTransform.position, runnerTransform.position);
        /* orientation wrt tagger */
        inputs[5] = Vector2.Angle(runnerForward, taggerToRunner);


        //isGrounded
        if (Physics.CheckSphere(runnerFoot.position, 0.2f, groundedLayers))
        {
            inputs[6] = 1;
        }
        else
        {
            inputs[6] = 0;
        }

        return inputs;
    }

    //Objective 7ai
    float[] TakeTaggerInput()
    {
        float[] inputs = new float[7];
        Vector2 forward = new(1, 0);
        Vector2 taggerForward = new(taggerTransform.forward.x, taggerTransform.forward.z);
        Vector2 taggerToRunner = new(runnerTransform.position.x - taggerTransform.position.x, runnerTransform.position.z - taggerTransform.position.z);

        /* xpos */
        inputs[0] = taggerTransform.position.x;
        /* ypos */
        inputs[1] = taggerTransform.position.y;
        /* zpos */
        inputs[2] = taggerTransform.position.z;
        /* orientation wrt forward */
        inputs[3] = Vector2.Angle(forward, taggerForward);
        /* distance to tagger */
        inputs[4] = Vector3.Distance(taggerTransform.position, runnerTransform.position);
        /* orientation wrt tagger */
        inputs[5] = Vector2.Angle(taggerForward, taggerToRunner);

        //isGrounded
        if (Physics.CheckSphere(taggerFoot.position, 0.2f, groundedLayers))
        {
            inputs[6] = 1;
        }
        else
        {
            inputs[6] = 0;
        }

        return inputs;
    }

    //Objective 7c
    void ResetRound()
    {
        //reset frame counter
        currentFrame = 0;

        //reset positions
        runnerObject.transform.position = runnerSpawn.position;
        taggerObject.transform.position = taggerSpawn.position;

        //reset velocities
        runnerObject.GetComponent<Rigidbody>().linearVelocity = Vector3.zero;
        taggerObject.GetComponent<Rigidbody>().linearVelocity = Vector3.zero;
        runnerObject.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        taggerObject.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

        //reset round winners
        taggerWin = false;
        runnerWin = false;

        //reset avg distance
        avgDistance = 0;

        //set round info text
        if (pausingEnabled)
        {
            roundText.text = $"Current Round: {currentRound+1}" + @"\" + $"Next Pause: {currentRound + (pauseInterval - (currentRound % pauseInterval))}";
            if ((currentRound + 1 == (currentRound + (pauseInterval - (currentRound % pauseInterval)))) && currentRound != 0)
            {
                roundText.text = $"Current Round: {currentRound+1}" + @"\" + $"Next Pause: {currentRound + (pauseInterval - (currentRound % pauseInterval))}";
                roundPaused = true;
            }
        }
        else
        {
            roundText.text = $"Current Round: {currentRound}";
        }

        //check if round is to be paused. if true, then pause round
        if (roundPaused)
        {
            waitingForNextRound = true;
            roundPausedText.text = "Round Paused";
        }
    }

    public void ResetPositions(){
        runnerObject.transform.position = runnerSpawn.position;
        taggerObject.transform.position = taggerSpawn.position;
    }

    //Objective 7aii
    void OutputsToProgram(bool[] movement, GameObject movingObject, Transform footTransform)
    {
        //get components
        Rigidbody rb = movingObject.GetComponent<Rigidbody>();
        Transform tf = movingObject.transform;

        //grounded check
        isGrounded = Physics.CheckSphere(footTransform.position, 0.2f, groundedLayers);
        if (movement[0])
        {
            rb.AddForce(tf.forward * forceMultiplier); //forward
        }

        if (movement[1])
        {
            rb.AddForce(tf.forward * -forceMultiplier); //backward
        }

        if (movement[2])
        {
            tf.Rotate(0, rotationMultiplier, 0); //left
        }

        if (movement[3])
        {
            tf.Rotate(0, -rotationMultiplier, 0); //right
        }

        if (movement[4] && isGrounded)
        {
            rb.AddForce(tf.up * jumpMultiplier); //jump
        }

    }

    //Objective 8ai
    //Write to File
    void StoreStats(int taggerRounds, int runnerRounds)
    {
        runnerRoundsWon += (float)runnerRounds / (taggerRounds + runnerRounds);
        taggerRoundsWon += (float)taggerRounds / (taggerRounds + runnerRounds);
        string statsString = $"{currentRound},{runnerReward},{taggerReward},{avgDistance},{runnerRoundsWon},{taggerRoundsWon}";
        using StreamWriter sw = new(filepath, true);
        sw.WriteLine(statsString);
    }
}
