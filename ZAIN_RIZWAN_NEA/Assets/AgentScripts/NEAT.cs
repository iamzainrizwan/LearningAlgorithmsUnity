//Objective 5
//OOP
//Classes
//Struct
//Enum
//Arrays
//Neural Networks
//Lists
//Dictionary
//Try/Catch
//NEAT
//Genomic Distance
//Genomic Crossover

//for activation functions
using FCNN;
using System.Collections.Generic;

//system usings
using System;
using System.Linq;

//for some reason the rest of the program cant find a genome without this
using static NEATImplementation.Genome;

//OOP
using Hyperparamters;

namespace NEATImplementation
{
    //Classes
    //genome - a single neural network to be a part of a specie
    public class Genome
    {
        //Structs
        //edge struct
        public struct Edge
        {
            public float weight;
            public bool enabled;
            public Edge(float weight_userinput)
            {
                weight = weight_userinput;
                enabled = true;
            }
            public void enable()
            {
                enabled = true;
            }
            public bool exists()
            {
                return true;
            }
        }

        //node struct
        public struct Node
        {
            public float output;
            public float bias;
            public Activation activation;
            public Node(Activation default_activation)
            {
                activation = default_activation;
                output = 0;
                bias = 0;
            }
        }

        public int inputs { get; }
        public int outputs { get; }
        public Activation default_activation { get; }
        public int unhidden, maxNode, fitness, adjustedFitness;
        public List<List<Edge>> edges;
        public List<Node> nodes;
        public Random rnd;

        public Genome(int inputs_userinput, int outputs_userinput, Activation default_activation_userinput)
        {
            inputs = inputs_userinput;
            outputs = outputs_userinput;
            default_activation = default_activation_userinput;
            unhidden = inputs + outputs;
            maxNode = inputs + outputs;
            fitness = 0;
            adjustedFitness = 0;
            edges = new List<List<Edge>>();
            nodes = new List<Node>();
            rnd = new Random();

            Console.WriteLine($"{inputs}, {outputs}, {maxNode}");
        }

        public void GenerateNetwork()
        {
            //generate the neural network of this genome with minimal initial topology
            //only input and output layers

            Node default_node = new();
            default_node.activation = default_activation;

            for (int i = 0; i < maxNode; i++)
            {
                nodes.Add(default_node);
                Console.WriteLine("Node added");
            }

            for (int i = 0; i < inputs; i++)
            {
                for (int j = inputs; j <= unhidden; j++)
                {
                    AddEdge(i, j, rnd.Next(-1, 1));
                    Console.WriteLine("Edge added");
                }
            }

            Console.WriteLine($"{edges.Capacity}, {nodes.Capacity}");
        }

        //Objective 5bii
        //Neural Network
        //Lists
        //Enum
        public float[] Forward(float[] programInputs) 
        {
            //loop through each input
            for (int i = 0; i < programInputs.Length; i++) 
            {
                //set node outputs to inputs
                var copy = nodes[i];
                copy.output = programInputs[i];
                nodes[i] = copy;
            }

            //keep track of which nodes are connected to which
            List<List<int>> from = new();
            //initalise 
            for (int i = 0; i < maxNode; i++)
            {
                from.Add(new List<int>());
            }

            //loop through every edge 
            for (int i = 0; i < edges.Count; i++)
            {
                for (int j = 0; j < edges[i].Count; j++)
                {
                    //if connection enabled, note source index
                    if (edges[i][j].enabled)
                    {
                        from[j].Add(i);
                    }
                }
            }

            //order nodes - first process hidden nodes, then input nodes
            var ordered_nodes = Enumerable.Range(unhidden, maxNode - unhidden).Concat(Enumerable.Range(inputs, unhidden - inputs));
            foreach (int j in ordered_nodes)
            {
                //weighted sum
                float ax = 0;
                foreach (int i in from[j])
                {
                    ax += edges[i][j].weight * nodes[i].output;
                }
                var currentnode = nodes[j];
                //compute final output using activation
                currentnode.output = currentnode.activation.DoActivation(ax + currentnode.bias);
            }

            //process and return
            float[] nnoutputs = new float[outputs];
            for (int i = 0; i < unhidden - inputs; i++)
            {
                nnoutputs[i] = nodes[i].output;
            }
            return nnoutputs;
        }

        //Objective 5dii
        //Dictionary
        //Lists
        public void Mutate(Dictionary<string, float> probabilities)
        {
            //randomly mutate genome to initiate variation

            if (IsDisabled())
            { //if everything is disabled, enable some
                AddEnabled();
            }

            var population = new List<string>(probabilities.Keys); //what happens to the population
            var weights = new List<float>(probabilities.Values); //weights of possibilities
            var choice = SelectRandomChoice(population, weights); //randomly choose a choice

            switch (choice) //perform choice
            {
                //Objective 5di\1\5a
                case "node":
                    AddNode();
                    break;

                //Objective 5di\1\5b
                case "edge":
                    (int, int) pair = RandomPair();
                    AddEdge(pair.Item1, pair.Item2, (float)rnd.NextDouble() * 2 - 1); //creates edge with weight between -1 and 1
                    break;

                //Objective 5di\1\5c
                case "weight perturb":
                case "weight set":
                    ShiftWeight(choice);
                    break;

                //Objective 5di\1\5d
                case "bias perturb":
                case "bias set":
                    ShiftBias(choice);
                    break;

            }

            Reset();
        }

        public string SelectRandomChoice(List<string> population, List<float> weights)
        {
            float totalWeight = 0;
            foreach (float weight in weights)
            {
                totalWeight += weight;
            }
            float[] cumulativeWeights = new float[weights.Count];
            cumulativeWeights[0] = weights[0] / totalWeight; //normalised first weight
            for (int i = 1; i < weights.Count; i++)
            {
                cumulativeWeights[i] = cumulativeWeights[i - 1] + (weights[i] / totalWeight); //cumulative, normalised weights
            }
            double randomValue = rnd.NextDouble(); //random number betwwen 0 and 1
            for (int i = 0; i < cumulativeWeights.Length; i++)
            {
                if (randomValue < cumulativeWeights[i])
                { //select index based on random value
                    return population[i];
                }
            }

            return population[0]; //fallback
        }

        //Objective 5di\1\5a
        //Neural Network
        private void AddNode()
        {
            //store all enabled nodes
            List<(int, int)> enabled = new();
            //iterate through to fine enabled
            for (int i = 0; i < edges.Count; i++)
            {
                for (int j = 0; j < edges[i].Count; j++)
                {
                    //if enabled, store on list
                    if (edges[i][j].enabled)
                    {
                        enabled.Add((i, j));
                    }
                }
            }

            //randomly select enabled edge to split and inset new node
            (int, int) selectedEdge = enabled[rnd.Next(0, enabled.Count - 1)];
            //retrieve selected edge
            Edge edge = edges[selectedEdge.Item1][selectedEdge.Item2];

            //disable selected edge
            edge.enabled = false;
            edges[selectedEdge.Item1][selectedEdge.Item2] = edge;

            //create new node
            int newNode = maxNode;
            maxNode++;

            //initalise new node
            Node node = new(default_activation);
            nodes.Add(node);

            //add edges connecting new node to the rest of the graph
            AddEdge(selectedEdge.Item1, newNode, 1);
            AddEdge(newNode, selectedEdge.Item2, edge.weight);
        }

        //Objective 5di\1\5b
        //Try/Catch
        private void AddEdge(int i, int j, float weight)
        {
            try
            {
                edges[i][j].enable(); //try to enable existing edge
            }
            catch
            {
                try { edges[i].Add(new Edge(weight)); } //try to add edge to node
                catch
                {
                    //create new list for node, add edge to that
                    List<Edge> temp = new();
                    edges.Add(temp);
                    edges[i].Add(new Edge(weight));
                }
            }
        }

        public void AddEnabled()
        {
            var disablededges = new List<Edge>();
            // find disabled edges
            foreach (var edgeList in edges)
            {
                disablededges.AddRange(edgeList.Where(e => !e.enabled));
            }

            if (disablededges.Count > 0)
            {
                var randomEdge = disablededges[rnd.Next(disablededges.Count)];
                randomEdge.enabled = true;
            }
        }
        //Objective 5di\1\5c
        public void ShiftWeight(string type)
        {
            //randomly shift/perturb or set one of the edges weights
            int i = rnd.Next(edges.Count);
            int j = rnd.Next(edges[i].Count);
            Edge edge = edges[i][j];
            if (type == "weight perturb")
            {
                edge.weight += (float)(rnd.NextDouble() * 2) - 1f;
            }
            else if (type == "weight set")
            {
                edge.weight = (float)(rnd.NextDouble() * 2) - 1f;
            }
            edges[i][j] = edge;
        }
        
        //Objective 5di\1\5d
        public void ShiftBias(string type)
        {
            //randomly shift/perturb or set one of the edges weights
            int i = rnd.Next(inputs, maxNode);
            Node node = nodes[i];
            if (type == "bias perturb")
            {
                node.bias += (float)(rnd.NextDouble() * 2) - 1f;
            }
            else if (type == "bias set")
            {
                node.bias = (float)(rnd.NextDouble() * 2) - 1f;
            }
            nodes[i] = node;
        }

        //Enum
        //Lists
        public (int, int) RandomPair()
        {
            /* select a pair of nodes such that 
             *  i is not an output
             *  j is not an input
             *  i != j
             */

            List<int> availableNodes = Enumerable.Range(0, maxNode) 
                                       .Where(n => !IsOutput(n)) 
                                       .ToList();

            int i = availableNodes[rnd.Next(availableNodes.Count)];
            int j = 0;

            List<int> jList = Enumerable.Range(0, maxNode)
                              .Where(n => !IsInput(n) && n != i)
                              .ToList();

            if (jList.Count == 0)
            {
                j = maxNode;
                AddNode();
            }
            else
            {
                j = jList[rnd.Next(jList.Count)];
            }

            return (i, j);
        }

        public bool IsInput(int n)
        {
            return 0 <= n && n < inputs;
        }

        public bool IsOutput(int n)
        {
            return inputs <= n && n < unhidden;
        }

        public bool IsHidden(int n)
        {
            return unhidden <= n && n < maxNode;
        }

        public bool IsDisabled()
        {
            for (int i = 0; i < edges.Count; i++)
            {
                for (int j = 0; j < edges[i].Count; j++)
                {
                    if (edges[i][j].enabled)
                    {
                        return false;
                    }
                }
            }

            return false;
        }

        public void Reset()
        {
            for (int i = 0; i < maxNode; i++)
            {
                Node node = nodes[i];
                node.output = 0;
                nodes[i] = node;
            }
            fitness = 0;
        }
    }
    public class Specie
    {
        public Random rnd;
        public int maxFitnessHistory { get; }
        public List<Genome> members { get; set; }
        public float fitnessSum;
        public List<float> fitnessHistory;

        public Specie(int input_max_fitness_history, List<Genome> input_members)
        {
            maxFitnessHistory = input_max_fitness_history;
            members = input_members;
            fitnessSum = 0;
            fitnessHistory = new List<float>();
            rnd = new Random();
        }

        //Objective 5di
        //Genomic Crossover
        public Genome Breed(Dictionary<string, float> mutationProbabilities, Dictionary<string, float> breedProbabilities)
        {
            //return a child as a result of either a mutated clone or crossover between two parent genomes
            Genome child;
            var population = new List<string>(breedProbabilities.Keys);
            var weights = new List<float>(breedProbabilities.Values);
            var choice = SelectRandomChoice(population, weights);

            if (choice == "asexual" || members.Count == 1)
            {
                //Objective 5di\1
                child = members[rnd.Next(members.Count)];
                child.Mutate(mutationProbabilities);
            }
            else
            { //choice == "sexual"
                //Objective 5di\2\5a
                int num1 = rnd.Next(members.Count);
                int num2 = rnd.Next(members.Count);
                while (num1 == num2)
                {
                    num2 = rnd.Next(members.Count);
                } //find 2 distinct parents in the specie

                //Objective 5di\2\5b
                child = GenomicCrossover(members[num1], members[num2]);
            }

            return child;
        }

        public void UpdateFitness()
        {
            fitnessSum = 0;
            foreach (var genome in members)
            {
                genome.adjustedFitness = genome.fitness / members.Count;
                fitnessSum += genome.adjustedFitness;
            }

            fitnessHistory.Add(fitnessSum);

            if (fitnessHistory.Count > maxFitnessHistory)
            {
                fitnessHistory.RemoveAt(0);
            }
        }

        public void CullGenomes(bool fittestOnly)
        {
            //externimate the weakest per specie
            if (fittestOnly)
            {
                Genome fittestGenome = members[0];
                foreach (Genome genome in members)
                {
                    fittestGenome = (genome.fitness > fittestGenome.fitness) ? genome : fittestGenome;
                }
                members = new List<Genome> { fittestGenome };
            }
            else
            {
                members = members.OrderByDescending(g => g.fitness).ToList();
                int countToTake = (int)Math.Ceiling(members.Count * 0.25);
                members = members.Take(countToTake).ToList();
            }
        }

        public Genome GetBest()
        {
            //get member with highest fitness
            Genome bestGenome = members[0];
            foreach (Genome genome in members)
            {
                bestGenome = (genome.fitness > bestGenome.fitness) ? genome : bestGenome;
            }
            return bestGenome;
        }

        public bool CanProgress()
        {
            //determines whether species should survive the culling
            int n = fitnessHistory.Count;
            float totalFitness = 0;

            foreach (Genome g in members)
            {
                totalFitness += g.fitness;
            }

            float avgFitness = totalFitness / n;
            return avgFitness > fitnessHistory[0] || n < maxFitnessHistory;
        }

        public string SelectRandomChoice(List<string> population, List<float> weights)
        {
            float totalWeight = 0;
            foreach (float weight in weights)
            {
                totalWeight += weight;
            }
            float[] cumulativeWeights = new float[weights.Count];
            cumulativeWeights[0] = weights[0] / totalWeight; //normalised first weight
            for (int i = 1; i < weights.Count; i++)
            {
                cumulativeWeights[i] = cumulativeWeights[i - 1] + (weights[i] / totalWeight); //cumulative, normalised weights
            }
            double randomValue = rnd.NextDouble(); //random number betwwen 0 and 1
            for (int i = 0; i < cumulativeWeights.Length; i++)
            {
                if (randomValue < cumulativeWeights[i])
                { //select index based on random value
                    return population[i];
                }
            }

            return population[0]; //fallback
        }

        //Objective 5di\2
        //Genomic Crossover
        //Lists
        public Genome GenomicCrossover(Genome a, Genome b)
        {
            //breed two genomes and return the child. matching genes are inherited randomly, while disjoin genes are inherited from the fitter parent
            Genome child = new(a.inputs, a.outputs, a.default_activation);
            List<List<Edge>> aIn = (a.edges.Count > b.edges.Count) ? a.edges : b.edges;
            List<List<Edge>> bIn = (a.edges.Count > b.edges.Count) ? b.edges : a.edges;
            List<Edge> tempList;
            List<List<(Edge, Edge)>> combinedIns = new();
            Edge childEdge;

            //create a list of edges that match between the two parents "homologous genees"
            for (int i = 0; i < aIn.Count; i++)
            {
                combinedIns.Add(new List<(Edge, Edge)>());
                tempList = (aIn[i].Count < bIn[i].Count) ? aIn[i] : bIn[i]; //templist is the shorter of both lists
                for (int j = 0; j < tempList.Count; j++)
                {
                    combinedIns[i].Add((aIn[i][j], bIn[i][j]));
                }
            }

            //child inherits either parent's homologous genes
            for (int i = 0; i < combinedIns.Count; i++)
            {
                List<(Edge, Edge)> edgePairList = combinedIns[i];
                foreach (var edgePair in edgePairList)
                {
                    childEdge = (rnd.Next(1) == 0) ? edgePair.Item1 : edgePair.Item2;
                    child.edges[i].Add(childEdge);
                }
            }

            //get fitter parent
            Genome fitterParent = (a.fitness > b.fitness) ? a : b;
            //if edge already exists, leave it be (homologous gene)
            //if edge does not already exist, add the fitter parent's edge (disjoint gene)
            for (int i = 0; i < fitterParent.edges.Count; i++)
            {
                for (int j = 0; j < fitterParent.edges[i].Count; j++)
                {
                    try
                    {
                        child.edges[i][j].exists();
                    }
                    catch
                    {
                        child.edges[i].Add(aIn[i][j]);
                    }
                }
            }

            //update maxNode
            for (int i = 0; i < child.edges.Count; i++)
            {
                for (int j = 0; j < child.edges[i].Count; j++)
                {
                    int currentMax = (i > j) ? i : j;
                    child.maxNode = (currentMax > child.maxNode) ? currentMax : child.maxNode;
                }
            }
            child.maxNode++;

            //inherit nodes
            for (int i = 0; i < child.maxNode; i++)
            {
                child.nodes.Add(fitterParent.nodes[i]);
            }

            child.Reset();
            return child;
        }
    }

    public class Brain
    {
        //base class for a 'brain' that learns through the evolution of a population of genomes
        public int inputs { get; }
        public int outputs { get; }
        public int populationSize { get; }
        public NEATHyperparameters hyperparameters { get; }
        public List<Specie> species;
        public int currentSpecies, currentGenome;
        public Genome globalBest;
        int generation = 0;
        public Brain(NEATHyperparameters input_NEAThyperparameters)
        {
            hyperparameters = input_NEAThyperparameters;
            inputs = hyperparameters.obsDim;
            outputs = hyperparameters.actDim;
            populationSize = input_NEAThyperparameters.populationSize;
            species = new();
            generation = 0;
            currentSpecies = 0;
            currentGenome = 0;
            globalBest = new Genome(inputs, outputs, hyperparameters.defaultActivation);
        }

        public void Generate()
        {
            //generate iniital population of genomes
            for (int i = 0; i < populationSize; i++)
            {
                Genome genome = new(inputs, outputs, hyperparameters.defaultActivation);
                genome.GenerateNetwork();
                ClassifyGenome(genome);
            }
        }

        //Objective 5c
        //Genomic Distance
        public void ClassifyGenome(Genome genome)
        {
            //classify genome into a species via genomic distance algorithm
            //see if genome fits into any existing species
            foreach (Specie specie in species)
            {
                Genome rep = specie.members[0];
                float distance = GenomicDistance(genome, rep, hyperparameters.distanceWeights);
                if (distance <= hyperparameters.deltaThreshold)
                {
                    specie.members.Add(genome);
                    return;
                }
            }
            //doesnt fit into any existing species, create new species
            List<Genome> listOfOneGenome = new() { genome };
            species.Add(new Specie(hyperparameters.maxFitnessHistory, listOfOneGenome));
        }

        //Genomic Distance
        public float GenomicDistance(Genome a, Genome b, Dictionary<string, float> weights)
        {
            //calculate genomic distance between two genomes
            List<List<Edge>> aIn = (a.edges.Count > b.edges.Count) ? a.edges : b.edges;
            List<List<Edge>> bIn = (a.edges.Count > b.edges.Count) ? b.edges : a.edges;
            List<List<(Edge, Edge)>> matchingEdges = new();
            int noOfMatchingEdges = 0;
            int noOfDisjointEdges = 0;
            List<Edge> tempList;
            //calculate length of each genome
            int aLength = 0;
            int bLength = 0;
            foreach (var aList in aIn)
            {
                foreach (var item in aList)
                {
                    aLength++;
                }
            }
            foreach (var bList in bIn)
            {
                foreach (var item in bList)
                {
                    bLength++;
                }
            }
            int N_edges = (aLength > bLength) ? aLength : bLength; //choose the greatest length
            int N_nodes = (a.maxNode < b.maxNode) ? a.maxNode : b.maxNode; //choose the lesser length

            //create a list of edges that match between the two parents "homologous genees"
            for (int i = 0; i < aIn.Count; i++)
            {
                matchingEdges.Add(new List<(Edge, Edge)>());
                tempList = (aIn[i].Count < bIn[i].Count) ? aIn[i] : bIn[i]; //templist is the shorter of both lists
                for (int j = 0; j < tempList.Count; j++)
                {
                    matchingEdges[i].Add((aIn[i][j], bIn[i][j]));
                    noOfMatchingEdges++;
                }
            }

            //find number of disjoint edges
            var aEdgesFlattened = aIn.SelectMany(x => x).ToList();
            var bEdgesFlattened = bIn.SelectMany(x => x).ToList();
            //find edges that are in a but not in b and vice versa
            var aMinusB = aEdgesFlattened.Except(bEdgesFlattened).ToList();
            var bMinusA = bEdgesFlattened.Except(aEdgesFlattened).ToList();
            //combine and take the length
            noOfDisjointEdges = aMinusB.Union(bMinusA).ToList().Count;

            //calculate difference in weights across every matching edge
            float weightDiff = 0;
            foreach (var list in matchingEdges)
            {
                foreach (var edgePair in list)
                {
                    weightDiff += Math.Abs(edgePair.Item1.weight - edgePair.Item2.weight);
                }
            }

            float biasDiff = 0;
            //calulate difference in biases across every matching node
            for (int i = 0; i < N_nodes; i++)
            {
                biasDiff += Math.Abs(a.nodes[i].bias - b.nodes[i].bias);
            }

            float c1 = weights["edge"] * noOfDisjointEdges / N_edges;
            float c2 = weights["weight"] * weightDiff / noOfMatchingEdges;
            float c3 = weights["bias"] * biasDiff / N_nodes;
            return c1 + c2 + c3;
        }

        public void UpdateFittest()
        {
            //update the highest performing genome in the population

            //get top performers from each species
            List<Genome> topPerformers = new();
            foreach (Specie specie in species)
            {
                topPerformers.Add(specie.GetBest());
            }

            //if any in top performers better than current best, update
            foreach (Genome genome in topPerformers)
            {
                globalBest = (genome.fitness > globalBest.fitness) ? genome : globalBest;
            }
        }

        public void Evolve()
        {
            //the final meat of the program.
            //evolve the population of genomes by elimnating the poorest performing genomes
            //and repopulating with mutated children
            //prioritising the most promising species

            //initalise currentGenome and currentSpecies for next round of training
            currentGenome = 0;
            currentSpecies = 0;

            //get fitness sum across all species
            float globalFitnessSum = 0;
            foreach (Specie specie in species)
            {
                specie.UpdateFitness();
                globalFitnessSum += specie.fitnessSum;
            }

            if (globalFitnessSum == 0)
            {
                //no progress, mutate everybody
                foreach (Specie specie in species)
                {
                    foreach (Genome genome in specie.members)
                    {
                        genome.Mutate(hyperparameters.mutationProbabilities);
                    }
                }
            }
            else
            {
                //only keep species that have potential to improve
                List<Specie> survivingSpecies = new();
                foreach (Specie specie in species)
                {
                    if (specie.CanProgress())
                    {
                        survivingSpecies.Add(specie);
                    }
                }
                species = survivingSpecies;

                //elimate the weakest genomes in each species
                foreach (Specie specie in species)
                {
                    specie.CullGenomes(false);
                }

                //repopulate
                foreach (Specie specie in species)
                {
                    float ratio = specie.fitnessSum / globalFitnessSum; //species quality with respect to global quality
                    float diff = populationSize - GetPopulationSize(); //how many extra genomes needed
                    int offspring = (int)Math.Round(ratio * diff); //how many offspring to create from this species
                    for (int i = 0; i < offspring; i++)
                    {
                        Genome child = specie.Breed(hyperparameters.mutationProbabilities, hyperparameters.breedProbabilities);
                        ClassifyGenome(child);
                    }
                }

                //no species survived
                //repopulate with mutated minimal structures and global best
                if (species.Count == 0)
                {
                    for (int i = 0; i < populationSize; i++)
                    {
                        Genome g;
                        if (i % 3 == 0)
                        {
                            g = globalBest;
                        }
                        else
                        {
                            g = new Genome(inputs, outputs, hyperparameters.defaultActivation);
                            g.GenerateNetwork();
                            g.Mutate(hyperparameters.mutationProbabilities);
                            ClassifyGenome(g);
                        }
                    }
                }
            }
            generation++;
        }

        public bool ShouldEvolve()
        {
            //determine if the system should continue to evolve
            //based on the maximum fitness and generation count
            UpdateFittest();
            bool fit = globalBest.fitness <= hyperparameters.maxFitness;
            bool end = generation != hyperparameters.maxGenerations;

            return fit && end;
        }

        public void NextIteration()
        {
            //call after every evaluation of individual genome to progress training
            Specie s = species[currentSpecies];
            if (currentGenome < s.members.Count - 1)
            {
                currentGenome++; //move to next genome
            }
            else if (currentSpecies < species.Count - 1)
            {
                currentSpecies++;
                currentGenome = 0; //move onto next species
            }
            else
            {
                currentGenome = 0;
                currentSpecies = 0;
            }

            //adapted from original -  Evolve() is now called from the program
        }

        public int GetPopulationSize()
        {
            //return true (calculated) population size
            int size = 0;
            foreach (Specie specie in species)
            {
                size += specie.members.Count;
            }
            return size;
        }

    }


    //Dictionary
    public class NEATHyperparameters : Hyperparameters
    {
        public float deltaThreshold, maxFitness, maxGenerations;
        public int maxFitnessHistory, populationSize;

        public Activation defaultActivation;
        public Dictionary<string, float> distanceWeights, breedProbabilities, mutationProbabilities;

        public NEATHyperparameters()
        {
            deltaThreshold = 1.5f;
            maxFitnessHistory = 30;
            populationSize = 15;
            defaultActivation = new LReLU();
            maxFitness = 10000000;
            maxGenerations = 100000000;
            distanceWeights = new Dictionary<string, float>{
                {"edge" , 1.0f},
                {"weight" , 1.0f},
                {"bias" , 1.0f}
            };
            breedProbabilities = new Dictionary<string, float>{
                {"asexual" , 0.5f},
                {"sexual" , 0.5f}
            };
            mutationProbabilities = new Dictionary<string, float>{
                {"node" , 0.01f},
                {"edge" , 0.09f},
                {"weight perturb" , 0.4f},
                {"weight set" , 0.1f},
                {"bias perturb", 0.3f},
                {"bias set", 0.1f}
            };
        }
    }
}