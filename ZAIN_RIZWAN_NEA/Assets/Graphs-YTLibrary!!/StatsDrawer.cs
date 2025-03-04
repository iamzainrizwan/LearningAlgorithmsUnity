//Objective 8c
//Unity UI
//Read Ffom File

using System.IO;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using System.Linq;
using Tensorflow;
using Unity.VisualScripting;


public class StatsDrawer : MonoBehaviour
{
    public string filepath;
    public TMP_Dropdown dropdown;
    public GraphHandler graphHandler;
    private List<int> roundNumbers;
    private List<float>[] stats;
    bool done = false;
    int numLines = 0;



    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        //add listener for dropdown
        dropdown.onValueChanged.AddListener(OnDropdownValueChanged);
        //intialise stats array of lists
        roundNumbers = new List<int>();
        stats = new List<float>[5] { new(), new(), new(), new(), new() };
        ParseStats();
        graphHandler.numPoints = numLines;
    }
    void OnDropdownValueChanged(int index)
    {
        UpdateGraph(index);
        Debug.Log("Dropdown changed to: " + index);
    }

    //Objective 8cii
    void UpdateGraph(int index)
    {
        for (int i = 0; i < numLines; i++)
        {
            graphHandler.ChangePoint(i, new Vector2(i, stats[index][i]));
            graphHandler.UpdateGraph();
            Debug.Log("Point: " + i + ", " + stats[index][i]);
        }
    }
    
    //Read from File
    public void ParseStats()
    {
        using StreamReader sr = new(filepath);

        //read first line
        string line = sr.ReadLine();

        //continue to read lines
        while (line != null)
        {
            //split up into individual values
            string[] lineValues = line.Split(',');
            Debug.Log(line);
            roundNumbers.Add(int.Parse(lineValues[0]));
            stats[0].Add(float.Parse(lineValues[1]));
            stats[1].Add(float.Parse(lineValues[2]));
            stats[2].Add(float.Parse(lineValues[3]));
            stats[3].Add(float.Parse(lineValues[4]));
            stats[4].Add(float.Parse(lineValues[5]));

            foreach (var value in lineValues[1])
            {
                Debug.Log(value);
            }

            //iterate count and read next line
            numLines++;
            line = sr.ReadLine();
        }

        Debug.Log(numLines);
        Debug.Log(stats[0].Count);
        Debug.Log(stats[1].Count);
        Debug.Log(stats[2].Count);
        Debug.Log(stats[3].Count);
        Debug.Log(stats[4].Count);

        foreach (var value in stats){
            Debug.Log("Values:");
            foreach (var val in value){
                Debug.Log(val);
            }
        }
    }
}
