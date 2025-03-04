using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Runtime.CompilerServices;
using UnityEngine.Rendering;
using System.IO;

public class UIManager : MonoBehaviour
{
    //ui elements
    public Button proceedButton;
    public TMP_Dropdown algorithmDropdown;
    public TMP_Text stateText;
    public string infostring;
    public GameObject inputCanvas;  

    //needed components from other gameobjects
    public FieldResetter resetter;
    public TakeFieldsInput[] agentFieldInputters;
    public TakeFieldsInput simulationFieldInputter;

    //sets of fields to hide/show
    public GameObject[] agentFields;
    public GameObject[] simulationFields;

    //params for other parts of program
    public string[] runnerParams;
    public string[] taggerParams;
    public string[] simulationParams;

    //indexes
    public int index = 0;
    public TakeFieldsInput selectedAgentFields;

    //communicate to game manager that game has started
    public GameManager gameManager;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        proceedButton.onClick.AddListener(proceed);
        //setup for first use
        simulationFields[0].SetActive(false);
        agentFields[0].SetActive(true);
        agentFields[1].SetActive(false);
        agentFields[2].SetActive(false);
        agentFields[3].SetActive(false);
        agentFields[4].SetActive(true);
        stateText.text = "Currently Configuring: Runner";
    }

    void proceed()
    {
        //if on either of the agent screens
        if (index == 0 || index == 1)
        {
            //use correct fields
            selectedAgentFields = agentFieldInputters[algorithmDropdown.value];
            infostring = selectedAgentFields.inputPurpose;
            //check which agent fields to take input from 
            selectedAgentFields.TakeInputs();
            if (index == 0)
            {
                runnerParams = selectedAgentFields.inputs; //take inputs
                stateText.text = "Currently Configuring: Tagger"; //move on to tagger
            } else if (index == 1 && selectedAgentFields) { //move onto simulation config
                taggerParams = selectedAgentFields.inputs; //take inputs
                stateText.text = "Currently Configuring: Simulation"; //move onto simulation
                //change active fields
                for (int i = 0; i < agentFields.Length; i++)
                {
                    agentFields[i].SetActive(false); 
                }
                for (int i = 0; i < simulationFields.Length;i++)
                {
                    simulationFields[i].SetActive(true);
                }
                
            }
            //increment index to update section position
            index++;
        } else if (index == 2 && simulationFieldInputter.isAllValid) { //start program
            simulationFieldInputter.TakeInputs(); //take inputs
            simulationParams = simulationFieldInputter.inputs;
            //create .csv file
            using (StreamWriter sw = new (simulationParams[3], false)) {
                sw.WriteLine("Round, Runner Reward, Tagger Reward, Average Distance, Runner Wins, Tagger Wins");
            }
            //send signal that simulation is to be started
            gameManager.gameRunning = true;
        }
    }
}
