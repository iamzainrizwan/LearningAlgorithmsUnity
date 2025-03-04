using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.UIElements.InputSystem;
using System.Diagnostics;

public class TakeFieldsInput : MonoBehaviour
{
    //ui elements & internal info
    public string inputPurpose = "";

    //fields to take input from
    public TMP_InputField[] inputFields;
    public Toggle[] inputToggles;

    //valids
    private FieldRangeValidator[] validators;
    private bool[] isValid;
    public bool isAllValid = true;

    //output string of outputs
    public string[] inputs;
    // Update is called once per frame
    public void TakeInputs()
    {
        //makes it easier to check each validator
        validators = new FieldRangeValidator[inputFields.Length];
        for (int j = 0; j < validators.Length; j++)
        {
            validators[j] = inputFields[j].GetComponent<FieldRangeValidator>();
        }

        //check if every input field has a valid input
        int i = 0;
        while (!isAllValid)
        {
            bool isFieldValid = validators[i].isValid;
            if (!isFieldValid)
            {
                isAllValid = false;
            }
            else
            {
                isValid[i] = true;
            }
            i++;
        }

        //store each value in inputs array
        inputs = new string[inputFields.Length + inputToggles.Length + 1];
        string teststring = "";
        if (isAllValid) {
            inputs[0] = inputPurpose;
            for (i = 0; i < inputFields.Length; i++)
            {
                inputs[i+1]  = inputFields[i].text;
                teststring += inputFields[i].text;
                teststring += "\n";
            }
            //store toggle results in input array
            if (inputToggles.Length > 0){
                for (i = 0; i < inputToggles.Length; i++)
                {
                    UnityEngine.Debug.Log(i);
                    if (inputToggles[i].isOn)
                    {
                        inputs[i + inputFields.Length + 1] = "true";
                    } else {
                        inputs[i + inputFields.Length + 1] = "false";
                    }
                }
            }

        } else {
            UnityEngine.Debug.Log("fix up ur inputs");
        }
        UnityEngine.Debug.Log(teststring);
    }
}
