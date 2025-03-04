//Objective 2c
//Unity UI

using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class FieldResetter : MonoBehaviour
{
    public Button resetButton;
    public TMP_Dropdown algoDropdown;
    public TMP_InputField[] DQNFields;
    public float[] ResetValues;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        resetButton.onClick.AddListener(ResetFields);
    }

    public void ResetFields()
    {
        algoDropdown.value = 0; //reset dropdown to DQN

        //reset all DQN values
        for (int i = 0; i < DQNFields.Length; i++)
        {
            DQNFields[i].text = ResetValues[i].ToString();
        }
    }
}
