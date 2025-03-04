//Objective 2ai
//Unity UI

using UnityEngine;
using TMPro;

public class DropdownHandler : MonoBehaviour
{
    public TMP_Dropdown dropdown; //reference to dropdown for user
    public GameObject[] fieldGroups; //array of field groups for different algorithms to store/hide
    public TakeFieldsInput chosenValidator;

    void Start()
    {
        //add listener for dropdown
        dropdown.onValueChanged.AddListener(OnDropdownValueChanged);
        //initalise field visibility based on default selection
        UpdateFieldVisibility(dropdown.value);

    }

    void OnDropdownValueChanged(int index)
    {
        UpdateFieldVisibility(index);
    }

    void UpdateFieldVisibility(int index)
    {
        for (int i = 0; i < fieldGroups.Length; i++)
        {
            fieldGroups[i].SetActive(i== index);
            chosenValidator = fieldGroups[index].GetComponent<TakeFieldsInput>();
        }
        switch (index) {
            case 0:
                chosenValidator.inputPurpose = "DQN";
                break;
            case 1:
                chosenValidator.inputPurpose = "PPO";
                break;
            case 2:
                chosenValidator.inputPurpose = "NEAT";
                break;
        }
    }
}
