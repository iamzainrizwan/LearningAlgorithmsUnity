using UnityEngine;
using TMPro;
using System.Text.RegularExpressions;

public class FieldRangeValidator : MonoBehaviour
{
    //diffentiatte between different types of validators
    public int validatorType = 0;
    //for type 0
    public float minValue = 0f;
    public float maxValue = 100f;

    private TMP_InputField inputField;
    public TMP_Text feedbackText;
    public bool isValid;
    
    void Awake()
    {
        //get component of input field
        inputField = GetComponent<TMP_InputField>();

        //add end edit listener
        inputField.onEndEdit.AddListener(ValidateInput);
    }

    void ValidateInput(string input)
    {
        if (validatorType == 0) {
            //take input
            float val = float.Parse(input);

            //check & output suitable message
            if (val < minValue) {
                UpdateText("value too small.", Color.red);
                isValid = false;
            } else if (val > maxValue) {
                UpdateText("value too large", Color.red);
                isValid = false;
            } else if (val >= minValue && val <= maxValue) {
                UpdateText("", Color.white);
                Debug.Log("aura gained");
                isValid = true;
            }
        } else if (validatorType == 1) {
            isValid = Regex.IsMatch(input, @"^[a-zA-Z]+$"); //needs a file path validator
            if (!isValid){
                UpdateText("Should be a file path.", Color.red);
            }
        }
    }

    void UpdateText(string feedback, Color colour)
    {
        if (feedbackText != null)
        {
            feedbackText.text = feedback;
            feedbackText.color = colour;
        }
    }
}
