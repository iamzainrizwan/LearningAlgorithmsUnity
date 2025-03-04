using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class SimStatsManager : MonoBehaviour
{
    //stats to be assigned from game manager
    public float[] taggerSimInput;
    public float[] runnerSimInput;
    public bool[] taggerSimOutput;
    public bool[] runnerSimOutput;
    public float runnerReward;
    public float taggerReward;
    public int taggerRoundsWon;
    public int runnerRoundsWon;
    public string lastRoundWinnerName;

    //scene assigns
    public GameManager gameManager;
    public Button toggleButton;
    public TMP_Text taggerColumn;
    public TMP_Text runnerColumn;
    public TMP_Text taggerScore;
    public TMP_Text runnerScore;
    public TMP_Text lastRoundWinner;

    //flag
    public bool isActive = false;

    //internal array so the program knows what each output means
    string[] outputLabels = {"forward", "backward", "turn left", "turn right", "jump"};
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        toggleButton.onClick.AddListener(ToggleStatsPanel);
        gameObject.SetActive(false);
        isActive = false;
    }

    public void UpdateStatsPanel(){
        taggerColumn.text = FormatString(taggerSimInput, taggerSimOutput, taggerReward);
        runnerColumn.text = FormatString(runnerSimInput, runnerSimOutput, runnerReward);
        taggerScore.text = taggerRoundsWon.ToString();
        runnerScore.text = runnerRoundsWon.ToString();
        lastRoundWinner.text = lastRoundWinnerName;
    }

    public void ToggleStatsPanel(){
        isActive = !isActive;
        gameObject.SetActive(isActive);
    }

    private string FormatString(float[] inputs, bool[] outputs, float reward){
        string outputsString = FormatString(outputs);

        return $"x position: {inputs[0]:F3}\n" +
               $"y position: {inputs[1]:F3}\n" +
               $"z position: {inputs[2]:F3}\n" +
               $"direction wrt forward: {inputs[3]:F3}\n" +
               $"distance: {inputs[4]:F3}\n" +
               $"angle wrt other {inputs[5]:F3}\n" +
               $"reward {reward:F3}\n" +
               $"actions taken:\n" +
               $"{outputsString}";
    }

    private string FormatString(bool[] outputs){
        System.Text.StringBuilder sb = new();
        //if output is "true", add output to 
        for (int i = 0; i < 5; i++){
            if (outputs[i]){
                if (sb.Length > 0) sb.Append(",\n");
                sb.Append(outputLabels[i]);
            }
        }
        return sb.ToString();
    }
}
