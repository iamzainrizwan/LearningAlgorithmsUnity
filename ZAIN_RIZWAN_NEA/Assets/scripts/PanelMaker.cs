using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class PanelMaker : MonoBehaviour
{
    public string headerText;
    [TextArea(3, 10)] //makes text area bigger in inspector
    public string bodyText;   // The text to display when the button is clicked

    public PanelManager panelManager;
    //add listener & panel manager
    void Start()
    {
        Button btn = GetComponent<Button>();
        btn.onClick.AddListener(CreatePanel);
    }
    
    
    // create panel
    public void CreatePanel()
    {
        panelManager.CreatePanel(headerText, bodyText);
    }
}
