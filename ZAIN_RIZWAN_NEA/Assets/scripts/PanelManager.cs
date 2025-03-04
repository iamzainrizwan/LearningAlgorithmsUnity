using UnityEngine;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine.AI;

public class PanelManager : MonoBehaviour
{
    [SerializeField]
    private GameObject panelPrefab; 
    [SerializeField]
    private Canvas targetCanvas;
    
    private List<DraggablePanel> panels = new();
    
    void Awake()
    {
        //find canvas
        targetCanvas = FindAnyObjectByType<Canvas>();
    }

    public DraggablePanel CreatePanel(string headerText, string bodyText, Vector2 position = default){
        GameObject panelObj = Instantiate(panelPrefab, targetCanvas.transform);

        //set position (defaults to center)
        RectTransform rectTransform = panelObj.GetComponent<RectTransform>();
        if (position == default){
            position = Vector2.zero;
        }
        rectTransform.anchoredPosition = position;

        //setup panel content
        DraggablePanel panel = panelObj.GetComponent<DraggablePanel>();
        panel.SetBody(bodyText);
        panel.SetHeader(headerText);

        //add to list
        panels.Add(panel);

        return panel;
    }

    public void DestroyAllPanels(){
        //if panel exists, destroy
        foreach (DraggablePanel panel in panels){
            if (panel != null){
                panel.DestroyPanel();
            }
        }

        //clear list
        panels.Clear();
    }
}
