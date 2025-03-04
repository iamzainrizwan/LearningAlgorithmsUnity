//Objective 1ciii
//Unity UI

using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.EventSystems;
using System.Collections.Generic;
using Unity.VisualScripting;

public class DraggablePanel : MonoBehaviour
{
    [Header("references")]
    [SerializeField] private RectTransform rectTransform;
    [SerializeField] private TMP_Text headerText;
    [SerializeField] private TMP_Text bodyText;
    [SerializeField] private Button closeButton;
    [SerializeField] private RectTransform dragHandle;
    [SerializeField] private CanvasGroup canvasGroup;

    [Header("animation settings")]
    [SerializeField] private float animationDuration = 0.3f;
    [SerializeField] private AnimationCurve fadeInCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
    [SerializeField] private AnimationCurve fadeOutCurve = AnimationCurve.EaseInOut(0, 1, 1, 0);
    [SerializeField] private AnimationCurve scaleCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);

    private bool isDragging = false;
    private bool isAnimating = false;
    private Canvas canvas;
    private Vector3 originalScale;
    
    void Awake()
    {
        //get canvas
        canvas = GetComponentInParent<Canvas>();

        //add listener to close button
        closeButton.onClick.AddListener(() => DestroyPanel());

        //set original scale
        originalScale = rectTransform.localScale;

        //set grabhandle to move
        EventTrigger trigger = dragHandle.gameObject.GetComponent<EventTrigger>();

        //setup drag handler

        //begin drag
        EventTrigger.Entry beginDrag = new()
        {
            eventID = EventTriggerType.BeginDrag
        };
        beginDrag.callback.AddListener((data) => OnBeginDrag((PointerEventData)data));
        trigger.triggers.Add(beginDrag);

        //during drag
        EventTrigger.Entry drag = new()
        {
            eventID = EventTriggerType.Drag
        };
        drag.callback.AddListener((data) => OnDrag((PointerEventData)data));
        trigger.triggers.Add(drag);

        //end drag
        EventTrigger.Entry endDrag = new()
        {
            eventID = EventTriggerType.EndDrag
        };
        endDrag.callback.AddListener((data) => OnEndDrag((PointerEventData)data));
        trigger.triggers.Add(endDrag);

        //start with panel invisible
        canvasGroup.alpha = 0;
        rectTransform.localScale = Vector3.zero;

        //entrance animation
        StartCoroutine(AnimateEntrance());
    }
    
    //let other processes set texts
    public void SetHeader(string text){
        headerText.text = text;
    }

    public void SetBody(string text){
        bodyText.text = text;
    }

    //destroy
    public void DestroyPanel(){
        if (!isAnimating){
            StartCoroutine(AnimateExit());
        }
    }
    
    //drag events
    private void OnBeginDrag(PointerEventData eventData){
        isDragging = true;
    }

    private void OnDrag(PointerEventData eventData){
        if (!isDragging) return;

        //change position wrt to start of drag
        rectTransform.anchoredPosition += eventData.delta / canvas.scaleFactor;
    }

    private void OnEndDrag(PointerEventData eventData){
        isDragging = false;
    }

    private IEnumerator<Null> AnimateEntrance(){
        isAnimating = true;
        float elapsed = 0;

        while(elapsed < animationDuration){
            //figure out progress through animation
            elapsed += Time.deltaTime;
            float progress = elapsed/animationDuration;

            //animate opacity
            canvasGroup.alpha = fadeInCurve.Evaluate(progress);

            //animate scale
            float scale = scaleCurve.Evaluate(progress);
            rectTransform.localScale = scale * originalScale;

            yield return null;
        }

        //ensure final values are set
        canvasGroup.alpha = 1;
        rectTransform.localScale = originalScale;
     
        isAnimating = false;
    }

    private IEnumerator<Null> AnimateExit(){
        isAnimating = true;
        float elapsed = 0;

        while (elapsed < animationDuration){
            elapsed += Time.deltaTime;
            float progress = elapsed / animationDuration;

            //animate opacity
            canvasGroup.alpha = fadeOutCurve.Evaluate(progress);

            //animate scale
            float scale = scaleCurve.Evaluate(1 - progress);
            rectTransform.localScale = scale * originalScale;

            yield return null;
        }

        //actually destroy panel
        Destroy(gameObject);
    }
}
