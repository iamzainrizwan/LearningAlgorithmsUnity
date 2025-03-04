//Objectives 1, 2
//Unity UI
//Arrays
//Coroutines

using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using Unity.VisualScripting;

public class UInterfaceManager : MonoBehaviour
{
    [Header("params")]
    public string runnerAlgorithm;
    public string taggerAlgorithm;

    [Header("screens")]
    public GameObject[] screens; //array of screens to switch between as program progresses
    //current screen index
    public int screenIndex = 0; 
    [Header("screen animation settings")]
    [SerializeField] private float animationDuration = 0.5f;
    [SerializeField] private float moveDistance = 1000f;
    [SerializeField] private AnimationCurve easingCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
    [SerializeField] private AnimationCurve fadeCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
    [Header("buttons")] [Space]
    //buttons to switch between config screens
    public Button nextButton;
    public Button backButton;
    //button to start simulation
    public Button startButton;
    [Header("managers & stats drawer")] [Space]
    //gameManager to manage simulation
    public GameManager gameManager;
    //panelsManager for UI panels
    public PanelManager panelManager;
    //stats drawer to control stats on final screen
    public StatsDrawer statsDrawer;
    //validators to ensure correct input before simulation started
    [Header("other stuff")][Space]
    public TakeFieldsInput[] validators;
    //dropdownManagers to get validators for agent configs
    public DropdownHandler runnerDropdownHandler;
    public DropdownHandler taggerDropdownHandler;
    //error text for user
    public TMP_Text errorText;
    //for config control items
    public GameObject configControl;
    //button to start stats screen
    public Button statsButton;
    //button to pause game
    public Button pauseButton;
    //button to resume game
    public Button resumeButton;
    //button to toggle stats while simulation is running
    public Button statsToggleButton;
    private bool isAnimating;
    private RectTransform[] screenRects;
    private CanvasGroup[] screenGroups;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        //initalise
        gameManager.gameRunning = false;
        screenRects = new RectTransform[screens.Length];
        screenGroups = new CanvasGroup[screens.Length];
        int i = 0;
        foreach (GameObject screen in screens) {
            screen.SetActive(false);
            screenRects[i] = screen.GetComponent<RectTransform>();
            screenGroups[i] = screen.GetComponent<CanvasGroup>();
            i++;
        }
        screens[0].SetActive(true);
        configControl.SetActive(true);
        validators = new TakeFieldsInput[3];

        //add listeners for buttons
        nextButton.onClick.AddListener(NextScreen);
        backButton.onClick.AddListener(PreviousScreen);
        startButton.onClick.AddListener(StartSimulationChecks);
        pauseButton.onClick.AddListener(PauseGame);
        resumeButton.onClick.AddListener(ResumeGame);


        //add validator for simulation settings will stay the same, so set it now
        validators[2] = screens[2].GetComponent<TakeFieldsInput>();
    }

    //Objective 1aii
    //Coroutines
    void NextScreen()
    {
        if (0 <= screenIndex && screenIndex < 2 && !isAnimating) {
            //deactivate current screen and activate next screen
            StartCoroutine(SwitchScreens(screenIndex, screenIndex + 1, true));
            screenIndex++;
        }
    }

    private IEnumerator<Null> SwitchScreens (int oldIndex, int newIndex, bool slideRight) {
        isAnimating = true;

        //set initial pos
        screens[newIndex].SetActive(true);
        Vector2 oldStartPos = screenRects[oldIndex].anchoredPosition;
        Vector2 newStartPos = oldStartPos;
        newStartPos.x = slideRight ? +moveDistance : -moveDistance;
        screenRects[newIndex].anchoredPosition = newStartPos;

        //set initial alphas
        screenGroups[oldIndex].alpha = 1;
        screenGroups[newIndex].alpha = 0;

        //calculate end positions
        Vector2 oldEndPos = oldStartPos;
        oldEndPos.x += slideRight ? -moveDistance : +moveDistance;
        Vector2 newEndPos = Vector2.zero;

        //move old screen out
        float elapsed = 0;
        while (elapsed < animationDuration){
            //calculate progress and curve values
            elapsed += Time.deltaTime;
            float progress = elapsed/animationDuration;
            float curveValue = easingCurve.Evaluate(progress);
            float fadeValue = fadeCurve.Evaluate(progress);

            //update positions
            screenRects[oldIndex].anchoredPosition = Vector2.Lerp(oldStartPos, oldEndPos, curveValue);

            //update fade
            screenGroups[oldIndex].alpha = Mathf.Lerp(1f, 0f, fadeValue);
            yield return null;
        }

        //set final values for old screen
        screenRects[oldIndex].anchoredPosition = oldEndPos;
        screenGroups[oldIndex].alpha = 0f;
        screens[oldIndex].SetActive(false);
        
        //move new screen in
        elapsed = 0;
        while (elapsed < animationDuration){
            //calculate progress and curve values
            elapsed += Time.deltaTime;
            float progress = elapsed/animationDuration;
            float curveValue = easingCurve.Evaluate(progress);
            float fadeValue = fadeCurve.Evaluate(progress);

            //update positions
            screenRects[newIndex].anchoredPosition = Vector2.Lerp(newStartPos, newEndPos, curveValue);

            //update fade
            screenGroups[newIndex].alpha = Mathf.Lerp(0f, 1f, fadeValue);

            yield return null;
        }

        //set final values for new screen
        screenRects[newIndex].anchoredPosition = newEndPos;
        screenGroups[newIndex].alpha = 1f;
        
        isAnimating = false;
    }

    //Objective 1aii
    //Coroutines
    void PreviousScreen()
    {
        if (2 >= screenIndex && screenIndex > 0 && !isAnimating) {
        //deactivate current screen and activate previous screen
        //deactivate current screen and activate next screen
            StartCoroutine(SwitchScreens(screenIndex, screenIndex - 1, false));
            screenIndex--;
        }
    }

    //Objective 2c
    void StartSimulationChecks()
    {
        if (screenIndex == 2) {
            //get appropriate validators
            validators[0] = runnerDropdownHandler.chosenValidator;
            validators[1] = taggerDropdownHandler.chosenValidator;
            //check if is all valid
            if (validators[0].isAllValid && validators[1].isAllValid && validators[2].isAllValid) {
                Debug.Log("START SIM");
                StartSimulation();
            } else {
                //show error message
                Debug.Log("Invalid input");
                errorText.text = $"Invalid input. For each section:\n " +
                    $"Runner Params: {validators[0].isAllValid} \n" +
                    $"Tagger Params: {validators[1].isAllValid} \n" +
                    $"Simulation Params: {validators[2].isAllValid} \n"
                    ;
            }
        } else {
            errorText.text = "Please complete all sections before starting simulation";
        }
    }

    void StartSimulation(){
        //remove panels
        panelManager.DestroyAllPanels();

        //deactivate current screen and activate next screen
        configControl.SetActive(false);
        screens[screenIndex].SetActive(false);
        screenIndex++;
        screens[screenIndex].SetActive(true);

        //start simulation
        foreach (TakeFieldsInput validator in validators) {
            validator.TakeInputs();
        }
        gameManager.InitaliseSimulation(validators[2].inputs, validators[1].inputs, validators[0].inputs);
        gameManager.gameRunning = true;
    }

    public void PauseGame()
    {
        gameManager.roundPaused  = true;
        gameManager.roundPausedText.text = "Next Round Paused";
    }

    private void ResumeGame()
    {
        gameManager.ResetPositions();
        gameManager.waitingForNextRound = false;
        gameManager.roundPaused= false;
        gameManager.roundPausedText.text = "";
    }

    public void GraphOverview(){
        //deactivate current screen
        screens[screenIndex].SetActive(false);

        //activate final stats panel
        screenIndex++;
        screens[screenIndex].SetActive(true);

        //draw graphs
        statsDrawer.filepath = gameManager.filepath;
        statsDrawer.ParseStats();
    }
}
