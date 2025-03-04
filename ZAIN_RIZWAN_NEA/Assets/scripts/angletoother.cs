using UnityEngine;

public class angletoother : MonoBehaviour
{
    public Transform tf1;
    public Transform tf2;
    // Update is called once per frame
    void Update()
    {
        Vector2 forward = new Vector2(tf1.forward.x, tf1.forward.z);
        Vector2 oneToTwo = new Vector2(tf2.position.x - tf1.position.x, tf2.position.z - tf1.position.z);

        Debug.Log(Vector2.SignedAngle(forward, oneToTwo));
    }
}
