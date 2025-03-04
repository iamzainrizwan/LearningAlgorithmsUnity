using UnityEngine;

public class playermov : MonoBehaviour
{
    
    Rigidbody rb;
    Transform bodytransform;
    public Transform groundedCheck;
    public LayerMask groundLayers;
    public float forceMultiplier = 100;
    public float jumpMultiplier = 0.04f;
    public float rotationMultiplier = 10;
    private bool isGrounded;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
       rb = GetComponent<Rigidbody>();
       bodytransform = GetComponent<Transform>(); 
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        isGrounded = Physics.CheckSphere(groundedCheck.position, 0.2f, groundLayers);


        if (Input.GetKey(KeyCode.W))
        {
            rb.AddForce(bodytransform.forward * forceMultiplier);
        }

        if (Input.GetKey(KeyCode.S))
        {
            rb.AddForce(bodytransform.forward * -forceMultiplier);
        }

        if (Input.GetKey(KeyCode.Space) && isGrounded)
        {
            rb.AddForce(bodytransform.up * jumpMultiplier);
        }

        if (Input.GetKey(KeyCode.A))
        {
            bodytransform.Rotate(0, rotationMultiplier, 0);
        }

        if (Input.GetKey(KeyCode.D))
        {
            bodytransform.Rotate(0, -rotationMultiplier, 0);
        }

        
    }
}
