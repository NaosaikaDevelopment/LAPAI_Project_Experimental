using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections;
using UnityEngine.Networking;
using System.Text;
using System.Diagnostics;

public class AIchatRespons : MonoBehaviour
{
    [Header("UI References")]
    //optional - Set as you want, modify as you want
    public TMP_InputField userInputField;   // drag from Inspector
    public TMP_Text aiResponseText;         // drag from Inspector
    public Button sendButton;               // drag from Inspector

    [Header("Server Settings")]
    private string apiUrl = "http://localhost:3440/v1/chat/completions";
    private string modelName = "Llama-3.2-3B-Instruct-Hybrid"; //from setting 1MainModelName.txt as you follow the guide; LAPAI_Guide_v1.3.pptx


    private void Start()
    {
        sendButton.onClick.AddListener(OnSendClicked);
    }

    private void OnSendClicked()
    {
        string userInput = userInputField.text;
        if (!string.IsNullOrEmpty(userInput))
        {
            StartCoroutine(SendToLocalAI(userInput));
            userInputField.text = "";
        }
    }

    private IEnumerator SendToLocalAI(string userInput)
    {
        string jsonData = @"{
            ""model"": """ + modelName + @""",
            ""messages"": [
                { ""role"": ""user"", ""content"": """ + userInput + @""" }
            ]
        }";

        using (UnityWebRequest request = new UnityWebRequest(apiUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            var apiKey = "lemonade"; //its actually optional but better than nothing
            request.SetRequestHeader("Authorization", "Bearer " + apiKey);

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string json = request.downloadHandler.text;
                UnityEngine.Debug.Log("Raw Response: " + json);

                // Parse response JSON
                LocalAIResponse parsed = JsonUtility.FromJson<LocalAIResponse>(json);

                if (parsed != null && parsed.choices != null && parsed.choices.Length > 0)
                {
                    ActionMessage msg = parsed.choices[0].message;

                    // Command AI to action
                    if (msg.execute)
                    {
                        //For example:
                        if (msg.command == "walk")
                        {
                            UnityEngine.Debug.Log("AI says: WALK");
                            // GetComponent<CharacterController>().Move(Vector3.forward * Time.deltaTime * 5f);
                        }
                        else if (msg.command == "jump")
                        {
                            UnityEngine.Debug.Log("AI says: JUMP");
                            // GetComponent<Animator>().SetTrigger("Jump");
                        }
                        // add anything you wat here, base from your command to AI, AI would automatic set as you order, if shoot AI will shoot, if walk AI will walk, anything
                    }
                    else
                    {
                        UnityEngine.Debug.Log("AI decided not to execute command: " + msg.reason);
                    }
                }
                else
                {
                    aiResponseText.text = "No response";
                }

            }
            else
            {
                aiResponseText.text = "Error: " + request.error;
            }
        }
    }

    // ==== Data classes for parsing JSON ====
    [System.Serializable]
    public class ActionMessage
    {
        public string command;
        public bool execute;
        public string reason;
    }

    [System.Serializable]
    public class Choice
    {
        public int index;
        public ActionMessage message;
        public string finish_reason;
    }

    [System.Serializable]
    public class LocalAIResponse
    {
        public string id;
        public string @object;
        public Choice[] choices;
    }


}
