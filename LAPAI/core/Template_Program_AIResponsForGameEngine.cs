using UnityEngine;
using UnityEngine.UI;
using TMPro; 
using System.Collections;
using UnityEngine.Networking;
using System.Text;

public class AIRespons : MonoBehaviour
{
    [Header("UI References")]
    public TMP_InputField userInputField;   // drag from Inspector
    public TMP_Text aiResponseText;         // drag from Inspector
    public Button sendButton;               // drag from Inspector

    [Header("Server Settings")]
    private string apiUrl = "Place-your-Url-here";
    private string modelName = "Place-your-main-model-here"; //from setting 1MainModelName.txt as you follow the guide; LAPAI_Guide_v1.3.pptx

    string persona = ""; // here if you wanna set persona but make sure PersonaAI.txt on Settings folder its empty

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
                { ""role"": ""system"", ""content"": """ + persona + @""" },
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
                Debug.Log("Raw Response: " + json);

                // Parse response JSON
                LocalAIResponse parsed = JsonUtility.FromJson<LocalAIResponse>(json);
                if (parsed != null && parsed.choices != null && parsed.choices.Length > 0)
                {
                    aiResponseText.text = parsed.choices[0].message.content;
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
    public class LocalAIResponse
    {
        public Choice[] choices;
    }

    [System.Serializable]
    public class Choice
    {
        public Message message;
    }

    [System.Serializable]
    public class Message
    {
        public string role;
        public string content;
    }
}
