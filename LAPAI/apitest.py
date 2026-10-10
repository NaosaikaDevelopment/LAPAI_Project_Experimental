from openai import OpenAI

client = OpenAI(base_url="http://localhost:2488/v1", api_key="Dummy" )
msg = "hallo"

reply = client.chat.completions.create(
    model="",
    messages=[{"role":"user","content": msg}]
)
print(reply.choices[0].message.content)