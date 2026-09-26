from MainCore.statecore import *
result = run_ranked_tool(
    "what time is it"
)

print(result)
print(
    run_ranked_tool(
        "add two numbers",
        {
            "a": 5,
            "b": 7
        }
    )
)