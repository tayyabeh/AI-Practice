from agents import Agent , Runner , function_tool
import os
import asyncio

# API key 
os.environ["GEMINI_API_KEY"] = ""
print(os.getenv("GEMINI_API_KEY")) 

# creating tools 
@function_tool
def get_weather(city:str)-> str:
    return f"The weather of {city} is sunny and warm its 70C"


# Creating an Agent
agent = Agent(
    name="Helpfull Assistant",
    instructions = "You are the helpful weather assistant",
    model = "litellm/gemini/gemini-1.5-flash",
    tools=[get_weather]
)


# Running an agent
# result = Runner.run_sync(agent , input="What is the capital of France?")
# print(result.final_output)

async def main():
    result = await Runner.run(agent , input="What is the weather of New York?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())