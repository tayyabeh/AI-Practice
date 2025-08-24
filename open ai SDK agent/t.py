from agents import Agent , Runner, handoff, RunContextWrapper, HandoffInputData
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel, EmailStr
import asyncio
import os
os.environ["GEMINI_API_KEY"] = ""
print(os.getenv("GEMINI_API_KEY")) 

class BillingProblem(BaseModel):
    invoice_no: str
    problem_type: str

# function input filter function  
def billing_input_filter(input):
    # The input is HandoffInputData with new_items containing HandoffOutputItem
    try:
        if hasattr(input, 'new_items') and input.new_items:
            handoff_item = input.new_items[-1]  # This is a HandoffOutputItem
            
            # Access the actual data from the HandoffOutputItem
            if hasattr(handoff_item, 'raw_item'):
                raw_data = handoff_item.raw_item
                
                # The raw_item is a dictionary, so we need to access it like a dict
                if isinstance(raw_data, dict):
                    invoice_no = raw_data.get('invoice_no')
                    if invoice_no is None:
                        raise ValueError(f"Invoice number not found in dict. Available keys: {list(raw_data.keys())}")
                    
                    # Validate invoice number length
                    if len(str(invoice_no)) != 6:
                        raise ValueError("Your invoice should be a 6 digit number")
                        
                # If it's already a BillingProblem object
                elif hasattr(raw_data, 'invoice_no'):
                    if len(raw_data.invoice_no) != 6:
                        raise ValueError("Your invoice should be a 6 digit number")
                else:
                    raise ValueError(f"Unexpected data type: {type(raw_data)}")
                    
            else:
                raise ValueError("HandoffOutputItem doesn't have raw_item attribute")
                
        else:
            raise ValueError("No new_items found in handoff input")
            
    except Exception as e:
        # Debug output
        print(f"Debug: Input type: {type(input)}")
        if hasattr(input, 'new_items') and input.new_items:
            handoff_item = input.new_items[-1]
            if hasattr(handoff_item, 'raw_item'):
                raw_data = handoff_item.raw_item
                print(f"Debug: raw_item type: {type(raw_data)}")
                if isinstance(raw_data, dict):
                    print(f"Debug: raw_item keys: {list(raw_data.keys())}")
                    print(f"Debug: raw_item values: {raw_data}")
        raise ValueError(f"Error processing billing input: {str(e)}")
    
    return input

billing_agent = Agent(
    name="Billing Agent",
    instructions=f'''
    {RECOMMENDED_PROMPT_PREFIX} 
    You are a helpful billing assistant agent and your work is to address user bills
    ''',
    model="litellm/gemini/gemini-1.5-flash"
)
def on_handoff_callback_billing(ctx : RunContextWrapper , input: BillingProblem):
    return "Yes the technical Agent is called!!"

billing_handoff = handoff(
    agent=billing_agent,    
    on_handoff=on_handoff_callback_billing,
    input_type=BillingProblem,
    input_filter=billing_input_filter    
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=f'''
    {RECOMMENDED_PROMPT_PREFIX} 
    You are a helpful assistant and your work is to address user queries.
    
    When a user mentions billing issues or invoice problems:
    1. Extract the invoice number from their message (remove any # symbols)
    2. Create a BillingProblem with the invoice number and problem type
    3. Handoff to Billing Agent
    
    When a user needs technical support:
    1. Handoff to Technical Support Agent
    
    For billing handoffs, use this format:
    - invoice_no: should be the invoice number (digits only, no # symbol)
    - problem_type: describe the type of billing issue
    ''',
    model="litellm/gemini/gemini-1.5-flash",
    handoffs=[billing_handoff]
)

async def main():
    result = await Runner.run(
        triage_agent, 
        input="I have a question about a charge on my invoice #987654."
    )
    print(result.final_output)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())