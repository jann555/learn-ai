# Call the OpenAI API with your prompt and print the response
from decouple import config
import openai

openai.api_base = "https://openai.vocareum.com/v1"

# Define your OpenAI API key
openai.api_key = OPEN_AI_KEY = config("OPEN_AI_KEY")

# Create variables to store the user inputs
restaurant_name = "Alinea"
cuisine_type = "new american"

prompt_template = (f"Provide a summary of customer sentiments for {restaurant_name}, focusing on their "
                   f"{cuisine_type} dishes. Highlight key sentiments and mention any standout dishes or services. ")


def generate_restaurant_review(prompt):
    try:
        # Calling the OpenAI API with a system message and our prompt in the user message content
        # Use openai.ChatCompletion.create for openai < 1.0
        # openai.chat.completions.create for openai > 1.0
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a restaurant critic. You are writing about reviews of restaurants. "
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # The response is a JSON object containing more information than the generated review. We want to return only
        # the message content
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


# Generating the response from the model
review_summary = generate_restaurant_review(prompt_template)

# Printing the output.
print("Generated review:")
print(review_summary)
