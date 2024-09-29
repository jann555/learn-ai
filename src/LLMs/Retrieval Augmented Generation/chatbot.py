from decouple import config
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import tiktoken
import numpy as np
from dateutil.parser import parse
import pandas as pd

import requests

OPEN_AI_KEY = config("OPEN_AI_KEY")
MAX_TOKENS = 150
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"

openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = OPEN_AI_KEY

ukraine_prompt = """
Question: "When did Russia invade Ukraine?"
Answer:
"""

twitter_prompt = """
Question: "Who owns Twitter?"
Answer:
"""

wiki_year_2022_events = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&titles=2022&explaintext=1&formatversion=2&format=json"


def get_answer(prompt: str, max_tokens: int):
    answer = openai.Completion.create(
        model=COMPLETION_MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens
    )["choices"][0]["text"].strip()
    return answer


def load_and_wrabgle(dataset_url):
    resp = requests.get(dataset_url)
    print(f'resp.status_code {resp.status_code}')
    # Load page text into a dataframe
    data_frame = pd.DataFrame()
    data_frame["text"] = resp.json()["query"]["pages"][0]["extract"].split("\n")

    # Clean up text to remove empty lines and headings
    data_frame = data_frame[(data_frame["text"].str.len() > 0) & (~data_frame["text"].str.startswith("=="))]

    # In some cases dates are used as headings instead of being part of the
    # text sample; adjust so dated text samples start with dates
    prefix = ""
    for (i, row) in data_frame.iterrows():
        # If the row already has " - ", it already has the needed date prefix
        if " – " not in row["text"]:
            try:
                # If the row's text is a date, set it as the new prefix
                parse(row["text"])
                prefix = row["text"]
            except:
                # If the row's text isn't a date, add the prefix
                row["text"] = prefix + " – " + row["text"]
    data_frame = data_frame[data_frame["text"].str.contains(" – ")].reset_index(drop=True)
    return data_frame


def generate_embeddings(df: pd.DataFrame, output_csv_file: str, embedding_model_name: str):
    """Generating Embeddings
    We'll use the `Embedding`
    tooling from OpenAI [documentation here](https://platform.openai.com/docs/guides/embeddings/embeddings)
    to create vectors representing each row of our custom dataset."""

    batch_size = 100
    embeddings = []
    for i in range(0, len(df), batch_size):
        # Send text data to OpenAI model to get embeddings
        response = openai.Embedding.create(
            input=df.iloc[i:i + batch_size]["text"].tolist(),
            engine=embedding_model_name
        )

        # Add embeddings to list
        embeddings.extend([data["embedding"] for data in response["data"]])

    # Add embeddings list to dataframe
    df["embeddings"] = embeddings

    # In order to avoid having to run that code again in the future, we'll save the generated embeddings as a CSV file.
    df.to_csv(output_csv_file)
    return df


def get_rows_sorted_by_relevance(question, df):
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """

    # Get embeddings for the question text
    question_embeddings = get_embedding(question, engine=EMBEDDING_MODEL_NAME)

    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(
        question_embeddings,
        df_copy["embeddings"].values,
        distance_metric="cosine"
    )

    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy


def create_prompt(question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count the number of tokens in the prompt template and question
    prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""

    current_token_count = len(tokenizer.encode(prompt_template)) + len(tokenizer.encode(question))

    context = []
    for text in get_rows_sorted_by_relevance(question, df)["text"].values:

        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), question)


def answer_question(
        question, df, max_prompt_tokens=1800, max_answer_tokens=150
):
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response, return the
    answer to the question according to an OpenAI Completion model

    If the model produces an error, return an empty string
    """

    prompt = create_prompt(question, df, max_prompt_tokens)

    try:
        response = openai.Completion.create(
            model=COMPLETION_MODEL_NAME,
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


print('Printing untrained answers')

initial_ukraine_answer = get_answer(ukraine_prompt, MAX_TOKENS)
print(initial_ukraine_answer)
#
initial_twitter_answer = get_answer(twitter_prompt, MAX_TOKENS)
print(initial_twitter_answer)

print('Preparing Data Set...')
# Step 1: Prepare Dataset
# Loading and Wrangling Data

try:
    data_f = pd.read_csv("embeddings.csv", index_col=0)
    data_f["embeddings"] = data_f["embeddings"].apply(eval).apply(np.array)
except:
    print("Creating Embedding and saving to CSV")
    # Get the Wikipedia page for "2022" since OpenAI's models stop in 2021
    dataset = load_and_wrabgle(wiki_year_2022_events)
    # Generating Embeddings
    data_f = generate_embeddings(dataset, "embeddings.csv", EMBEDDING_MODEL_NAME)
else:
    print('"Embedding is loaded to CSV"')

print('Answering Trained questions')
custom_ukraine_answer = answer_question(ukraine_prompt, data_f)
print(custom_ukraine_answer)
custom_twitter_answer = answer_question(twitter_prompt, data_f)
print(custom_twitter_answer)
