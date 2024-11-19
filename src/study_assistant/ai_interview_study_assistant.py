import random
from decouple import config
import openai
import json
import datetime

# Set up your OpenAI API key
openai.api_key = config("OPEN_AI_KEY")
openai.api_base = "https://openai.vocareum.com/v1"
MODEL_NAME = "gpt-3.5-turbo-instruct"
MAX_TOKENS = 2500


# List of questions from the provided page

def ask_question(question):
    print(question)
    answer = input("Your answer: ")
    return answer


def load_questions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['questions']


def save_questions_for_review(questions, file_path):
    with open(file_path, 'a') as file:
        json.dump(questions, file, indent=4)
        return True


def select_random_questions(selection, questions):
    chosen = []
    size = len(questions) - 1
    i = 0

    if selection > size:
        selection = size

    while i < selection:
        question = questions[random.randint(0, size)]
        if question not in chosen:
            chosen.append(question)
            i += 1
    return chosen


def analyze_answer(question, answer):
    prompt = (f"Question: {question}\nAnswer: {answer}\nState whether the answer is correct or incorrect. Example "
              f"Result: correct or Result: incorrect. Then proceed to Provide feedback on the quality of the answer")
    response = openai.Completion.create(
        engine=MODEL_NAME,
        prompt=prompt,
        max_tokens=MAX_TOKENS
    )
    feedback = response.choices[0].text.strip()
    return feedback


def study_assistant(file_source):
    loaded_questions = load_questions(file_source)
    questions_number = input(f"How many questions do you want to answer? Available questions: {len(loaded_questions)}\n")
    selected_questions = select_random_questions(int(questions_number), loaded_questions)
    correct_answers = 0
    ts = datetime.date.today()
    practice_review = {f'questions-{ts}': []}

    # Loop through all the questions, analyze the results and provide feedback for each question
    for i, question in enumerate(selected_questions):
        question_for_review = {'question': "", 'answer': "", 'feedback': ""}
        answer = ask_question(f'{i+1}.- {question}')
        feedback = analyze_answer(question, answer)
        if "incorrect".lower() not in feedback:
            correct_answers += 1
        else:
            # Question added to review stack
            question_for_review['question'] = question
            question_for_review['answer'] = answer
            question_for_review['feedback'] = feedback
            practice_review[f'questions-{ts}'].append(question_for_review)

        print(f'Score [{correct_answers}/{i + 1}] correct answers')
        print(f"Feedback: {feedback}\n")

    # Save questions that need reviewing
    if practice_review[f'questions-{ts}']:
        save_questions_for_review(practice_review, f'review-{file_source}')
    # Display final results to the user
    final_score = f'[{correct_answers}/{questions_number}]'
    print(f'Final Score {final_score} correct answers')
    print('You have reached the end. I hope you learned something')


def main():
    path = 'questions/'
    study_assistant(f'{path}java_questions.json')


if __name__ == "__main__":
    main()
