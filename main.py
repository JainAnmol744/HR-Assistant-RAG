from hr_assistant import config
from hr_assistant.pipeline import ask,build_hr_assistant

def main():
    print("building the HR policy assistant")
    agent = build_hr_assistant()
    print("Assistant Ready!\n")

    demo_questions = [
        "how many paid annual leave days do I get ?",
        "what is the notice period during probation?",
        "Can I work from home every day ?"
    ]

    for question in demo_questions:
        print("="*60)
        print('Question:',question)
        print("="*60)
        answer = ask(agent, question)
        print("ANSWER:", answer)
        print("="*60)


if __name__ == "__main__":
    main()