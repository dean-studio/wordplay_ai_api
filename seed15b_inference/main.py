from quiz_generator import QuizGenerator
from gradio_interface import GradioInterface

def main():
    """메인 함수"""
    quiz_generator = QuizGenerator()
    interface = GradioInterface(quiz_generator)
    interface.launch()

if __name__ == "__main__":
    main()