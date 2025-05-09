# gradio_interface.py
import gradio as gr
from quiz_generator import QuizGenerator
import json


class GradioInterface:
    """기본적인 Gradio UI 인터페이스 클래스"""

    def __init__(self, quiz_generator: QuizGenerator):
        self.quiz_generator = quiz_generator
        self.css = """
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        """

    def generate_quiz(self, content, mc_count, ox_count):
        """퀴즈 생성 함수"""
        try:
            result = self.quiz_generator.generate_quiz(content, int(mc_count), int(ox_count))
            return result
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return f"오류 발생: {str(e)}"

    def build_interface(self):
        """기본 Gradio 인터페이스 구성"""
        with gr.Blocks(css=self.css) as demo:
            gr.Markdown("## 교육 콘텐츠 문제 생성기")

            with gr.Row():
                with gr.Column(scale=3):
                    txt = gr.Textbox(
                        label="교육 콘텐츠 입력",
                        placeholder="교과서나 학습 자료의 내용을 붙여넣으세요.",
                        lines=10
                    )

                with gr.Column(scale=1):
                    mc_slider = gr.Slider(
                        minimum=1, maximum=2, value=2, step=1,
                        label="객관식 문제 수"
                    )
                    ox_slider = gr.Slider(
                        minimum=1, maximum=2, value=2, step=1,
                        label="OX 문제 수"
                    )
                    submit_btn = gr.Button("문제 생성하기")

            output_json = gr.JSON(label="생성된 JSON 문제")

            submit_btn.click(
                fn=self.generate_quiz,
                inputs=[txt, mc_slider, ox_slider],
                outputs=[output_json]
            )

        return demo

    def launch(self, server_port: int = 7860, server_name: str = "0.0.0.0"):
        """서버 시작"""
        demo = self.build_interface()
        print("Gradio 서버 시작 중...")
        demo.launch(server_port=server_port, server_name=server_name)