# gradio_interface.py
import gradio as gr
from quiz_generator import QuizGenerator
import json


class GradioInterface:
    """현대적이고 고급스러운 Gradio UI 인터페이스 클래스"""

    def __init__(self, quiz_generator: QuizGenerator):
        self.quiz_generator = quiz_generator
        self.css = """
        /* 전체 테마 색상 변수 */
        :root {
            --primary-color: #3f51b5;
            --secondary-color: #f50057;
            --background-color: #f9f9f9;
            --card-color: #ffffff;
            --text-color: #333333;
            --border-color: #e0e0e0;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --success-color: #4caf50;
        }

        /* 기본 스타일 */
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
            font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* 전체 컨테이너 */
        .gradio-container {
            width: 100% !important;
            min-height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
            max-width: none !important;
            background-color: var(--background-color);
        }

        /* 헤더 스타일 */
        .gradio-container h1, .gradio-container h2, .gradio-container h3 {
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            position: relative;
            padding-bottom: 0.5rem;
        }

        .gradio-container h2 {
            font-size: 2rem;
            letter-spacing: -0.5px;
        }

        .gradio-container h2::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 60px;
            height: 4px;
            background-color: var(--secondary-color);
            border-radius: 2px;
        }

        /* 카드 스타일 */
        .gradio-container .card, .header-section, .input-card, .settings-card, .result-section, .info-section, .footer-section {
            border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 16px var(--shadow-color) !important;
            padding: 1.5rem !important;
            background-color: var(--card-color) !important;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 1.5rem;
        }

        .gradio-container .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px var(--shadow-color) !important;
        }

        /* 텍스트 영역 스타일 */
        .gradio-container textarea, .gradio-container input[type="text"] {
            border-radius: 8px !important;
            border: 2px solid var(--border-color) !important;
            padding: 12px !important;
            font-size: 1rem !important;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        .gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(63, 81, 181, 0.2) !important;
            outline: none !important;
        }

        /* 버튼 스타일 */
        .gradio-container button.primary, button.secondary-button, button.tool-button {
            background-color: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            border: none !important;
            transition: background-color 0.3s, transform 0.2s !important;
            text-transform: none !important;
            box-shadow: 0 4px 6px rgba(63, 81, 181, 0.2) !important;
            margin: 0.5rem 0;
            cursor: pointer;
        }

        .gradio-container button.secondary-button {
            background-color: white !important;
            color: var(--primary-color) !important;
            border: 1px solid var(--primary-color) !important;
            box-shadow: none !important;
        }

        .gradio-container button.tool-button {
            background-color: #f5f5f5 !important;
            color: var(--text-color) !important;
            box-shadow: none !important;
            padding: 8px 16px !important;
            font-size: 0.9rem !important;
        }

        .gradio-container button:hover {
            background-color: #303f9f !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(63, 81, 181, 0.3) !important;
        }

        .gradio-container button.secondary-button:hover {
            background-color: rgba(63, 81, 181, 0.1) !important;
        }

        .gradio-container button.tool-button:hover {
            background-color: #e0e0e0 !important;
        }

        /* 슬라이더 스타일 */
        .gradio-container input[type="range"] {
            -webkit-appearance: none;
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            background-image: linear-gradient(var(--primary-color), var(--primary-color));
            background-repeat: no-repeat;
        }

        .gradio-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--primary-color);
            cursor: pointer;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
            transition: background 0.3s, box-shadow 0.3s;
        }

        .gradio-container input[type="range"]::-webkit-slider-thumb:hover {
            background: #303f9f;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        }

        /* 코드 블록 스타일 */
        .gradio-container pre, .gradio-container code {
            border-radius: 8px !important;
            font-family: 'JetBrains Mono', 'D2Coding', monospace !important;
            background-color: #f5f5f5 !important;
            padding: 1rem !important;
            box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.1) !important;
            max-height: 500px !important;
            overflow-y: auto !important;
        }

        /* 로딩 애니메이션 */
        .progress-bar {
            position: relative;
            height: 4px;
            width: 100%;
            border-radius: 2px;
            overflow: hidden;
            background-color: #e0e0e0;
            margin: 1rem 0;
        }

        .progress-bar::after {
            content: "";
            position: absolute;
            height: 100%;
            width: 30%;
            background-color: var(--primary-color);
            animation: progress-animation 1.5s infinite ease-in-out;
            border-radius: 2px;
        }

        @keyframes progress-animation {
            0% { left: -30%; }
            100% { left: 100%; }
        }

        /* 레이아웃 스페이싱 */
        .container-wrapper {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .main-row {
            display: flex;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        /* 반응형 조정 */
        @media (max-width: 768px) {
            .container-wrapper {
                padding: 1rem;
            }

            .main-row {
                flex-direction: column;
            }

            .gradio-container h2 {
                font-size: 1.5rem;
            }
        }

        /* 결과 카드 스타일 */
        .result-card {
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 6px 16px var(--shadow-color);
            padding: 1.5rem;
            margin-top: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .result-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 6px;
            height: 100%;
            background-color: var(--success-color);
        }

        .output-label {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .output-label::before {
            content: "";
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: var(--success-color);
        }

        /* 웹폰트 추가 */
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        """

    def _process_quiz_response(self, content, mc_count, ox_count):
        """퀴즈 생성 결과를 처리하고 추가 정보 제공"""
        try:
            # 퀴즈 생성
            quiz_json = self.quiz_generator.generate_quiz(content, mc_count, ox_count)
            quiz_data = json.loads(quiz_json)

            # 통계 정보
            stats = {
                "총 문제 수": len(quiz_data),
                "객관식 문제 수": mc_count,
                "OX 문제 수": ox_count,
                "생성 상태": "성공"
            }

            return quiz_json, json.dumps(stats, ensure_ascii=False, indent=2)
        except Exception as e:
            error_msg = f"오류 발생: {str(e)}"
            return "{}", json.dumps({"생성 상태": "실패", "오류": error_msg}, ensure_ascii=False, indent=2)

    def build_interface(self):
        """향상된 Gradio 인터페이스 구성"""
        with gr.Blocks(css=self.css, theme=gr.themes.Default()) as demo:
            with gr.Column(elem_classes="container-wrapper"):
                # 헤더 섹션
                with gr.Group(elem_classes="header-section"):
                    gr.Markdown(
                        """
                        # 📚 교육 콘텐츠 문제 생성기

                        ### 교과서나 학습 자료의 내용을 분석하여 객관식 및 OX 문제를 자동으로 생성합니다.

                        > 하이브리드 AI 접근법으로 고품질 문제를 생성합니다. 파일 업로드를 지원하며, 텍스트 영역에 직접 입력할 수도 있습니다.
                        """
                    )

                # 메인 콘텐츠 섹션
                with gr.Row(elem_classes="main-row"):
                    # 입력 섹션
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes="input-card"):
                            gr.Markdown("### 📝 교육 콘텐츠 입력")

                            with gr.Tab("텍스트 입력"):
                                txt = gr.Textbox(
                                    placeholder="교과서나 학습 자료의 내용을 붙여넣으세요. 내용은 충분히 상세하게 입력할수록 더 좋은 문제가 생성됩니다.",
                                    show_label=False,
                                    lines=12,
                                    elem_classes="content-input"
                                )

                            with gr.Tab("파일 업로드"):
                                file_input = gr.File(label="텍스트 파일 업로드 (.txt)")
                                file_button = gr.Button("파일 내용 불러오기")

                    # 설정 및 제어 섹션
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes="settings-card"):
                            gr.Markdown("### ⚙️ 생성 설정")

                            with gr.Group():
                                gr.Markdown("#### 문제 유형 및 수량")
                                mc_slider = gr.Slider(
                                    minimum=1, maximum=2, value=2, step=1,
                                    label="객관식 문제 수",
                                    info="최대 2개까지 생성 가능"
                                )
                                ox_slider = gr.Slider(
                                    minimum=1, maximum=2, value=2, step=1,
                                    label="OX 문제 수",
                                    info="최대 2개까지 생성 가능"
                                )

                            with gr.Group(elem_classes="action-buttons"):
                                submit_btn = gr.Button("문제 생성하기")
                                clear_btn = gr.Button("모두 지우기")

                # 결과 섹션
                with gr.Group(visible=False, elem_classes="result-section") as result_container:
                    gr.Markdown("### 🎯 생성된 문제")

                    with gr.Row():
                        with gr.Column(scale=8):
                            output_json = gr.Code(
                                language="json",
                                label="📋 JSON 형식 문제",
                                elem_classes="result-json",
                                lines=15
                            )

                        with gr.Column(scale=2):
                            stats_output = gr.Code(
                                language="json",
                                label="📊 생성 통계",
                                elem_classes="stats-output",
                                lines=6
                            )

                    with gr.Row(elem_classes="action-row"):
                        copy_btn = gr.Button("JSON 복사")
                        download_btn = gr.Button("JSON 다운로드")

                # 하단 정보 섹션
                with gr.Group(elem_classes="info-section"):
                    gr.Markdown(
                        """
                        ### 📋 사용 가이드

                        1. **교육 콘텐츠 입력**: 텍스트 직접 입력 또는 파일 업로드를 통해 분석할 교육 콘텐츠를 제공합니다.
                        2. **생성 설정**: 원하는 객관식 문제와 OX 문제의 수를 설정합니다.
                        3. **문제 생성**: "문제 생성하기" 버튼을 클릭하여 AI가 콘텐츠를 분석하고 문제를 생성합니다.
                        4. **결과 활용**: 생성된 JSON 형식의 문제를 복사하거나 다운로드하여 활용합니다.

                        > **참고**: 더 자세한 내용이 포함된 교육 콘텐츠일수록 높은 품질의 문제가 생성됩니다.
                        """
                    )

                # 푸터 섹션
                with gr.Group(elem_classes="footer-section"):
                    gr.Markdown(
                        """
                        #### 교육 콘텐츠 문제 생성기 (v1.0.0) | 하이브리드 AI 접근법 | ©2025
                        """
                    )

            # 이벤트 핸들러
            def show_results():
                return gr.update(visible=True)

            def clear_inputs():
                return "", gr.update(value=None)

            def load_file_content(file_obj):
                if file_obj is None:
                    return ""

                try:
                    content = file_obj.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content = file_obj.decode('cp949')  # 한국어 Windows 인코딩 시도
                    except UnicodeDecodeError:
                        return "파일 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949 형식의 텍스트 파일을 업로드해주세요."

                return content

            # 이벤트 연결
            submit_btn.click(
                fn=lambda x, mc, ox: (self._process_quiz_response(x, int(mc), int(ox)), show_results()),
                inputs=[txt, mc_slider, ox_slider],
                outputs=[[output_json, stats_output], result_container]
            )

            clear_btn.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[txt, file_input]
            )

            file_button.click(
                fn=load_file_content,
                inputs=[file_input],
                outputs=[txt]
            )

            # 자바스크립트 기능 (복사 및 다운로드)
            copy_btn.click(
                None,
                _js="""
                function() {
                    const jsonText = document.querySelector('.result-json textarea').value;
                    navigator.clipboard.writeText(jsonText);
                    alert('JSON이 클립보드에 복사되었습니다.');
                    return [];
                }
                """
            )

            download_btn.click(
                None,
                _js="""
                function() {
                    const jsonText = document.querySelector('.result-json textarea').value;
                    const blob = new Blob([jsonText], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'generated_quiz_' + new Date().toISOString().slice(0,10) + '.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    return [];
                }
                """
            )

        return demo

    def launch(self, server_port: int = 7860, server_name: str = "0.0.0.0"):
        """서버 시작"""
        demo = self.build_interface()
        print("Gradio 서버 시작 중...")
        demo.launch(server_port=server_port, server_name=server_name)


# 이전 버전과의 호환성을 위해 별칭 생성
ModernGradioInterface = GradioInterface