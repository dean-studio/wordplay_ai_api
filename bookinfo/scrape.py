from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import re
from selenium.common.exceptions import TimeoutException, WebDriverException


def scrape_book_info(url, max_retries=3):
    for retry in range(max_retries):
        try:
            print(f"시도 {retry + 1}/{max_retries}...")

            # Chrome 옵션 설정 - 더 가벼운 설정
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")  # 새로운 헤드리스 모드 사용
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            # JavaScript 비활성화 제거 (동적 콘텐츠가 필요할 수 있음)
            # chrome_options.add_argument("--disable-javascript")
            # 이미지 로딩 활성화 (일부 사이트는 이미지 로딩이 필요할 수 있음)
            # chrome_options.add_argument("--blink-settings=imagesEnabled=false")
            chrome_options.add_argument("--window-size=1280,720")  # 작은 창 크기

            # 사용자 에이전트 설정
            chrome_options.add_argument(
                "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

            # 불필요한 로깅 제거
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

            # 페이지 리소스 제한 - 일부 제한 완화
            chrome_options.add_experimental_option('prefs', {
                'profile.default_content_setting_values': {
                    'plugins': 2,  # 플러그인 차단
                    'popups': 2,  # 팝업 차단
                    'geolocation': 2,  # 위치 차단
                    'notifications': 2,  # 알림 차단
                    'auto_select_certificate': 2,  # 인증서 선택 차단
                    'fullscreen': 2,  # 전체 화면 차단
                    'mouselock': 2,  # 마우스 잠금 차단
                    'mixed_script': 2,  # 혼합 스크립트 차단
                    'media_stream': 2,  # 미디어 스트림 차단
                    'media_stream_mic': 2,  # 마이크 차단
                    'media_stream_camera': 2,  # 카메라 차단
                    'protocol_handlers': 2,  # 프로토콜 핸들러 차단
                    'ppapi_broker': 2,  # PPAPI 브로커 차단
                    'automatic_downloads': 2,  # 자동 다운로드 차단
                    'midi_sysex': 2,  # MIDI 차단
                    'push_messaging': 2,  # 푸시 메시지 차단
                    'ssl_cert_decisions': 2,  # SSL 인증서 결정 차단
                    'metro_switch_to_desktop': 2,  # 메트로 전환 차단
                    'protected_media_identifier': 2,  # 보호된 미디어 식별자 차단
                    'app_banner': 2,  # 앱 배너 차단
                    'site_engagement': 2,  # 사이트 참여 차단
                    'durable_storage': 2  # 내구성 있는 스토리지 차단
                }
            })

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            # 타임아웃 설정
            driver.set_page_load_timeout(360)  # 페이지 로드 타임아웃 360초

            book_info = {
                'isbn': None,
                'title': None,
                'subtitle': None,
                'cover_image_url': None,
                'description_image_url': None,  # 추가 설명 이미지 URL
                'description': None,  # 책 소개 내용
                'curriculum_connection': None,  # 교과 연계 정보
                'category': None
            }

            try:
                print("페이지 로딩 시작...")
                start_time = time.time()

                # 페이지 요청
                driver.get(url)

                # 로딩 시간 측정 및 출력
                load_time = time.time() - start_time
                print(f"페이지 로딩 완료: {load_time:.2f}초")

                # 페이지 소스 가져오기
                page_source = driver.page_source
                print(f"페이지 소스 길이: {len(page_source)} 바이트")

                # HTML 일부 출력 (디버깅용)
                print("\n--- 페이지 소스 일부 ---")
                print(page_source[:1000] + "...")
                print("--- 페이지 소스 일부 끝 ---\n")

                # HTML 파싱 방식으로 필요한 정보 추출
                print("HTML 파싱으로 정보 추출 시작...")

                # 1. ISBN 파싱
                isbn_pattern = r'<th[^>]*>ISBN</th>\s*<td[^>]*>(\d+)</td>'
                isbn_match = re.search(isbn_pattern, page_source)
                if isbn_match:
                    book_info['isbn'] = isbn_match.group(1)
                    print(f"ISBN 추출 성공: {book_info['isbn']}")
                else:
                    print("ISBN 패턴 매치 실패")

                # 2. 제목 파싱
                title_pattern = r'<span[^>]*class="prod_title"[^>]*>(.*?)</span>'
                title_match = re.search(title_pattern, page_source)
                if title_match:
                    book_info['title'] = title_match.group(1).strip()
                    print(f"제목 추출 성공: {book_info['title']}")
                else:
                    print("제목 패턴 매치 실패")

                # 3. 부제목 파싱 (여러 패턴 시도)
                print("\n부제목 추출 시도:")
                subtitle_patterns = [
                    r'<div[^>]*class="auto_overflow_inner"[^>]*>.*?<span[^>]*class="prod_desc"[^>]*>(.*?)</span>',
                    r'<span[^>]*class="prod_desc"[^>]*>(.*?)</span>',
                    r'<div[^>]*id="subTitleArea"[^>]*>.*?<span[^>]*class="prod_desc"[^>]*>(.*?)</span>'
                ]

                for i, pattern in enumerate(subtitle_patterns):
                    subtitle_match = re.search(pattern, page_source, re.DOTALL)
                    if subtitle_match:
                        subtitle_text = subtitle_match.group(1).strip()
                        print(f"부제목 패턴 {i + 1} 매치 성공: {subtitle_text}")
                        if '|' in subtitle_text:
                            book_info['subtitle'] = subtitle_text.split('|')[0].strip()
                            print(f"부제목 추출 성공 (| 이후 제거): {book_info['subtitle']}")
                        else:
                            book_info['subtitle'] = subtitle_text
                            print(f"부제목 추출 성공: {book_info['subtitle']}")
                        break
                    else:
                        print(f"부제목 패턴 {i + 1} 매치 실패")

                if not book_info['subtitle']:
                    print("모든 부제목 추출 패턴 실패")

                    # 페이지 소스에서 관련 부분 검색 (디버깅용)
                    prod_desc_index = page_source.find('prod_desc')
                    if prod_desc_index != -1:
                        context = page_source[max(0, prod_desc_index - 100):prod_desc_index + 500]
                    else:
                        print("'prod_desc' 클래스를 페이지에서 찾을 수 없음")

                # 4. 커버 이미지 URL 파싱
                image_pattern = r'<div[^>]*class="portrait_img_box[^"]*"[^>]*>.*?<img[^>]*src="([^"]+)"[^>]*>'
                image_match = re.search(image_pattern, page_source, re.DOTALL)
                if image_match:
                    book_info['cover_image_url'] = image_match.group(1)
                    print(f"커버 이미지 URL 추출 성공: {book_info['cover_image_url']}")
                else:
                    print("커버 이미지 URL 패턴 매치 실패")

                # 5. 추가 설명 이미지 URL 파싱
                desc_image_pattern = r'<div[^>]*class="product_detail_area detail_img"[^>]*>.*?<img[^>]*src="([^"]+)"[^>]*>'
                desc_image_match = re.search(desc_image_pattern, page_source, re.DOTALL)
                if desc_image_match:
                    book_info['description_image_url'] = desc_image_match.group(1)
                    print(f"추가 설명 이미지 URL 추출 성공: {book_info['description_image_url']}")
                else:
                    print("추가 설명 이미지 URL 패턴 매치 실패, 대체 패턴 시도")
                    # 대체 패턴 시도
                    alt_desc_image_pattern = r'<div[^>]*class="product_detail_area[^"]*"[^>]*>.*?<div[^>]*class="inner"[^>]*>.*?<img[^>]*src="([^"]+)"[^>]*>'
                    alt_desc_image_match = re.search(alt_desc_image_pattern, page_source, re.DOTALL)
                    if alt_desc_image_match:
                        book_info['description_image_url'] = alt_desc_image_match.group(1)
                        print(f"추가 설명 이미지 URL 추출 성공 (대체 패턴): {book_info['description_image_url']}")
                    else:
                        print("대체 패턴으로도 추가 설명 이미지 URL 추출 실패")

                # 6. 설명(description)과 교과 연계(curriculum_connection) 파싱
                print("\n설명 및 교과 연계 정보 추출 시도:")
                intro_bottom_pattern = r'<div[^>]*class="intro_bottom"[^>]*>(.*?)</div>'
                intro_bottom_match = re.search(intro_bottom_pattern, page_source, re.DOTALL)

                if intro_bottom_match:
                    intro_bottom_html = intro_bottom_match.group(1)
                    print("intro_bottom HTML 발견")
                    print(f"intro_bottom HTML 길이: {len(intro_bottom_html)} 바이트")
                    print(f"intro_bottom HTML 미리보기: {intro_bottom_html[:200]}...")

                    # 모든 info_text 추출
                    info_texts = re.findall(r'<div[^>]*class="info_text[^"]*"[^>]*>(.*?)</div>', intro_bottom_html,
                                            re.DOTALL)

                    if info_texts:
                        print(f"info_text 요소 {len(info_texts)}개 발견")

                        # description: 모든 info_text 내용을 합침
                        description_parts = []
                        curriculum_data = None

                        for i, info_text in enumerate(info_texts):
                            print(f"\ninfo_text {i + 1} 미리보기: {info_text[:100]}...")

                            # HTML 태그 제거 (단, <br>는 개행문자로 변환)
                            clean_text = re.sub(r'<br\s*/?>', '\n', info_text)
                            clean_text = re.sub(r'<[^>]*>', '', clean_text)
                            clean_text = clean_text.strip()

                            if clean_text:
                                description_parts.append(clean_text)
                                print(f"정제된 텍스트 {i + 1} 미리보기: {clean_text[:100]}...")

                                # 교과 연계 정보 확인
                                if '초등 교과 연계' in clean_text:
                                    print(f"교과 연계 정보 발견 in info_text {i + 1}")

                                    # 교과 연계 부분만 추출
                                    curriculum_text = clean_text
                                    curriculum_parts = curriculum_text.split('초등 교과 연계')

                                    if len(curriculum_parts) > 1:
                                        curriculum_lines = curriculum_parts[1].strip()
                                        print(f"교과 연계 부분: {curriculum_lines}")

                                        if curriculum_lines.startswith('★'):
                                            curriculum_lines = curriculum_lines.split('★', 2)[-1].strip()
                                            print(f"★ 제거 후: {curriculum_lines}")

                                        # 각 학년/학기별 항목 파싱
                                        curriculum_items = re.findall(r'\[(.*?)\](.*?)(?=\[|$)', curriculum_lines,
                                                                      re.DOTALL)

                                        if curriculum_items:
                                            print(f"학년/학기별 항목 {len(curriculum_items)}개 발견:")
                                            curriculum_dict = {}

                                            for grade_term, content in curriculum_items:
                                                grade_term = grade_term.strip()
                                                content = content.strip()
                                                curriculum_dict[grade_term] = content
                                                print(f"  {grade_term}: {content}")

                                            curriculum_data = curriculum_dict
                                        else:
                                            print("학년/학기별 항목 패턴 매치 실패")
                                    else:
                                        print("'초등 교과 연계' 분리 실패")

                        # 최종 description 생성
                        if description_parts:
                            book_info['description'] = '\n\n'.join(description_parts)
                            print(f"설명(description) 추출 성공: {len(book_info['description'])} 바이트")
                        else:
                            print("유효한 description 부분 없음")

                        # 교과 연계 정보가 있으면 JSON으로 저장
                        if curriculum_data:
                            book_info['curriculum_connection'] = json.dumps(curriculum_data, ensure_ascii=False)
                            print(f"교과 연계(curriculum_connection) 추출 성공: {book_info['curriculum_connection']}")
                        else:
                            print("교과 연계 데이터 추출 실패")
                    else:
                        print("info_text 요소를 찾지 못함")

                        # 페이지 소스에서 관련 부분 검색 (디버깅용)
                        info_text_index = page_source.find('info_text')
                        if info_text_index != -1:
                            context = page_source[max(0, info_text_index - 100):info_text_index + 500]
                            print(f"\ninfo_text 관련 HTML 부분:\n{context}\n")
                        else:
                            print("'info_text' 클래스를 페이지에서 찾을 수 없음")
                else:
                    print("intro_bottom 요소를 찾지 못함")

                    # 페이지 소스에서 관련 부분 검색 (디버깅용)
                    intro_bottom_index = page_source.find('intro_bottom')
                    if intro_bottom_index != -1:
                        context = page_source[max(0, intro_bottom_index - 100):intro_bottom_index + 500]
                        print(f"\nintro_bottom 관련 HTML 부분:\n{context}\n")
                    else:
                        print("'intro_bottom' 클래스를 페이지에서 찾을 수 없음")

                        # 추가 검색: 다른 가능한 컨테이너 찾기
                        print("\n대체 컨테이너 검색:")
                        alternative_containers = [
                            'product_detail_area book_intro',
                            'box_detail_content',
                            'intro_book'
                        ]

                        for container in alternative_containers:
                            container_index = page_source.find(container)
                            if container_index != -1:
                                print(f"'{container}' 발견! 해당 부분 미리보기:")
                                context = page_source[max(0, container_index - 50):container_index + 500]
                                print(context[:300] + "...")

                                # 대체 description 추출 시도
                                if not book_info['description'] and container == 'product_detail_area book_intro':
                                    alt_desc_pattern = r'<div[^>]*class="product_detail_area book_intro"[^>]*>.*?<div[^>]*class="intro_book"[^>]*>(.*?)</div>'
                                    alt_desc_match = re.search(alt_desc_pattern, page_source, re.DOTALL)
                                    if alt_desc_match:
                                        raw_desc = alt_desc_match.group(1).strip()
                                        # HTML 태그 제거
                                        clean_desc = re.sub(r'<br\s*/?>', '\n', raw_desc)
                                        clean_desc = re.sub(r'<[^>]*>', '', clean_desc)
                                        clean_desc = clean_desc.strip()

                                        if clean_desc:
                                            book_info['description'] = clean_desc
                                            print(f"대체 방법으로 description 추출 성공: {len(book_info['description'])} 바이트")
                            else:
                                print(f"'{container}'를 페이지에서 찾을 수 없음")

                # 7. 카테고리 파싱
                category_list_pattern = r'<ul\s+class="intro_category_list">(.*?)</ul>'
                category_list_match = re.search(category_list_pattern, page_source, re.DOTALL)

                if category_list_match:
                    category_list_html = category_list_match.group(1)
                    print("카테고리 목록 HTML 발견")

                    # 카테고리 항목 파싱
                    category_items = re.findall(r'<li\s+class="category_list_item">(.*?)</li>', category_list_html,
                                                re.DOTALL)
                    print(f"카테고리 항목 {len(category_items)}개 발견")

                    category_paths = []

                    for idx, item in enumerate(category_items):
                        # 각 항목에서 카테고리 링크 텍스트 추출
                        links = re.findall(r'<a\s+href="[^"]*"\s+class="intro_category_link">(.*?)</a>', item)

                        path = []
                        for link_text in links:
                            clean_text = re.sub(r'<.*?>', '', link_text).strip()  # HTML 태그 제거
                            if clean_text:
                                path.append(clean_text)

                        if path:
                            category_paths.append(path)
                            print(f"경로 {idx + 1} 추출: {' > '.join(path)}")

                    if category_paths:
                        book_info['category'] = json.dumps(category_paths, ensure_ascii=False)
                        print(f"카테고리 파싱 성공: {len(category_paths)}개 경로")
                else:
                    print("카테고리 목록 HTML을 찾지 못함")

                # 정보 추출 요약
                print("\n--- 정보 추출 결과 요약 ---")
                for key, value in book_info.items():
                    if value:
                        if key in ['description', 'category', 'curriculum_connection']:
                            print(f"{key}: [데이터 있음, 길이: {len(value)} 바이트]")
                        else:
                            print(f"{key}: {value}")
                    else:
                        print(f"{key}: 추출 실패")

                # 주요 정보가 추출되었는지 확인
                required_info = ['isbn', 'title', 'cover_image_url']
                all_required_extracted = all(book_info[key] is not None for key in required_info)

                if all_required_extracted:
                    print("필수 정보 추출 성공!")
                    return book_info

                # 일부 정보만 추출된 경우
                print("일부 정보만 추출됨. 추출된 정보 반환.")
                return book_info

            except TimeoutException as te:
                print(f"페이지 로딩 타임아웃: {te}")
                if retry == max_retries - 1:  # 마지막 시도에서만 부분 정보 반환
                    return book_info
            except Exception as e:
                print(f"스크래핑 중 오류 발생: {e}")
                if retry == max_retries - 1:  # 마지막 시도에서만 부분 정보 반환
                    return book_info
            finally:
                end_time = time.time()
                total_time = end_time - start_time
                print(f"총 실행 시간: {total_time:.2f}초")
                driver.quit()

        except WebDriverException as wde:
            print(f"WebDriver 오류: {wde}")
            if retry == max_retries - a:
                return book_info
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            if retry == max_retries - 1:
                return book_info

    return book_info  # 모든 시도 실패 시 빈 정보 반환


if __name__ == "__main__":
    url = "https://product.kyobobook.co.kr/detail/S000216249638"

    print("도서 정보 스크래핑 시작")
    book_info = scrape_book_info(url)

    print("\n--- 최종 추출 결과 ---")
    for key, value in book_info.items():
        if value:
            if key == 'category' or key == 'curriculum_connection':
                # JSON 문자열을 파싱하여 보기 좋게 출력
                try:
                    parsed_data = json.loads(value)
                    print(f"{key}:")
                    if isinstance(parsed_data, list):
                        for i, path in enumerate(parsed_data, 1):
                            print(f"  경로 {i}: {' > '.join(path)}")
                    elif isinstance(parsed_data, dict):
                        for grade_term, content in parsed_data.items():
                            print(f"  {grade_term}: {content}")
                    else:
                        print(f"  {parsed_data}")
                except:
                    print(f"{key}: {value}")
            elif key == 'description':
                # description은 줄바꿈을 유지하면서 일부만 출력
                preview = value[:100] + '...' if len(value) > 100 else value
                preview = preview.replace('\n', '\\n')
                print(f"{key}: {preview}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: 추출 실패")