"""
内部训练模型的简单测试，不保存用户日志
husky gradio 的模型：添加api
1. 传入图片：调用处理图片的prompt, 缓存结果。只需要处理一次
2. 用户输入文字：调用处理文字的api (用户可以选择是否调用api，添加6个打勾的选项)
3. 添加api state 状态管理机制
nginx转发端口
"""
import torch
import gradio as gr
from PIL import Image
from peft import PeftModel
import argparse

from transformers import (
    LlamaTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

import sys
import os
import json
from husky.conversation import (
    get_conv_template,
)

from husky.video_transformers import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    get_index
)

import torchvision.transforms as T
from decord import VideoReader, cpu
from husky.compression import compress_module
from torchvision.transforms.functional import InterpolationMode
import requests
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

IGNORE_INDEX = -100
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"

DEFAULT_VIDEO_START_TOKEN = "<vid>"
DEFAULT_VIDEO_END_TOKEN = "</vid>"

API_MAPPING = {'visual_grounding': 'Grounding',
               'celebrity_recognition': '名人',
               'visual_entity_retrieval': '视觉检索',
               'entity_retrieval': '名词检索',
               'optical_character_recognition': 'OCR',
               'art_retrieval': '艺术品'}

def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory

def load_model(model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, lora_weights=None):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)

    if lora_weights is None:
        model = HuskyForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        kwargs["device_map"] = "auto"
        model = HuskyForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            lora_weights,
            **kwargs
        )

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    model = model.eval()
    return model, tokenizer

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def start():
    chat_state = get_conv_template("husky")
    video_list = []
    image_list = []
    if args.use_api and args.output_api_info:
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(
            interactive=True), gr.update(interactive=False), gr.update(interactive=True), gr.update(
            interactive=True), chat_state, video_list, image_list
    else:
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(
            interactive=True), gr.update(interactive=False), gr.update(
            interactive=True), chat_state, video_list, image_list

def upload_file(video, image, chat_state):
    video_list = []
    image_list = []
    if (video is None) and (image is None):
        return None, None, gr.update(interactive=True), chat_state, video_list, image_list, ''
    llm_message = chat.upload_file(video, image, chat_state, video_list, image_list)
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(value="Start Chatting",
                                                                                 interactive=False), chat_state, video_list, image_list

def load_video(video_path, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)
    # transform
    crop_size = 224
    scale_size = 224
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)
    ])
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    video = transform(images_group)
    return video

def load_image(image_file, input_size=448):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = transform(image)
    return image

##### gradio setting ###
def gradio_ask(user_message, chatbot, chat_state, api_check_box, image):
    """image is a path"""
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    prompt = chat.ask(user_message, chat_state, api_check_box, image)
    chatbot = chatbot + [[user_message, None]]
    if args.use_api and args.output_api_info:
        return '', chatbot, chat_state, prompt
    else:
        return '', chatbot, chat_state

def gradio_answer(chat_state, chatbot, video_list, image_list):
    llm_message = chat.answer(chat_state, video_list, image_list)
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, video_list, image_list

def gradio_reset(chat_state, video_list, image_list):
    if chat_state is not None:
        chat_state.messages = []
    if video_list is not None:
        video_list = []
    if image_list is not None:
        image_list = []

    if args.use_api and args.output_api_info:
        return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(
            placeholder='Type and press Enter', interactive=True), gr.update(value=None, interactive=True), gr.update(
            value="Upload Video/Image", interactive=True), chat_state, video_list, image_list
    else:
        return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(
            placeholder='Type and press Enter', interactive=True), gr.update(value="Upload Video/Image",
                                                                             interactive=True), chat_state, video_list, image_list

def gradio_reset_text_box(chat_state):
    if chat_state is not None:
        chat_state.messages = []

    if args.use_api and args.output_api_info:
        return None, gr.update(placeholder='Type and press Enter', interactive=True), gr.update(value=None,
                                                                                                interactive=True), chat_state, gr.update(
            value=None, interactive=True)
    else:
        return None, gr.update(placeholder='Type and press Enter', interactive=True), chat_state

#### gradio setting ###
class Chat:
    def __init__(
            self,
            model_path,
            device,
            num_gpus=1,
            load_8bit=False,
            temperature=0.7,
            max_new_tokens=512,
            lora_path=None,
    ):
        model, tokenizer = load_model(
            model_path, device, num_gpus, load_8bit=load_8bit, lora_weights=lora_path
        )
        self.model = model
        self.tokenizer = tokenizer
        num_queries = model.config.num_query_tokens

        self.device = device
        self.dtype = model.dtype

        stop_words = ["Human: ", "Assistant: ", "###", "\n\n"]
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.conv = get_conv_template("husky")

        self.image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMG_END_TOKEN
        self.video_query = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_END_TOKEN

        self.generation_config = GenerationConfig(
            bos_token_id=1,
            pad_token_id=0,
            do_sample=True,
            top_k=20,
            top_p=0.25,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria
        )

        # set api info
        self.image_api_info = []
        self.image_file = []
        self.text_api_info = []
        self.image_text_api_info = []

    @torch.no_grad()
    def upload_file(self, video, image, conv, video_list, image_list):
        if video != None and image == None:
            video_pixel_values = load_video(video)
            TC, H, W = video_pixel_values.shape
            video_pixel_values = video_pixel_values.reshape(TC // 3, 3, H, W).transpose(0, 1)  # [C, T, H, W]
            video_pixel_values = video_pixel_values.unsqueeze(0).to(self.device, dtype=self.dtype)
            assert len(video_pixel_values.shape) == 5
            video_language_model_inputs = self.model.extract_feature(video_pixel_values)
            video_list.append(video_language_model_inputs)
            conv.append_message(conv.roles[0], self.video_query + "\n")
            msg = "Received."
            return msg, ''

        elif video == None and image != None:
            pixel_values = load_image(image)
            pixel_values = pixel_values.unsqueeze(0).to(self.device, dtype=self.dtype)
            image_language_model_inputs = self.model.extract_feature(pixel_values)
            image_list.append(image_language_model_inputs)
            conv.append_message(conv.roles[0], self.image_query + "\n")
            msg = "Received."

            self.image_file = image
            print(f'file path: {self.image_file}')  # image path
            return msg

    def ask(self, text, conv, api_check_box, image):
        if args.use_api:
            # 使用图片 api来生成图片的prompt
            image_api_info = call_apis(tool_without_instruction, api_check_box, image)
            if len(tool_with_instruction) > 0:
                text_api_info = call_apis(tool_with_instruction, api_check_box, image, text)
                image_text_api_info = update_result(image_api_info, text_api_info)
            else:
                image_text_api_info = image_api_info
            text = self.gen_prompt(image_text_api_info, text)  # 生成prompt模板。 每次ask都需要调用api TODO 需要加instruction的api

        conversations = []
        if len(conv.messages) == 0:
            conv.append_message(conv.roles[0], text)
        elif DEFAULT_IMG_START_TOKEN in conv.messages[-1][1] or DEFAULT_VIDEO_START_TOKEN in conv.messages[-1][1]:
            conv.messages[-1][1] = conv.messages[-1][1] + text  #
        else:
            conv.append_message(conv.roles[0], text)

        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        self.conversations = conversations
        return text

    @torch.no_grad()
    def answer(self, conv, video_list, image_list):
        model_inputs = self.tokenizer(
            self.conversations,
            return_tensors="pt",
        )
        print('conversations:', self.conversations)
        model_inputs.pop("token_type_ids", None)

        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        if len(video_list) == 0 and len(image_list) == 0:
            modal_type = 'text'
        else:
            modal_type = 'image'
            if len(video_list) > 0:
                language_model_inputs = video_list[0]
            else:
                language_model_inputs = image_list[0]

        if modal_type == "text":
            generation_output = self.model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        else:
            pixel_values = model_inputs.pop("pixel_values", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)

            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                language_model_inputs=language_model_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )

        preds = generation_output.sequences
        outputs = self.tokenizer.batch_decode(preds, skip_special_tokens=True)[0]

        if modal_type == "text":
            skip_echo_len = len(self.conversations[0]) - self.conversations[0].count("</s>") * 3
            outputs = outputs[skip_echo_len:].strip()
        print('model output:', outputs)
        conv.messages[-1][1] = outputs

        return outputs

    def gen_prompt(self, image_text_api_info, user_request):

        if len(image_text_api_info) > 0:
            new_results = []
            for item in image_text_api_info:
                temp = {}
                temp["tool"] = item["tool"]
                temp["results"] = item['results']
                new_results.append(temp)  # {"tool": "optical_character_recognition", "results": "[拼多多, 上海欢迎您]"}
            if len(new_results) > 0:
                prompt = "请根据图像内容和以下提供的先验知识，详细回答相关问题。\n{{information}}\n\n问题: {{instruction}}\n"
                prompt = prompt.replace("{{information}}", json.dumps(new_results, ensure_ascii=False))
                prompt = prompt.replace("{{instruction}}", user_request)
            else:
                prompt = user_request
        else:
            prompt = user_request
        return prompt

###################### API Setting ######################################
def messages2str(messages):
    m_str = ''
    for m in messages:
        if m.get('role') == "system":
            m_str += '<system>: ' + m['content'] + '\n'
        elif m.get('role') == "user":
            m_str += '<Human>: ' + m['content'] + '\n'
        elif m.get('role') == "assistant":
            m_str += '<Assistant>: ' + m['content'] + '\n'
    return m_str

def _exec(func, arguments_list):
    results = []
    for arguments in arguments_list:
        results.append(func(*arguments))

    return results

def _async_exec(func, arguments_list, max_workers=10, await_result=False):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for arguments in arguments_list:
            results.append(executor.submit(func, *arguments))

    if not await_result:
        return results

    for i, r in enumerate(results):
        results[i] = r.result()

    return results

def batch_exec(func, arguments_list, max_workers=0, await_result=False):
    if max_workers == 0:
        return _exec(func, arguments_list)

    return _async_exec(func, arguments_list, max_workers, await_result)

def send_request(endpoint, api_meta, file_path=None, text=None):
    if endpoint is None or endpoint == '':
        return None
    files = None
    data = None
    if file_path is not None:
        files = {'image': open(file_path, 'rb')}

    if text is not None and len(text) > 0:
        data = {'text': text}
    print(f"logger: sending request to {endpoint}. Input: files = {files}, data = {data}")

    try:
        if 'visual_grounding' in endpoint:
            response = requests.post(endpoint, files=files, data=data, timeout=10)
        else:
            response = requests.post(endpoint, files=files, data=data, timeout=5)
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            # 请求发生异常
            print(f"Request Error {endpoint}: {e}")
            return None

        if response.status_code == 200:
            result = response_data.get("result")
            print(result)
        else:
            # server error
            print(f"Error {endpoint}: {response_data.get('detail')}")
            return None
    except requests.exceptions.RequestException as e:
        # 请求发生异常
        print(f"Request Error {endpoint}: {e}")
        return None

    print(f"logger: result from {endpoint}: {result}")
    tool_name = api_meta['name']
    # print(f'{tool_name}: {api_meta}')
    # print(f"{getattr(sys.modules['__main__'], f'_{tool_name}_wrapper')}")
    if file_path is not None and text is None:
        result = getattr(sys.modules['__main__'], f'_{tool_name}_wrapper')(
            result, file_path
        )

    elif file_path is None and text is not None:
        result = getattr(sys.modules['__main__'], f'_{tool_name}_wrapper')(
            result, text
        )
    elif file_path is not None and text is not None:
        result = getattr(sys.modules['__main__'], f'_{tool_name}_wrapper')(
            result, file_path, text
        )
    return result

def parse_api_config(api_config):
    api_dict = {}
    for item in api_config:
        if not item['accessble']:
            continue
        api_dict[item['name']] = item
    return api_dict

def task_planning(llm, user_request, task_planning_config):
    tool_mem = get_conv_template("husky")
    pass

def call_tool(tool_json, api_dict):
    pass

def cut_memory(conv):
    raise NotImplementedError('cut_memory')

def _visual_grounding_wrapper(result, image, text):
    template_result = {
        'tool': 'visual_grounding',
        'results': [],
    }
    # result is a list
    print(f'type(result): {type(result)}; result: {result}')
    if result == ["No objects to detect"] or result == []:
        return template_result
    class_dict = {}
    for i, item in enumerate(result):
        label = item['label'].strip()
        if label not in class_dict:
            class_dict[label] = []
        bbox = [round(b, 3) for b in item['box']]
        class_dict[label].append(bbox)

    returns = []
    for key, val in class_dict.items():
        for i, bbox in enumerate(val):
            returns.append([f"{key}_{i + 1}: {bbox}"])
    template_result['results'] = returns
    return template_result

def _celebrity_recognition_wrapper(result, image):
    template_result = {
        'tool': 'celebrity_recognition',
        'results': [],
    }

    results = []
    if result is None:
        return template_result
    for item in result:
        bbox = [round(b, 3) for b in item['box']]
        results.append({item['label']: bbox})
    template_result['results'] = results
    return template_result

def _optical_character_recognition_wrapper(result, image):
    template_result = {
        'tool': 'optical_character_recognition',
        'results': [],
    }

    if 'error' in result or 'Error' in result or result is None:
        template_result['results'] = []
    else:
        template_result['results'] = result

    return template_result

def _art_retrieval_wrapper(result, image):
    template_result = {
        'tool': 'art_retrieval',
        'results': result,
    }
    if result is None:
        template_result['results'] = []

    return template_result

def _entity_retrieval_wrapper(result, text):
    template_result = {
        'tool': 'entity_retrieval',
        'results': result,
    }
    return template_result

def _visual_entity_retrieval_wrapper(result, image):
    template_result = {
        'tool': 'visual_entity_retrieval',
        'results': result,
    }
    return template_result

def call_apis(api_dict, api_check_box, image=None, text=None):
    arguments_list = []

    for api, api_meta in api_dict.items():
        # print(f"api: {api}; api_meta: {api_meta}")
        endpoint = api_meta.get('endpoint', None)
        if endpoint == '' or endpoint is None:
            continue

        # 是否使用api TODO
        if api not in api_check_box:
            continue

        args = [arg['type'] for arg in api_meta['args']]
        if 'image' in args:
            input_image = image
        else:
            input_image = None
        if 'text' in args:
            input_text = text
        else:
            input_text = None
        arguments_list.append((endpoint, api_meta, input_image, input_text))

    # '''
    results = batch_exec(
        send_request, arguments_list, max_workers=6, await_result=True
    )
    new_results = []
    for r in results:
        if r is not None:
            new_results.append(r)

    return new_results

def update_result(resource_info, new_infos):
    existing_tool = [item['tool'] for item in resource_info]
    for item in new_infos:
        idx = -1
        if item['tool'] in existing_tool:
            idx = existing_tool.index(item['tool'])

        if idx != -1:
            resource_info[idx] = item
        else:
            resource_info.append(item)
    return resource_info

###################### API Setting ######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=17615)
    parser.add_argument('--model_path', type=str,
                        default='/mnt/afs/share_data/tongwenwen/models/husky_models/multi_model/mmhusky_v2_21b_stage4_v1.0_fp16')
    parser.add_argument("--api_config", type=str,
                        default='/mnt/afs/user/tongwenwen/workspace/codes/husky_tools/Husky/configs/api_st_public.json')
    parser.add_argument('--gradio_head', type=str, default='husky-13b')
    parser.add_argument('--model_type', type=str, default='13b')
    parser.add_argument('--use_api', action='store_true', default=False, help="use the api info")
    parser.add_argument('--output_api_info', action='store_true', default=False, help="output the prompt info")
    parser.add_argument('--use_root_path', action='store_true', default=False,
                        help="use root path for nginx port transfer")
    parser.add_argument('--root_path', type=str, default="husky", help="output the prompt info")

    args = parser.parse_args()
    # model info
    if args.model_type == '13b':
        print('model size: 13b')
        from husky.model.modeling_husky_embody import HuskyForConditionalGeneration
    elif args.model_type == '20b':
        print('model size: 20b')
        from husky.model.modeling_husky_2 import HuskyForConditionalGeneration
    elif args.model_type == 'vit_e':
        print('model size: vit_e')
        from husky.model.modeling_husky_vit_E import HuskyForConditionalGeneration
    elif args.model_type == 'internlm20b':
        print('model size:', args.model_type)
        from husky.model.modeling_husky_intern import HuskyForConditionalGeneration

    # api 接受
    api_config = json.load(open(args.api_config, 'r'))
    api_dict = parse_api_config(api_config)
    tool_without_instruction = {}
    tool_with_instruction = {}
    for api, meta in api_dict.items():
        if 'text' in [arg['type'] for arg in meta['args']]:
            tool_with_instruction[api] = meta  # 含有text输入的tool
        else:
            tool_without_instruction[api] = meta  # 没有text输入的tool

    # output_api_info
    if args.use_api and args.output_api_info:
        print('output_api_info')
        print('number of api tool_without_instruction:', len(tool_without_instruction))
        print('number of api tool_with_instruction:', len(tool_with_instruction))

    # root_path
    root_path = '/' + args.root_path
    print('use root path:', args.use_root_path)
    print('root_path:', root_path)  # todo: 使用nginx转发gradio需要设置这个

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chat = Chat(args.model_path, device=device, num_gpus=1, max_new_tokens=1024, load_8bit=False)
    title = f'<h1 align="center">Husky: {args.gradio_head} </h1>'
    print('title:', title)
    description = """<h2 align="center">This is the demo of Husky supporting Chinese and English. Upload your image or video and start chatting!</h2>"""
    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=0.5):
                with gr.Tab("Image"):
                    image = gr.Image(type='filepath', interactive=False)  # 使用filepath
                with gr.Tab("Video"):
                    video = gr.Video(type='filepath', label="Video", interactive=False)
                with gr.Row():
                    upload_button = gr.Button(value="Upload Video/Image", interactive=False, variant="primary")
                    clear = gr.Button("Restart", interactive=False)
                    setup = gr.Button("Setup", interactive=True)
            with gr.Column():
                chat_state = gr.State()
                video_list = gr.State()
                image_list = gr.State()
                # api state session
                if args.output_api_info:
                    # api_check_box = gr.CheckboxGroup(["OCR", "名人", "艺术品", "Grounding", "视觉检索", "名词检索"], label="api_tools")
                    api_check_box = gr.CheckboxGroup(
                        ["optical_character_recognition", "celebrity_recognition", "art_retrieval",
                         "visual_grounding", "visual_entity_retrieval", "entity_retrieval"], label="api_tools")

                    text_api_prompt = gr.Textbox(label='api_prompt_output', interactive=False)
                    image_text_api_info = gr.State()

                chatbot = gr.Chatbot(label="Husky")
                text_input = gr.Textbox(label='User', placeholder='Type and press Enter', interactive=False)
                clear_text_box = gr.Button("Clear text box")
        gr.Markdown("**User Manual:**")
        gr.Markdown("1. **text-modal dialogue**: Click “Setup” => Chat with Husky")
        gr.Markdown(
            "2. **image-text-modal dialogue**: Click “Setup” => Select 'Image' tab => Drop image and click “Upload Video/Image” => Chat with Husky")
        gr.Markdown(
            "3. **video-text-modal dialogue**: Click “Setup” => Select 'Video' tab => Drop video and click “Upload Video/Image” => Chat with Husky")

        if args.use_api and args.output_api_info:
            setup.click(start, [],
                        [video, image, upload_button, clear, setup, text_input, text_api_prompt, chat_state, video_list,
                         image_list])
            upload_button.click(upload_file, [video, image, chat_state],
                                [video, image, upload_button, chat_state, video_list, image_list])

            text_input.submit(gradio_ask, [text_input, chatbot, chat_state, api_check_box, image],
                              [text_input, chatbot, chat_state, text_api_prompt]).then(
                gradio_answer, [chat_state, chatbot, video_list, image_list],
                [chatbot, chat_state, video_list, image_list])

            clear.click(gradio_reset, [chat_state, video_list, image_list],
                        [chatbot, video, image, text_input, text_api_prompt, upload_button, chat_state, video_list,
                         image_list], queue=False)
            clear_text_box.click(gradio_reset_text_box, [chat_state],
                                 [chatbot, text_input, text_api_prompt, chat_state, api_check_box], queue=False)
        else:  # useless
            setup.click(start, [],
                        [video, image, upload_button, clear, setup, text_input, chat_state, video_list, image_list])
            upload_button.click(upload_file, [video, image, chat_state],
                                [video, image, upload_button, chat_state, video_list, image_list])

            text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
                gradio_answer, [chat_state, chatbot, video_list, image_list],
                [chatbot, chat_state, video_list, image_list])

            clear.click(gradio_reset, [chat_state, video_list, image_list],
                        [chatbot, video, image, text_input, upload_button, chat_state, video_list, image_list],
                        queue=False)
            clear_text_box.click(gradio_reset_text_box, [chat_state], [chatbot, text_input, chat_state], queue=False)

    if args.use_root_path:
        demo.launch(server_name="0.0.0.0", server_port=args.port, root_path=root_path)
    else:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
