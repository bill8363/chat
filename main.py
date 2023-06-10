import os
import gradio as gr
import clueai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "./ChatYuan-large-v2"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


# 使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.half()

base_info = ""


def preprocess(text):
    text = f"{base_info}{text}"
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t").replace(
        '%20', '  ')  # .replace(" ", "&nbsp;")


generate_config = {
    'do_sample': True,
    'top_p': 0.9,
    'top_k': 50,
    'temperature': 0.7,
    'num_beams': 1,
    'max_length': 1024,
    'min_length': 3,
    'no_repeat_ngram_size': 5,
    'length_penalty': 0.6,
    'return_dict_in_generate': True,
    'output_scores': True
}


def answer(
        text,
        top_p,
        temperature,
        sample=True,
):
    '''sample：是否抽样。生成任务，可以设置为True;
  top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text],
                         truncation=True,
                         padding=True,
                         max_length=1024,
                         return_tensors="pt").to(device)
    if not sample:
        out = model.generate(**encoding,
                             return_dict_in_generate=True,
                             output_scores=False,
                             max_new_tokens=1024,
                             num_beams=1,
                             length_penalty=0.6)
    else:
        out = model.generate(**encoding,
                             return_dict_in_generate=True,
                             output_scores=False,
                             max_new_tokens=1024,
                             do_sample=True,
                             top_p=top_p,
                             temperature=temperature,
                             no_repeat_ngram_size=12)
    # out=model.generate(**encoding, **generate_config)
    out_text = tokenizer.batch_decode(out["sequences"],
                                      skip_special_tokens=True)
    return postprocess(out_text[0])


def clear_session():
    return '', None


def chatyuan_bot(input, history, top_p, temperature, num):
    history = history or []
    if len(history) > num:
        history = history[-num:]

    context = "\n".join([
        f"用户：{input_text}\n小元：{answer_text}"
        for input_text, answer_text in history
    ])
    # print(context)

    input_text = context + "\n用户：" + input + "\n小元："
    input_text = input_text.strip()
    output_text = answer(input_text, top_p, temperature)
    print("open_model".center(20, "="))
    print(f"{input_text}\n{output_text}")
    # print("="*20)
    history.append((input, output_text))
    # print(history)
    return '', history, history


def chatyuan_bot_regenerate(input, history, top_p, temperature, num):
    history = history or []

    if history:
        input = history[-1][0]
        history = history[:-1]

    if len(history) > num:
        history = history[-num:]

    context = "\n".join([
        f"用户：{input_text}\n小元：{answer_text}"
        for input_text, answer_text in history
    ])
    # print(context)

    input_text = context + "\n用户：" + input + "\n小元："
    input_text = input_text.strip()
    output_text = answer(input_text, top_p, temperature)
    print("open_model".center(20, "="))
    print(f"{input_text}\n{output_text}")
    history.append((input, output_text))
    # print(history)
    return '', history, history


block = gr.Blocks()

with block as demo:

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label='聊天内容').style(height=400)

        with gr.Column(scale=1):
            num = gr.Slider(minimum=4,
                            maximum=20,
                            label="最大的对话轮数",
                            value=5,
                            step=1)
            top_p = gr.Slider(minimum=0,
                              maximum=1,
                              label="模型多样性",
                              value=1,
                              step=0.1)
            temperature = gr.Slider(minimum=0,
                                    maximum=1,
                                    label="模型想象力",
                                    value=0.7,
                                    step=0.1)
            clear_history = gr.Button("👋 清除历史对话 | Clear History")
            send = gr.Button("🚀 发送 | Send")
            regenerate = gr.Button("🚀 重新生成本次结果 | regenerate")
    message = gr.Textbox(label='对话输入区')
    state = gr.State()
    message.submit(chatyuan_bot,
                   inputs=[message, state, top_p, temperature, num],
                   outputs=[message, chatbot, state])
    regenerate.click(chatyuan_bot_regenerate,
                     inputs=[message, state, top_p, temperature, num],
                     outputs=[message, chatbot, state])
    send.click(chatyuan_bot,
               inputs=[message, state, top_p, temperature, num],
               outputs=[message, chatbot, state])

    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[chatbot, state],
                        queue=False)


def ChatYuan(api_key, text_prompt, top_p):
    generate_config = {
        "do_sample": True,
        "top_p": top_p,
        "max_length": 128,
        "min_length": 10,
        "length_penalty": 1.0,
        "num_beams": 1
    }
    cl = clueai.Client(api_key, check_api_key=True)
    # generate a prediction for a prompt
    # 需要返回得分的话，指定return_likelihoods="GENERATION"
    prediction = cl.generate(model_name='ChatYuan-large', prompt=text_prompt)
    # print the predicted text
    # print('prediction: {}'.format(prediction.generations[0].text))
    response = prediction.generations[0].text
    if response == '':
        response = "很抱歉，我无法回答这个问题"

    return response


def chatyuan_bot_api(api_key, input, history, top_p, num):
    history = history or []

    if len(history) > num:
        history = history[-num:]

    context = "\n".join([
        f"用户：{input_text}\n小元：{answer_text}"
        for input_text, answer_text in history
    ])

    input_text = context + "\n用户：" + input + "\n小元："
    input_text = input_text.strip()
    output_text = ChatYuan(api_key, input_text, top_p)
    print("api".center(20, "="))
    print(f"api_key:{api_key}\n{input_text}\n{output_text}")

    history.append((input, output_text))

    return '', history, history

gui = gr.TabbedInterface(
    interface_list=[demo],
    tab_names=["作业展示"])
gui.launch(quiet=True, show_api=False, share=True)