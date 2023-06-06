import os
import gradio as gr
import clueai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse

params = argparse.ArgumentParser()
params.add_argument('-float', help="使用 float，当显卡不支持 half 时可以使用，但显存占用量会增加", action="store_true")
params = params.parse_args()

tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v2")
model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2")

# 使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 当硬件不支持 half 时使用 float
# When hardware does not support the half-precision data type Half, you can use the single-precision data type Float
if not params.float:
    model.half()

base_info = ""


def preprocess(text):
    text = f"{base_info}{text}"
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t").replace(
        '%20', '  ')  #.replace(" ", "&nbsp;")


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
    #out=model.generate(**encoding, **generate_config)
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
    #print(context)

    input_text = context + "\n用户：" + input + "\n小元："
    input_text = input_text.strip()
    output_text = answer(input_text, top_p, temperature)
    print("open_model".center(20, "="))
    print(f"{input_text}\n{output_text}")
    #print("="*20)
    history.append((input, output_text))
    #print(history)
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
    #print(context)

    input_text = context + "\n用户：" + input + "\n小元："
    input_text = input_text.strip()
    output_text = answer(input_text, top_p, temperature)
    print("open_model".center(20, "="))
    print(f"{input_text}\n{output_text}")
    history.append((input, output_text))
    #print(history)
    return '', history, history


block = gr.Blocks()

with block as demo:
    gr.Markdown("""<h1><center>聊天机器人</center></h1>
        <font size=4>欢迎与聊天机器人聊天</font>
        <font size=4>注意：gradio对markdown代码格式展示有限</font>
    """)
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label='聊天机器人').style(height=400)

        with gr.Column(scale=1):

            num = gr.Slider(minimum=4,
                            maximum=10,
                            label="最大的对话轮数",
                            value=5,
                            step=1)
            top_p = gr.Slider(minimum=0,
                              maximum=1,
                              label="top_p",
                              value=1,
                              step=0.1)
            temperature = gr.Slider(minimum=0,
                                    maximum=1,
                                    label="temperature",
                                    value=0.7,
                                    step=0.1)
            clear_history = gr.Button("👋 清除历史对话 | Clear History")
            send = gr.Button("🚀 发送 | Send")
            regenerate = gr.Button("🚀 重新生成本次结果 | regenerate")
    message = gr.Textbox()
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
    #print('prediction: {}'.format(prediction.generations[0].text))
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


block = gr.Blocks()

with block as demo_1:
    gr.Markdown("""<h1><center>聊天机器人</center></h1>
    <font size=4>回答来自ChatYuan, 以上是模型生成的结果, 请谨慎辨别和参考, 不代表任何人观点  | Answer generated by ChatYuan model</font>
    <font size=4>注意：gradio对markdown代码格式展示有限</font>
    <font size=4>在使用此功能前，你需要有个API key. API key 可以通过这个<a href='https://www.clueai.cn/' target="_blank">平台</a>获取</font>
    """)
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label='ChatYuan').style(height=400)

        with gr.Column(scale=1):
            api_key = gr.inputs.Textbox(label="请输入你的api-key(必填)",
                                        default="",
                                        type='password')
            num = gr.Slider(minimum=4,
                            maximum=10,
                            label="最大的对话轮数",
                            value=5,
                            step=1)
            top_p = gr.Slider(minimum=0,
                              maximum=1,
                              label="top_p",
                              value=1,
                              step=0.1)
            clear_history = gr.Button("👋 清除历史对话 | Clear History")
            send = gr.Button("🚀 发送 | Send")

    message = gr.Textbox()
    state = gr.State()
    message.submit(chatyuan_bot_api,
                   inputs=[api_key, message, state, top_p, num],
                   outputs=[message, chatbot, state])

    send.click(chatyuan_bot_api,
               inputs=[api_key, message, state, top_p, num],
               outputs=[message, chatbot, state])
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[chatbot, state],
                        queue=False)

block = gr.Blocks()
with block as introduction:
    gr.Markdown("""<h1><center>聊天机器人</center></h1>
    
<font size=4>😉ChatYuan: General Model for Dialogue with ChatYuan
<br>
<br>
<br>
<br>
<br>
<br>
Based on the original functions of Chatyuan-large-v1, we optimized the model as follows:
-Added the ability to speak in both Chinese and English.
-Added the ability to refuse to answer. Learn to refuse to answer some dangerous and harmful questions.
-Added code generation functionality. Basic code generation has been optimized to a certain extent.
-Enhanced basic capabilities. The original contextual Q&A and creative writing skills have significantly improved.
-Added a table generation function. Make the generated table content and format more appropriate.
-Enhanced basic mathematical computing capabilities.
-The maximum number of length tokens has been expanded to 4096.
-Enhanced ability to simulate scenarios< br>
<br>
👀<a href='https://www.cluebenchmarks.com/clueai.html'>PromptCLUE-large</a>在1000亿token中文语料上预训练, 累计学习1.5万亿中文token, 并且在数百种任务上进行Prompt任务式训练. 针对理解类任务, 如分类、情感分析、抽取等, 可以自定义标签体系; 针对多种生成任务, 可以进行采样自由生成.  <br> 
<br>
 &nbsp; <a href='https://modelscope.cn/models/ClueAI/ChatYuan-large/summary' target="_blank">ModelScope</a> &nbsp; | &nbsp; <a href='https://huggingface.co/ClueAI/ChatYuan-large-v1' target="_blank">Huggingface</a> &nbsp; | &nbsp; <a href='https://www.clueai.cn' target="_blank">官网体验场</a> &nbsp; | &nbsp; <a href='https://github.com/clue-ai/clueai-python#ChatYuan%E5%8A%9F%E8%83%BD%E5%AF%B9%E8%AF%9D' target="_blank">ChatYuan-API</a> &nbsp; | &nbsp; <a href='https://github.com/clue-ai/ChatYuan' target="_blank">Github项目地址</a> &nbsp; | &nbsp; <a href='https://openi.pcl.ac.cn/ChatYuan/ChatYuan/src/branch/main/Fine_tuning_ChatYuan_large_with_pCLUE.ipynb' target="_blank">OpenI免费试用</a> &nbsp;
</font>
<center><a href="https://clustrmaps.com/site/1bts0"  title="Visit tracker"><img src="//www.clustrmaps.com/map_v2.png?d=ycVCe17noTYFDs30w7AmkFaE-TwabMBukDP1802_Lts&cl=ffffff" /></a></center>
    """)

gui = gr.TabbedInterface(
    interface_list=[demo],
    tab_names=["聊天机器人"])
gui.launch(quiet=True, show_api=False, share=True)
